#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Running the more complex resilient simulation'''
import os
os.environ["OMP_NUM_THREADS"] = "8"

import remove_d

from random import random
from typing import List, Literal, Optional
from dataclasses import dataclass, field

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
from scipy.linalg import block_diag
import cProfile

import networkx as nx

import info_robust_graph as irg
from adaptive_step import AdaptiveStepController, AdaptiveStepConfig, create_controller

@dataclass
class simulation_config:
    '''A container just to hold simulation parameters'''
    step_size: float = 1/40.0
    num_iter: int = 1_000
    num_rounds: int = 1
    init_state: np.ndarray = None
    # Adaptive step size configuration
    step_method: Literal['constant', 'adagrad', 'rmsprop', 'adam', 'amsgrad', 'nadam', 'accelerated'] = 'constant'
    adaptive_base_lr: float = 0.1
    # Accelerated GRANE (Nesterov-style): momentum for state extrapolation
    accelerated_momentum: float = 0.9
    num_iter: int = 1000
    graph_switch_period: int = 100
    seed: int = 420
    # Projection settings: 'none', 'box', 'ball'
    projection: Literal['none', 'box', 'ball'] = 'none'
    projection_params: dict = field(default_factory=dict)

def action_select_matrix(dim_action_list: List[int]) -> np.ndarray:
    '''Given the dim of each agent's action return the action select matrix, i.e., the R matrix'''
    dim_action = sum(dim_action_list)
    num_agents = len(dim_action_list)
    dim_state = dim_action * num_agents

    mtx = np.zeros([dim_action, dim_state])

    row, col = 0, 0
    for dim in dim_action_list:
        mtx[row:row + dim, col:col + dim] = np.eye(dim)
        row, col = row + dim, col + dim + dim_action

    return mtx

def action_mask(dim_action_list: List[int]) -> np.ndarray:
    '''Returns a matrix that sets the estimate components to 0'''
    mtx = action_select_matrix(dim_action_list)
    return mtx.transpose().dot(mtx)

def save_plot(figure, name):
    '''Saves the figure as a pdf'''
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    figure.savefig(f"{name}.pdf", format='pdf', bbox_inches='tight', pad_inches=0.01)

class PlotErrorFigure(object):
    ''' Context Manager for plotting the error figures '''
    def __init__(self, name):
        self.name = name
        self.figure = None
    def __enter__(self):
        self.figure = plt.figure()
        return self.figure
    def __exit__(self, exc_type, exc_value, exc_traceback):
        plt.yscale("log")
        plt.xlabel('Iteration')
        plt.ylabel(r'$||x_{k} - x^{*}||$')
        plt.show()
        save_plot(self.figure, self.name)
        plt.close()

def adjlist_to_csr(adj_list_gc, N):
    offsets = np.zeros(N + 1, dtype=np.int64)
    neighbors = []
    for i in range(N):
        offsets[i+1] = offsets[i] + len(adj_list_gc[i])
        neighbors.extend(adj_list_gc[i])
    return offsets, np.asarray(neighbors, dtype=np.int64)


class Resilient:
    """Simulation for the resilient algorithm"""
    def __init__(self, sim_config, grid_width, random_agents=None, constant_agents=None, l_inf_ball = 1, D = 1, corner_size = 1):
        self.sim_config = sim_config        
        self.grid_width = grid_width
        self.corner_size = corner_size

        self.dim_action_i = 2
        self.N = grid_width**2 - 4*corner_size**2
        self.dim_action_list = self.dim_action_i * np.ones(self.N, dtype=int)
        self.dim_action = self.dim_action_i * self.N
        self.dim_state = self.N * self.dim_action
        self.random_agents = random_agents if random_agents else set()
        self.constant_agents = constant_agents if constant_agents else set()

        self.D = D

        self.Gc = irg.grid_l_inf_to_adj_matrix(grid_width, l_inf_ball)
        self.corners = irg.get_corners(self.Gc, corner_size)
        self.Gc = irg.remove_nodes_from_adj_matrix(self.Gc, self.corners)
        self.Go = self.Gc + np.eye(self.N, dtype=int)
        self.adj_list_gc = irg.adj_matrix_to_adj_in_set(self.Gc, self_loop=False)
        self.offsets, self.neighbors = adjlist_to_csr(self.adj_list_gc, self.N)


        self.R = action_select_matrix(self.dim_action_list)
        self.A, self.b = self.get_gradient()

        self.F = block_diag(*[self.A[2*i:2*(i+1),:] for i in range(self.N)])
        self.RB = self.R.transpose().dot(self.b)
        self.NE = -np.linalg.inv(self.A).dot(self.b)

        self.example = 'position_plot'

        self.l_inf_ball = l_inf_ball
        self.graph_switch_period = sim_config.graph_switch_period  # e.g., 1000 or None
        self.rng = np.random.default_rng(420)
        
        # Initialize adaptive step controller
        self.step_controller = None
        self._init_step_controller()

    def update_graph(self, iteration, desired_kappa=1, max_tries=50):
        """
        Switches between Grid and Ring topologies while ensuring info-robustness.
        """
        def generate_ring_adj_matrix(num_nodes, num_jumps):
            """
            Generates a random circulant graph.
            - num_jumps: number of unique 'distances' each node connects to.
            """
            adj = np.zeros((num_nodes, num_nodes), dtype=int)
            
            # 1. Pick unique random jump distances (offsets)
            # We pick from 1 to floor(N/2) to avoid redundant edges in undirected graphs
            possible_jumps = list(range(1, num_nodes // 2))
            jumps = self.rng.choice(possible_jumps, size=num_jumps, replace=False)
            
            # 2. Build the basic circulant structure
            for i in range(num_nodes):
                for s in jumps:
                    j_left = (i - s) % num_nodes
                    j_right = (i + s) % num_nodes
                    adj[i, j_left] = 1
                    adj[i, j_right] = 1
                    
            # 3. Randomize the "look" by shuffling node indices (Graph Isomorphism)
            perm = np.random.permutation(num_nodes)
            adj = adj[perm, :]
            adj = adj[:, perm]
            
            return adj

        for attempt in range(max_tries):
            # Decide randomly: 0 for Grid, 1 for Ring
            graph_type = self.rng.choice(['grid', 'ring', 'ring'])
            
            if graph_type == 'grid':
                r = self.rng.choice([1, 2])
                G_full = irg.grid_l_inf_to_adj_matrix(self.grid_width, r)
                Gc = irg.remove_nodes_from_adj_matrix(G_full, self.corners)
            
            else: # Ring Topology
                # Start with a radius that has a chance of meeting kappa
                # A ring with radius 'r' is 2r-connected
                min_radius = int(np.ceil(desired_kappa / 2))
                r = self.rng.integers(min_radius, min_radius + 2)
                Gc = generate_ring_adj_matrix(self.N, r)

            # Ensure undirected and no self-loops for the robustness check
            Gc = np.maximum(Gc, Gc.T)
            np.fill_diagonal(Gc, 0)

            # Check info robustness
            kappa = irg.get_k_info_robust(Gc, Gc)
            if kappa >= desired_kappa:
                print(f"[iter {iteration}] Accepted {graph_type} graph (kappa={kappa})")
                self.Gc = Gc
                self.Go = Gc + np.eye(self.N)
                self.adj_list_gc = irg.adj_matrix_to_adj_in_set(self.Gc, self_loop=False)
                self.offsets, self.neighbors = adjlist_to_csr(self.adj_list_gc, self.N)
                return

        raise RuntimeError(f"Failed to find a {desired_kappa}-robust graph after {max_tries} tries.")

    def visualize_graph(self, iteration: int):
        """Generates a PDF plot of the current communication graph using the iteration count."""
        # 1. Convert adjacency matrix to a NetworkX graph
        G = nx.from_numpy_array(self.Gc)
        
        # 2. Compute physical grid positions based on your grid logic
        pos = {}
        node_idx = 0
        end = self.grid_width - 1
        
        for row in range(self.grid_width):
            for col in range(self.grid_width):
                is_corner = (
                    (row < self.corner_size and col < self.corner_size) or
                    (row < self.corner_size and col > end - self.corner_size) or
                    (row > end - self.corner_size and col < self.corner_size) or
                    (row > end - self.corner_size and col > end - self.corner_size)
                )
                
                if not is_corner:
                    # Map grid to plot coords: y = row (not end - row) so vertical matches
                    # the second action component / trajectory y vs matplotlib y-up.
                    pos[node_idx] = (col, row)
                    node_idx += 1

        # 3. Create the figure
        fig = plt.figure(figsize=(7, 7))
        plt.title(f"Communication Graph - Iteration {iteration}")
        
        # 4. Identify adversarial agents for color-coding
        adversarial_nodes = set(self.random_agents) | set(self.constant_agents)
        node_colors = ['#ff7f0e' if node in adversarial_nodes else '#1f77b4' for node in G.nodes()]
        
        # 5. Draw
        nx.draw(G, pos, 
                with_labels=True, 
                node_color=node_colors, 
                node_size=500, 
                edge_color='gray',
                alpha=0.8,
                font_size=9,
                font_color='white')
        
        # 6. Save with zero-padded iteration for easy sorting (e.g., graph_0100.pdf)
        filename = f"graph_plots/graph_iter_{iteration:04d}.pdf"
        fig.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close(fig)

    def _init_step_controller(self):
        """Initialize the adaptive step size controller based on config"""
        if self.sim_config.step_method in ('constant', 'accelerated'):
            self.step_controller = None
        else:
            self.step_controller = create_controller(
                dim_state=self.dim_state,
                method=self.sim_config.step_method,
                base_lr=self.sim_config.adaptive_base_lr
            )
    
    def set_step_method(self, method: str, base_lr: float = 0.1, momentum: Optional[float] = None):
        """
        Change the step size method dynamically.
        
        Parameters
        ----------
        method : str
            One of 'constant', 'adagrad', 'rmsprop', 'adam', 'amsgrad', 'nadam', 'accelerated'
        base_lr : float
            Base learning rate for adaptive methods; for 'accelerated' this is the step size.
        momentum : float, optional
            For 'accelerated' only: Nesterov momentum (default from config, typically 0.9).
        """
        self.sim_config.step_method = method
        self.sim_config.adaptive_base_lr = base_lr
        if momentum is not None and method == 'accelerated':
            self.sim_config.accelerated_momentum = momentum
        self._init_step_controller()

    def block_diag(self):
        ''' Block diag the matrix A to create F '''
        answer = np.zeros([self.dim_action, self.dim_state])

        for i in range(self.N):
            answer[self.dim_action_i*i:self.dim_action_i*(i+1),self.dim_action*i:self.dim_action*(i+1)] = self.A[self.dim_action_i*i:self.dim_action_i*(i+1),:]

        return answer

    def get_gradient(self):
        ''' Returns the gradient for the cost function '''
        local_cost = irg.grid_l_one_to_adj_matrix(self.grid_width, 1)
        local_cost = irg.remove_nodes_from_adj_matrix(local_cost, self.corners)
       
        grad_rel_position = irg.laplacian_from_adj_mtx(local_cost, self.dim_action_i)
        grad_average = ((1/self.N)**2)*np.kron(np.ones([self.N,self.N]),np.eye(self.dim_action_i))
        A = grad_average + grad_rel_position

        L = np.array([irg.grid_node_has_left(self.grid_width, self.corner_size)]).transpose()
        R = np.array([irg.grid_node_has_right(self.grid_width, self.corner_size)]).transpose()
        U = np.array([irg.grid_node_has_up(self.grid_width, self.corner_size)]).transpose()
        D = np.array([irg.grid_node_has_down(self.grid_width, self.corner_size)]).transpose()

        cost_x_rel = R - L
        cost_y_rel = U - D
        b = np.kron(cost_x_rel, np.array([[1],[0]])) + np.kron(cost_y_rel, np.array([[0],[1]]))

        return A, b

    def remove_extreme_D_average(self, received_values, agent_value):
        '''Filter that removes the extreme messages received'''

        # TODO: Re write this function using multiprocessing library.
        # This sidesteps the issues with the GIL to allow proper 
        # parallelism that can't be done with threads in python.
        # This method is slow because this is converting everything into 
        # matrix operations but the matrix is massive. Probably faster to
        # just have each agent have there own process and compute the update
        # instead of trying to use numpy/matrix computation for speed up.
        # Especially since this part of the code can't be written as a 
        # matrix operation.

        received_values.sort()
        max_index = len(received_values)-1
        low_index = 0
        high_index = max_index

        for j_highest in range(self.D):
            if received_values[max_index - j_highest] >= agent_value:
                high_index = high_index-1
            else:
                break
        for j_lowest in range(self.D):
            if received_values[j_lowest] <= agent_value:
                low_index = low_index+1
            else:
                break

        # Note: Using numpy sum, and mean is acutally slower than doing this?!
        average = agent_value
        for index in range(low_index, high_index + 1):
            average += received_values[index]
        average = average/(high_index - low_index + 2)
        return average

    def filter_communicated_message(self, state_y):
        state_v = np.zeros([self.dim_state,1])

        for agent_i in range(self.N):
            for state_j in range(self.N):
                for component_k in range(self.dim_action_i):
                    offset = self.dim_action_i*state_j+component_k
                    state_index_i = self.dim_action*agent_i + offset
                    if self.Go[agent_i, state_j] == 1:
                        state_index_j = self.dim_action*state_j + offset
                        state_v[state_index_i] = state_y[state_index_j,state_j]
                    else:
                        agent_i_in_messages = [ state_y[self.dim_action*X + offset, agent_i] for X in self.adj_list_gc[agent_i]]
                        #state_v[state_index_i] = remove_d.remove_extreme_D_average(agent_i_in_messages, state_y[state_index_i,agent_i], self.D)
                        state_v[state_index_i] = self.remove_extreme_D_average(agent_i_in_messages, state_y[state_index_i,agent_i])

        return state_v

    def adversarial_communication(self, state_x):
        dim = self.dim_action
        state_y = np.kron(state_x, np.ones([1,self.N]))

        for agent in self.random_agents:
            state_y[agent*dim:(agent + 1)*dim,:] = state_y[agent*dim:(agent + 1)*dim,:] + np.random.normal(0,1,[dim,self.N])
            state_y[agent*dim:(agent + 1)*dim,agent] = state_x[agent*dim:(agent+1)*dim].transpose()[0]
        for agent in self.constant_agents:
            state_y[agent*dim:(agent + 1)*dim,:] = np.ones([dim,self.N])
            state_y[agent*dim:(agent + 1)*dim,agent] = state_x[agent*dim:(agent+1)*dim].transpose()[0]

        return state_y

    def project_state(self, state_x: np.ndarray) -> np.ndarray:
        method = getattr(self.sim_config, 'projection', 'none')
        params = getattr(self.sim_config, 'projection_params', {}) or {}
        
        # Configuration: 32 agents, each with 64 dimensions
        num_agents = (self.grid_width ** 2 - 4)
        dim_per_agent = num_agents * 2

        if method == 'none':
            return state_x

        if method == 'box':
            low = params.get('low', None)
            high = params.get('high', None)
            if low is None or high is None:
                return state_x
            
            # Reshape to (32, 64) so boundaries match the second dimension
            reshaped_x = state_x.reshape(num_agents, dim_per_agent)
            
            # Clip will now broadcast correctly across the 32 agents
            clipped_x = np.clip(reshaped_x, low.flatten(), high.flatten())
            
            # Return to original shape (2048, 1)
            return clipped_x.reshape(state_x.shape)

        if method == 'ball':
            radius = float(params.get('radius', 1.0))
            center = params.get('center', None)
            if center is None:
                center = np.zeros(2) # (x, y) center
            else:
                center = np.asarray(center).flatten()[:2]

            # 1. Reshape to (32 agents, 32 beliefs, 2 coordinates)
            # This isolates every single (x, y) pair in the entire simulation
            reshaped_x = state_x.reshape(num_agents, 32, 2)
            
            # 2. Calculate the relative vector from center for every (x, y) pair
            diff = reshaped_x - center # Broadcasting (32, 32, 2) - (2,)
            
            # 3. Calculate norm for every single belief point (axis=2)
            # Result shape: (32, 32, 1)
            norms = np.linalg.norm(diff, axis=2, keepdims=True)
            
            # 4. Calculate scaling factor for every single belief point
            # If any specific (x, y) pair is outside the radius, scale it back
            scale = np.where(norms > radius, radius / np.maximum(norms, 1e-9), 1.0)
            
            # 5. Apply the projection to the differences
            projected_reshaped = center + (diff * scale)
            
            # 6. Return to the original flat (2048, 1)
            return projected_reshaped.reshape(state_x.shape)

        return state_x


    def iterate_algo(self, init_state, verbose: bool = True):
        """
        Run the resilient Nash equilibrium seeking algorithm.
        
        Parameters
        ----------
        init_state : np.ndarray
            Initial state vector
        verbose : bool
            Whether to print progress updates
        
        Returns
        -------
        tuple
            (error_records, position_records, graph_history, final_state)
            graph_history is a list of (iteration, Gc_copy) tuples recorded
            at initialization and each topology switch.
        """
        state_x = init_state
        if verbose:
            print('Starting iteration with method:', self.sim_config.step_method)
            print('Initial state norm:', np.linalg.norm(state_x))
        
        records = [np.linalg.norm(self.NE-self.R.dot(state_x),2)]
        pos_records = [self.R.dot(state_x)]
        graph_history = [(0, self.Gc.copy())]
        
        # Reset adaptive controller if using one
        if self.step_controller is not None:
            self.step_controller.reset()
        
        # For accelerated GRANE: Nesterov extrapolation on filtered state
        state_v_prev = None
        alpha = self.sim_config.adaptive_base_lr if self.sim_config.step_method == 'accelerated' else self.sim_config.step_size
        beta = self.sim_config.accelerated_momentum

        # visualize the communication graph
        self.visualize_graph(0)
        
        for i in range(self.sim_config.num_iter):
            if verbose and (i%100) == 0:
                print(f"Iteration {i} of {self.sim_config.num_iter}, error: {records[-1]:.6f}")
            
            state_y = self.adversarial_communication(state_x)
            state_v = remove_d.filter_communicated_message(state_y, self.Go, self.offsets, self.neighbors, self.D)

            # Change the graph topology and visualize in the form of a pdf
            if self.graph_switch_period and i > 0 and (i % self.graph_switch_period == 0):
                self.update_graph(i, desired_kappa=2 * self.D + 1)
                self.visualize_graph(i)
                graph_history.append((i, self.Gc.copy()))

            if self.sim_config.step_method == 'accelerated':
                # Accelerated GRANE: extrapolate filtered state, gradient at extrapolated point, then update
                if state_v_prev is None:
                    y = state_v
                else:
                    y = state_v + beta * (state_v - state_v_prev)
                # Gradient at extrapolated state y (no extra communication)
                temp1 = self.F.dot(y)
                temp2 = self.R.transpose().dot(temp1)
                gradients = temp2 + self.RB
                state_x = y - alpha * gradients
                # Project onto feasible set if configured
                state_x = self.project_state(state_x)
                state_v_prev = state_v
            else:
                # Compute gradient at current filtered state
                temp1 = self.F.dot(state_v)
                temp2 = self.R.transpose().dot(temp1)
                gradients = temp2 + self.RB
                # Apply step size (constant or adaptive)
                if self.step_controller is None:
                    state_x = state_v - self.sim_config.step_size * gradients
                    # Project onto feasible set if configured
                    state_x = self.project_state(state_x)
                else:
                    update = self.step_controller.get_update(gradients)
                    state_x = state_v - update
                    # Project onto feasible set if configured
                    state_x = self.project_state(state_x)

            records.append(np.linalg.norm(self.NE-self.R.dot(state_x),2))
            pos_records.append(self.R.dot(state_x))

        return records, pos_records, graph_history, state_x

    def position_plot(self, pos_record, save=False, index_set = None, adversarial=None):
        figure = plt.figure()
        pos_record = np.reshape(pos_record,[-1, self.dim_action])

        indexs = range(self.N) if index_set is None else index_set
        for i in indexs:
            if i in adversarial:
                plt.plot(pos_record[:,2*i],pos_record[:,2*i+1], '--')
            else:
                plt.plot(pos_record[:,2*i],pos_record[:,2*i+1])
        
        for i in range(self.N):
            plt.plot(self.NE[2*i], self.NE[2*i+1], marker='.', markersize=3, color="red")
        
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')

        # plt.rc('axes', labelsize = 12)    # fontsize of the x and y labels
        # plt.rc('xtick', labelsize = 8)    # fontsize of the tick labels
        # plt.rc('ytick', labelsize = 8)    # fontsize of the tick labels
      
        # Draw projection constraint (if configured) on the same axes
        try:
            method = getattr(self.sim_config, 'projection', 'none')
            params = getattr(self.sim_config, 'projection_params', {}) or {}
            ax = plt.gca()

            # helper to extract a 2D (x,y) pair from possibly full-state or simple 2-vector
            def _extract_xy(param):
                if param is None:
                    return None
                arr = np.asarray(param).flatten()
                if arr.size == 0:
                    return None
                # If param matches full state length, take first agent's action components
                try:
                    if arr.size == self.dim_state:
                        return (float(arr[0]), float(arr[1]))
                except Exception:
                    pass
                # If vector has at least 2 entries, take the first two
                if arr.size >= 2:
                    return (float(arr[0]), float(arr[1]))
                # If single value, return it as x with y=0
                return (float(arr[0]), 0.0)

            if method == 'ball':
                center = params.get('center', None)
                radius = params.get('radius', None)
                if radius is not None:
                    center_xy = _extract_xy(center) or (0.0, 0.0)
                    r = float(np.asarray(radius).flatten().item()) if np.asarray(radius).size > 0 else float(radius)
                    circ = Circle(center_xy, r, fill=False, edgecolor='gray', linestyle='--', linewidth=1.5, alpha=0.8)
                    ax.add_patch(circ)

            elif method == 'box':
                low = params.get('low', None)
                high = params.get('high', None)
                if low is not None and high is not None:
                    low_xy = _extract_xy(low)
                    high_xy = _extract_xy(high)
                    if low_xy is not None and high_xy is not None:
                        lx, ly = low_xy
                        hx, hy = high_xy
                        x0 = float(min(lx, hx))
                        y0 = float(min(ly, hy))
                        width = float(abs(hx - lx))
                        height = float(abs(hy - ly))
                        rect = Rectangle((x0, y0), width, height, fill=False, edgecolor='gray', linestyle='--', linewidth=1.5, alpha=0.8)
                        ax.add_patch(rect)

            # ensure equal aspect so circle looks like a circle
            ax.set_aspect('equal', adjustable='box')
        except Exception:
            # Don't crash plotting if drawing the constraint fails
            pass

        plt.show()
        if save:
            save_plot(figure, "{0}".format(self.example))
        plt.close()

def compare_step_methods(game, init_state, methods=None, num_iter=1000):
    """
    Compare different step size methods on the same initial conditions.
    
    Parameters
    ----------
    game : Resilient
        The game instance to use
    init_state : np.ndarray
        Initial state (same for all methods)
    methods : list
        List of (method_name, base_lr) tuples to compare
        Default: [('constant', 0.025), ('adagrad', 0.5), ('rmsprop', 0.1), ('adam', 0.1)]
    num_iter : int
        Number of iterations to run
    
    Returns
    -------
    dict
        Dictionary mapping method names to their error records
    """
    if methods is None:
        methods = [
            ('constant', 0.025),
            ('adagrad', 0.5),
            ('rmsprop', 0.1),
            ('adam', 0.1)
        ]
    
    original_num_iter = game.sim_config.num_iter
    game.sim_config.num_iter = num_iter
    
    results = {}
    
    for method, base_lr in methods:
        print(f"\n{'='*50}")
        print(f"Running {method} with base_lr={base_lr}")
        print('='*50)
        
        if method == 'constant':
            game.sim_config.step_size = base_lr
            game.step_controller = None
            game.sim_config.step_method = 'constant'
        elif method == 'accelerated':
            game.set_step_method('accelerated', base_lr)
        else:
            game.set_step_method(method, base_lr)
        
        err_record, _, _, _ = game.iterate_algo(init_state.copy(), verbose=True)
        results[f"{method}_lr{base_lr}"] = err_record
        
        # Report convergence stats
        final_error = err_record[-1]
        min_error = min(err_record)
        print(f"Final error: {final_error:.6f}, Min error: {min_error:.6f}")
    
    game.sim_config.num_iter = original_num_iter
    return results


def plot_convergence_comparison(results, title="Step Method Comparison", save_path=None):
    """
    Plot convergence curves for multiple methods.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping method names to error records
    title : str
        Plot title
    save_path : str, optional
        If provided, save the plot to this path
    """
    figure = plt.figure(figsize=(10, 6))
    
    for method_name, err_record in results.items():
        plt.plot(err_record, label=method_name)
    
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(r'$||x_{k} - x^{*}||$')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    if save_path:
        save_plot(figure, save_path)
    
    plt.close()
    return figure


def animate_position_comparison(game, init_state, num_iter=500, save_path=None, frame_skip=10):
    """
    Animate agent positions over time: constant vs Nesterov, side by side.
    
    Parameters
    ----------
    game : Resilient
        The game instance
    init_state : np.ndarray
        Initial state (same for both methods)
    num_iter : int
        Number of iterations to run
    save_path : str, optional
        If provided, save animation to this path (.gif)
    frame_skip : int
        Plot every frame_skip-th iteration (reduces frame count for faster animation)
    """
    methods = [('constant', 0.025), ('accelerated', 0.05)]
    pos_records = {}
    err_records = {}
    
    original_num_iter = game.sim_config.num_iter
    game.sim_config.num_iter = num_iter
    
    for method, base_lr in methods:
        print(f"Running {method} (lr={base_lr}) for animation...")
        if method == 'constant':
            game.sim_config.step_size = base_lr
            game.step_controller = None
            game.sim_config.step_method = 'constant'
        else:
            game.set_step_method('accelerated', base_lr)
        err_rec, pos_rec, _, _ = game.iterate_algo(init_state.copy(), verbose=False)
        pos_records[method] = pos_rec
        err_records[method] = err_rec
    
    game.sim_config.num_iter = original_num_iter
    
    # Subsample frames
    indices = list(range(0, num_iter + 1, frame_skip))
    if indices[-1] != num_iter:
        indices.append(num_iter)
    
    N = game.N
    NE = game.NE
    adversarial = game.random_agents | game.constant_agents
    honest = [i for i in range(N) if i not in adversarial]
    
    # Shared axis limits from full trajectory
    all_x = []
    all_y = []
    for pos_rec in pos_records.values():
        for pos in pos_rec:
            p = np.reshape(pos, (-1,))
            for i in range(N):
                all_x.append(p[2*i])
                all_y.append(p[2*i+1])
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    margin = 0.1 * max(x_max - x_min, y_max - y_min, 1)
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin
    
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    
    # NE positions (red dots)
    ne_flat = np.ravel(NE)
    ne_x = [float(ne_flat[2*i]) for i in range(N)]
    ne_y = [float(ne_flat[2*i+1]) for i in range(N)]
    
    scat_left_honest = ax_left.scatter([], [], c='blue', s=20, alpha=0.7, label='Honest', zorder=6)
    scat_left_adv = ax_left.scatter([], [], c='red', s=30, alpha=0.8, marker='s', label='Adversarial', zorder=6)
    scat_right_honest = ax_right.scatter([], [], c='orange', s=20, alpha=0.7, label='Honest', zorder=6)
    scat_right_adv = ax_right.scatter([], [], c='red', s=30, alpha=0.8, marker='s', label='Adversarial', zorder=6)
    ax_left.scatter(ne_x, ne_y, c='darkgreen', s=30, marker='x', label='NE', zorder=2)
    ax_right.scatter(ne_x, ne_y, c='darkgreen', s=30, marker='x', label='NE', zorder=2)
    
    for ax in (ax_left, ax_right):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='upper right', fontsize=8)
    
    ax_left.set_title('Constant (α=0.025)')
    ax_right.set_title('Nesterov (α=0.05, β=0.9)')
    
    title_text = fig.suptitle('', fontsize=12)
    
    def init():
        scat_left_honest.set_offsets(np.empty((0, 2)))
        scat_left_adv.set_offsets(np.empty((0, 2)))
        scat_right_honest.set_offsets(np.empty((0, 2)))
        scat_right_adv.set_offsets(np.empty((0, 2)))
        return scat_left_honest, scat_left_adv, scat_right_honest, scat_right_adv, title_text
    
    def update(frame_idx):
        k = indices[min(frame_idx, len(indices) - 1)]
        
        pos_const = np.reshape(pos_records['constant'][k], (-1,))
        pos_accel = np.reshape(pos_records['accelerated'][k], (-1,))
        
        x_const_h = [pos_const[2*i] for i in honest]
        y_const_h = [pos_const[2*i+1] for i in honest]
        x_const_a = [pos_const[2*i] for i in adversarial]
        y_const_a = [pos_const[2*i+1] for i in adversarial]
        x_accel_h = [pos_accel[2*i] for i in honest]
        y_accel_h = [pos_accel[2*i+1] for i in honest]
        x_accel_a = [pos_accel[2*i] for i in adversarial]
        y_accel_a = [pos_accel[2*i+1] for i in adversarial]
        
        scat_left_honest.set_offsets(np.c_[x_const_h, y_const_h] if honest else np.empty((0, 2)))
        scat_left_adv.set_offsets(np.c_[x_const_a, y_const_a] if adversarial else np.empty((0, 2)))
        scat_right_honest.set_offsets(np.c_[x_accel_h, y_accel_h] if honest else np.empty((0, 2)))
        scat_right_adv.set_offsets(np.c_[x_accel_a, y_accel_a] if adversarial else np.empty((0, 2)))
        
        err_const = err_records['constant'][k]
        err_accel = err_records['accelerated'][k]
        title_text.set_text(f'Iteration {k}  |  Constant error: {err_const:.2e}  |  Nesterov error: {err_accel:.2e}')
        
        return scat_left_honest, scat_left_adv, scat_right_honest, scat_right_adv, title_text
    
    n_frames = len(indices)
    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames, interval=80,
                        blit=True)
    
    plt.tight_layout()
    
    if save_path:
        if not save_path.endswith('.gif'):
            save_path = save_path + '.gif'
        print(f"Saving animation to {save_path}...")
        writer = PillowWriter(fps=15)
        anim.save(save_path, writer=writer)
        print("Done.")
    else:
        plt.show()
    
    plt.close()
    return anim


def animate_trajectory_with_graph(game, init_state, num_iter=500, save_path=None, frame_skip=10,
                                  constant_lr=0.025, accel_lr=0.025, accel_momentum=0.9):
    """
    Animate agent positions converging to NE alongside the time-varying communication graph.

    Three-panel layout:
      Left:   communication graph with edges updating at each topology switch.
      Center: agent trajectory using constant step size.
      Right:  agent trajectory using Nesterov (accelerated) method.

    Both simulations see the same graph-switch sequence (game state + rng restored
    between runs).

    Parameters
    ----------
    game : Resilient
        The game instance
    init_state : np.ndarray
        Initial state vector (same for both methods)
    num_iter : int
        Number of algorithm iterations
    save_path : str, optional
        Save animation as .gif to this path
    frame_skip : int
        Show every N-th iteration as a frame
    constant_lr : float
        Step size for the constant method
    accel_lr : float
        Base learning rate for the accelerated (Nesterov) method
    accel_momentum : float
        Momentum parameter for the accelerated method
    """
    from matplotlib.collections import LineCollection
    import copy

    original_num_iter = game.sim_config.num_iter
    original_step_method = game.sim_config.step_method
    original_step_size = game.sim_config.step_size
    original_base_lr = game.sim_config.adaptive_base_lr
    original_momentum = game.sim_config.accelerated_momentum
    game.sim_config.num_iter = num_iter

    saved_Gc = game.Gc.copy()
    saved_Go = game.Go.copy()
    saved_adj_list = copy.deepcopy(game.adj_list_gc)
    saved_offsets = game.offsets.copy()
    saved_neighbors = game.neighbors.copy()
    saved_rng_state = copy.deepcopy(game.rng.bit_generator.state)

    # --- Run 1: constant step ---
    game.sim_config.step_method = 'constant'
    game.sim_config.step_size = constant_lr
    game.step_controller = None
    print(f"Running constant (lr={constant_lr}, {num_iter} iters) for animation...")
    err_rec_const, pos_rec_const, graph_history, _ = game.iterate_algo(init_state.copy(), verbose=True)

    # Restore graph state + rng so the accelerated run sees the same topology sequence
    game.Gc = saved_Gc.copy()
    game.Go = saved_Go.copy()
    game.adj_list_gc = copy.deepcopy(saved_adj_list)
    game.offsets = saved_offsets.copy()
    game.neighbors = saved_neighbors.copy()
    game.rng.bit_generator.state = copy.deepcopy(saved_rng_state)

    # --- Run 2: accelerated (Nesterov) ---
    game.set_step_method('accelerated', accel_lr, momentum=accel_momentum)
    print(f"Running Nesterov (lr={accel_lr}, momentum={accel_momentum}, {num_iter} iters) for animation...")
    err_rec_accel, pos_rec_accel, _, _ = game.iterate_algo(init_state.copy(), verbose=True)

    # Restore original game settings
    game.sim_config.num_iter = original_num_iter
    game.sim_config.step_method = original_step_method
    game.sim_config.step_size = original_step_size
    game.sim_config.adaptive_base_lr = original_base_lr
    game.sim_config.accelerated_momentum = original_momentum
    game._init_step_controller()

    # Subsample frame indices
    indices = list(range(0, num_iter + 1, frame_skip))
    if indices[-1] != num_iter:
        indices.append(num_iter)

    N = game.N
    NE = game.NE
    adversarial = game.random_agents | game.constant_agents
    honest = [i for i in range(N) if i not in adversarial]
    tail_length = max(20, 2 * frame_skip)

    # Fixed node positions for the graph panel (grid layout)
    graph_pos = {}
    node_idx = 0
    end = game.grid_width - 1
    for row in range(game.grid_width):
        for col in range(game.grid_width):
            is_corner = (
                (row < game.corner_size and col < game.corner_size) or
                (row < game.corner_size and col > end - game.corner_size) or
                (row > end - game.corner_size and col < game.corner_size) or
                (row > end - game.corner_size and col > end - game.corner_size)
            )
            if not is_corner:
                # Match visualize_graph: (col, row) for alignment with trajectory axes
                graph_pos[node_idx] = (col, row)
                node_idx += 1

    # Per-panel axis limits so a divergent method doesn't crush the other panel
    def _compute_limits(pos_list):
        ax_all, ay_all = [], []
        for pos in pos_list:
            p = np.ravel(pos)
            for i in range(N):
                ax_all.append(p[2 * i])
                ay_all.append(p[2 * i + 1])
        xlo, xhi = min(ax_all), max(ax_all)
        ylo, yhi = min(ay_all), max(ay_all)
        m = 0.1 * max(xhi - xlo, yhi - ylo, 1)
        return xlo - m, xhi + m, ylo - m, yhi + m

    limits_const = _compute_limits(pos_rec_const)
    limits_accel = _compute_limits(pos_rec_accel)

    ne_flat = np.ravel(NE)

    def get_graph_for_iter(k):
        active_gc = graph_history[0][1]
        for switch_iter, gc in graph_history:
            if switch_iter <= k:
                active_gc = gc
            else:
                break
        return active_gc

    # --- Figure setup: 3 panels ---
    fig, (ax_graph, ax_const, ax_nesterov) = plt.subplots(1, 3, figsize=(20, 6))
    fig.subplots_adjust(wspace=0.25)

    # --- Graph panel (left) ---
    graph_pos_arr = np.array([graph_pos[i] for i in range(N)])
    node_colors_graph = ['#ff7f0e' if i in adversarial else '#1f77b4' for i in range(N)]
    ax_graph.scatter(graph_pos_arr[:, 0], graph_pos_arr[:, 1],
                     c=node_colors_graph, s=120, zorder=5, edgecolors='white', linewidths=0.5)
    for i in range(N):
        ax_graph.annotate(str(i), graph_pos[i], fontsize=5, ha='center', va='center',
                          color='white', zorder=6)
    ax_graph.set_xlim(-0.5, end + 0.5)
    ax_graph.set_ylim(-0.5, end + 0.5)
    ax_graph.set_aspect('equal')
    ax_graph.set_title('Communication Graph')
    ax_graph.set_xlabel('Grid column')
    ax_graph.set_ylabel('Grid row')

    edge_collection = LineCollection([], colors='gray', linewidths=0.4, alpha=0.5, zorder=1)
    ax_graph.add_collection(edge_collection)
    graph_title = ax_graph.text(0.5, 1.06, '', transform=ax_graph.transAxes,
                                ha='center', fontsize=9)

    # --- Helper to set up a trajectory axis ---
    ne_x = [float(ne_flat[2 * i]) for i in range(N)]
    ne_y = [float(ne_flat[2 * i + 1]) for i in range(N)]

    def setup_trajectory_ax(ax, title, limits):
        x_lo, x_hi, y_lo, y_hi = limits
        # NE behind trails and agents so agent dots stay visible at convergence
        ax.scatter(ne_x, ne_y, c='darkgreen', s=40, marker='x', label='NE', zorder=2)
        scat_h = ax.scatter([], [], c='#1f77b4', s=25, alpha=0.8, label='Honest', zorder=6)
        scat_a = ax.scatter([], [], c='#ff7f0e', s=35, alpha=0.8, marker='s', label='Adversarial', zorder=6)
        trails_h = [ax.plot([], [], color='#1f77b4', alpha=0.25, linewidth=0.6, zorder=3)[0] for _ in honest]
        trails_a = [ax.plot([], [], color='#ff7f0e', alpha=0.25, linewidth=0.6, linestyle='--', zorder=3)[0] for _ in adversarial]
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=7)
        return scat_h, scat_a, trails_h, trails_a

    scat_const_h, scat_const_a, trails_const_h, trails_const_a = setup_trajectory_ax(
        ax_const, f'Constant ({chr(945)}={constant_lr})', limits_const)
    scat_accel_h, scat_accel_a, trails_accel_h, trails_accel_a = setup_trajectory_ax(
        ax_nesterov, f'Nesterov ({chr(945)}={accel_lr}, {chr(946)}={accel_momentum})', limits_accel)

    title_text = fig.suptitle('', fontsize=11, y=0.98)

    def _edges_from_gc(gc):
        segments = []
        for i in range(N):
            for j in range(i + 1, N):
                if gc[i, j]:
                    segments.append([graph_pos[i], graph_pos[j]])
        return segments

    def _update_trajectory_panel(k, pos_rec, scat_h, scat_a, trails_h, trails_a):
        p = np.ravel(pos_rec[k])
        hx = [p[2 * i] for i in honest]
        hy = [p[2 * i + 1] for i in honest]
        ax_ = [p[2 * i] for i in adversarial]
        ay = [p[2 * i + 1] for i in adversarial]
        scat_h.set_offsets(np.c_[hx, hy] if honest else np.empty((0, 2)))
        scat_a.set_offsets(np.c_[ax_, ay] if adversarial else np.empty((0, 2)))

        start = max(0, k - tail_length)
        for idx_h, agent in enumerate(honest):
            xs = [float(np.ravel(pos_rec[t])[2 * agent]) for t in range(start, k + 1)]
            ys = [float(np.ravel(pos_rec[t])[2 * agent + 1]) for t in range(start, k + 1)]
            trails_h[idx_h].set_data(xs, ys)
        for idx_a, agent in enumerate(adversarial):
            xs = [float(np.ravel(pos_rec[t])[2 * agent]) for t in range(start, k + 1)]
            ys = [float(np.ravel(pos_rec[t])[2 * agent + 1]) for t in range(start, k + 1)]
            trails_a[idx_a].set_data(xs, ys)

    def update(frame_idx):
        k = indices[min(frame_idx, len(indices) - 1)]

        # Graph panel
        gc = get_graph_for_iter(k)
        edge_collection.set_segments(_edges_from_gc(gc))
        num_edges = int(np.sum(gc) / 2)
        graph_title.set_text(f'Edges: {num_edges}')

        # Constant trajectory panel
        _update_trajectory_panel(k, pos_rec_const, scat_const_h, scat_const_a,
                                 trails_const_h, trails_const_a)
        # Nesterov trajectory panel
        _update_trajectory_panel(k, pos_rec_accel, scat_accel_h, scat_accel_a,
                                 trails_accel_h, trails_accel_a)

        err_c = err_rec_const[k]
        err_n = err_rec_accel[k]
        title_text.set_text(
            f'Iteration {k}/{num_iter}  |  Constant err: {err_c:.3e}  |  Nesterov err: {err_n:.3e}')

        return (edge_collection, graph_title, title_text,
                scat_const_h, scat_const_a, scat_accel_h, scat_accel_a,
                *trails_const_h, *trails_const_a, *trails_accel_h, *trails_accel_a)

    n_frames = len(indices)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True)

    if save_path:
        if not save_path.endswith('.gif'):
            save_path += '.gif'
        print(f"Saving animation to {save_path} ({n_frames} frames)...")
        writer = PillowWriter(fps=12)
        anim.save(save_path, writer=writer)
        print("Done.")
    else:
        plt.show()

    plt.close()
    return anim


def plot_save_file_data(selected, adversarial):
    pos_records = []
    with open('position_data.txt') as f:
        for line in f:
            data = [ float(num) for num in line.split(',')]
            pos_records.append(np.array([data]).T)

    game.position_plot(pos_records, save = True, index_set=selected, adversarial=adversarial)

    err_record = []
    with open('error_data.txt') as f:
        for line in f:
            err_record.append(float(line))

    with PlotErrorFigure('error_plot') as error_fig:
        plt.plot(err_record)

def main(game, sim_config):
    '''main function to run the examples'''
    init_state = -7 + 14*np.random.rand(game.dim_state,1)
    for i in range(sim_config.num_rounds):
        print(f'Executing round: {i} / {sim_config.num_rounds}')
        err_record, pos_record, _, last_iter = game.iterate_algo(init_state)

        # Writing the results to a file because the run time can be long.
        # If the program crashes or needs to stop then the simulation can
        # be continued by using these files.
        with open('position_data.txt', 'a') as f:
            for data in pos_record:
                record = data.T
                record = list(record[0])
                f.write("%s\n" % ",".join([str(num) for num in record]))

        with open('error_data.txt', 'a') as f:
            for data in err_record:
                f.write("%s\n" % data)

        with open('last_state.txt', 'a') as f:
            data = last_iter.T
            f.write("%s\n" % ",".join([str(num) for num in data[0]]))

        print(last_iter)
        init_state = last_iter

    return err_record, pos_record, last_iter

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Resilient Nash Equilibrium Seeking Simulation')
    parser.add_argument('--mode', choices=['original', 'compare', 'adaptive', 'animate', 'animate-graph'], default='original',
                        help='Run mode: original, compare, adaptive, animate (constant vs Nesterov), or animate-graph (trajectory + topology)')
    parser.add_argument('--step-method', choices=['constant', 'adagrad', 'rmsprop', 'adam', 'amsgrad', 'nadam', 'accelerated'], default='adam',
                        help='Step method to use in adaptive mode (accelerated = Nesterov GRANE)')
    parser.add_argument('--base-lr', type=float, default=0.1, help='Base learning rate / step size for adaptive and accelerated methods')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for accelerated (Nesterov) method')
    parser.add_argument('--num-iter', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--grid-width', type=int, default=10, help='Grid width for the network')
    parser.add_argument('--profile', action='store_true', help='Run with cProfile')
    parser.add_argument('--save-animation', type=str, default=None, metavar='PATH',
                        help='Save animation to PATH (e.g. animation.gif). Use MPLBACKEND=Agg for headless.')
    parser.add_argument('--frame-skip', type=int, default=10, help='Animation: plot every Nth iteration (default 10)')
    parser.add_argument('--seed', type=int, default=420, help='Random seed for reproducible experiments')
    parser.add_argument('--projection', choices=['none','box','ball','custom'], default='none',
                        help='Projection to apply after each gradient update')
    parser.add_argument('--projection-params', type=str, default='{}',
                        help='JSON string with projection params (e.g. "{\"low\":-5,\"high\":5}")')
    args = parser.parse_args()
    
    continue_run = False
    
    if not continue_run:
        if os.path.exists("error_data.txt"):
            os.remove("error_data.txt")
        if os.path.exists("position_data.txt"):
            os.remove("position_data.txt")
        if os.path.exists("last_state.txt"):
            os.remove("last_state.txt")


    print('Warning: failed to parse --projection-params JSON, using {}')
    projection_params = {}

    sim_config = simulation_config(num_iter=args.num_iter, seed=args.seed,
                                   projection=args.projection,
                                   projection_params=projection_params)
    
    selected = [0,1,2,3,4,5,6,7,8,9,10,11]
    random_agents = []
    constant_agents = []
    adversarial = random_agents + constant_agents

    # Create game instance
    if args.grid_width == 15:
        game = Resilient(sim_config, grid_width=15, 
                        random_agents=set[int]([5, 71, 8, 74, 10, 78, 17, 87, 28, 95, 46, 61]), 
                        constant_agents=None, l_inf_ball=2, D=3, corner_size=1)
    elif args.grid_width == 8:
        game = Resilient(sim_config, grid_width=8,
                        random_agents=set([4, 6, 11, 19, 26, 32, 38, 41]),
                        constant_agents=None, l_inf_ball=1, D=1, corner_size=1)
    elif args.grid_width == 7:
        game = Resilient(sim_config, grid_width=7,
                        random_agents=set([4, 6, 11, 19, 26, 31]),
                        constant_agents=None, l_inf_ball=1, D=1, corner_size=1)
    else:
        game = Resilient(sim_config, grid_width=args.grid_width, 
                        random_agents=None, constant_agents=None, 
                        l_inf_ball=1, D=1, corner_size=1)

    # If projection requested but no params provided, create reasonable defaults
    try:
        proj = sim_config.projection
        params = sim_config.projection_params or {}
        if proj == 'box':
            if 'low' not in params or 'high' not in params:
                # derive box from NE agent positions (x,y) with margin
                ne_flat = np.ravel(game.NE)
                xs = [float(ne_flat[2*i]) for i in range(game.N)]
                ys = [float(ne_flat[2*i+1]) for i in range(game.N)]
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                margin = 0.1 * max(maxx - minx, maxy - miny, 1.0)
                low = [minx - margin, miny - margin]
                high = [maxx + margin, maxy + margin]
                # expand to full state dimension (repeat per agent)
                low_full = np.array(low * game.N).reshape(-1, 1)
                high_full = np.array(high * game.N).reshape(-1, 1)
                sim_config.projection_params.update({'low': low_full, 'high': high_full})
        elif proj == 'ball':
            if 'radius' not in params:
                ne_flat = np.ravel(game.NE)
                xs = np.array([float(ne_flat[2*i]) for i in range(game.N)])
                ys = np.array([float(ne_flat[2*i+1]) for i in range(game.N)])
                cx = float(xs.mean())
                cy = float(ys.mean())
                dists = np.sqrt((xs - cx)**2 + (ys - cy)**2)
                radius = float(dists.max() + 0.1 * max(xs.max()-xs.min(), ys.max()-ys.min(), 1.0))
                # center expanded to full state dim
                center_full = np.array([cx, cy] * game.N).reshape(-1, 1)
                sim_config.projection_params.update({'center': center_full, 'radius': radius})
    except Exception:
        # don't crash if something goes wrong building defaults
        pass

    if args.mode == 'compare':
        # Run comparison of all methods
        print("\n" + "="*60)
        print("COMPARING STEP SIZE METHODS")
        print("="*60)
        
        init_state = -7 + 14*np.random.rand(game.dim_state, 1)
        
        # Compare different methods (accelerated = Nesterov-style GRANE)
        methods = [
            ('constant', 0.025),   # Original fixed step size
            ('accelerated', 0.025), # Accelerated GRANE (Nesterov on filtered state)
            # ('adam', 0.1),
            # ('amsgrad', 0.1),
            # ('nadam', 0.1)
        ]
        
        results = compare_step_methods(game, init_state, methods=methods, num_iter=args.num_iter)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for method, errors in results.items():
            print(f"{method:20s}: Final={errors[-1]:.6f}, Min={min(errors):.6f}")
        
        # Plot comparison
        plot_convergence_comparison(results, 
                                   title=f"Step Method Comparison (grid={args.grid_width}, iter={args.num_iter})",
                                   save_path="convergence_comparison")
        
    elif args.mode == 'animate':
        # Animate constant vs Nesterov position comparison
        print("\n" + "="*60)
        print("ANIMATING: Constant vs Nesterov")
        print("="*60)
        init_state = -7 + 14*np.random.rand(game.dim_state, 1)
        animate_position_comparison(
            game, init_state,
            num_iter=args.num_iter,
            save_path=args.save_animation,
            frame_skip=args.frame_skip
        )
        
    elif args.mode == 'animate-graph':
        print("\n" + "="*60)
        print("ANIMATING: Constant vs Nesterov Trajectory + Time-Varying Graph")
        print("="*60)
        init_state = -7 + 14*np.random.rand(game.dim_state, 1)

        animate_trajectory_with_graph(
            game, init_state,
            num_iter=args.num_iter,
            save_path=args.save_animation,
            frame_skip=args.frame_skip
        )

    elif args.mode == 'adaptive':
        # Run with specified adaptive or accelerated method
        print(f"\nRunning with {args.step_method} (base_lr={args.base_lr})")
        if args.step_method == 'accelerated':
            game.set_step_method(args.step_method, args.base_lr, momentum=args.momentum)
        else:
            game.set_step_method(args.step_method, args.base_lr)
        
        if args.profile:
            cProfile.run('main(game, sim_config)', sort='cumulative')
        else:
            main(game, sim_config)
        
        plot_save_file_data(selected, adversarial)
        
    else:
        # Original behavior with fixed step size
        print("\nRunning with original fixed step size")
        
        if args.profile:
            cProfile.run('main(game, sim_config)', sort='cumulative')
        else:
            main(game, sim_config)
        
        from pypdf import PdfWriter
        import glob

        def consolidate_graphs_to_pdf(output_filename="simulation_report.pdf"):
            """Merges all iteration graph PDFs into a single multi-page document."""
            writer = PdfWriter()
            
            # Grab all graph files and sort them alphabetically
            files = sorted(glob.glob("graph_plots/graph_iter_*.pdf"))
            
            if not files:
                print("No graph PDFs found to consolidate.")
                return

            print(f"Consolidating {len(files)} graphs into {output_filename}...")
            
            for file in files:
                writer.append(file)
            
            with open(output_filename, "wb") as f:
                writer.write(f)
                
            print("Consolidation complete.")

            # Optional: Clean up the individual files to keep your workspace tidy
            # for file in files:
            #     os.remove(file)

        consolidate_graphs_to_pdf("topology_evolution.pdf")
        plot_save_file_data(selected, adversarial)
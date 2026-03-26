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
from scipy.linalg import block_diag
import cProfile

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
        
        # Initialize adaptive step controller
        self.step_controller = None
        self._init_step_controller()

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
            (error_records, position_records, final_state)
        """
        state_x = init_state
        if verbose:
            print('Starting iteration with method:', self.sim_config.step_method)
            print('Initial state norm:', np.linalg.norm(state_x))
        
        records = [np.linalg.norm(self.NE-self.R.dot(state_x),2)]
        pos_records = [self.R.dot(state_x)]
        
        # Reset adaptive controller if using one
        if self.step_controller is not None:
            self.step_controller.reset()
        
        # For accelerated GRANE: Nesterov extrapolation on filtered state
        state_v_prev = None
        alpha = self.sim_config.adaptive_base_lr if self.sim_config.step_method == 'accelerated' else self.sim_config.step_size
        beta = self.sim_config.accelerated_momentum
        
        for i in range(self.sim_config.num_iter):
            if verbose and (i%100) == 0:
                print(f"Iteration {i} of {self.sim_config.num_iter}, error: {records[-1]:.6f}")
            
            state_y = self.adversarial_communication(state_x)
            # state_v = self.filter_communicated_message(state_y)
            state_v = remove_d.filter_communicated_message(state_y, self.Go, self.offsets, self.neighbors, self.D)

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
                state_v_prev = state_v
            else:
                # Compute gradient at current filtered state
                temp1 = self.F.dot(state_v)
                temp2 = self.R.transpose().dot(temp1)
                gradients = temp2 + self.RB
                # Apply step size (constant or adaptive)
                if self.step_controller is None:
                    state_x = state_v - self.sim_config.step_size * gradients
                else:
                    update = self.step_controller.get_update(gradients)
                    state_x = state_v - update

            records.append(np.linalg.norm(self.NE-self.R.dot(state_x),2))
            pos_records.append(self.R.dot(state_x))

        return records, pos_records, state_x

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
        
        err_record, _, _ = game.iterate_algo(init_state.copy(), verbose=True)
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
        err_rec, pos_rec, _ = game.iterate_algo(init_state.copy(), verbose=False)
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
    
    scat_left_honest = ax_left.scatter([], [], c='blue', s=20, alpha=0.7, label='Honest')
    scat_left_adv = ax_left.scatter([], [], c='red', s=30, alpha=0.8, marker='s', label='Adversarial')
    scat_right_honest = ax_right.scatter([], [], c='orange', s=20, alpha=0.7, label='Honest')
    scat_right_adv = ax_right.scatter([], [], c='red', s=30, alpha=0.8, marker='s', label='Adversarial')
    ax_left.scatter(ne_x, ne_y, c='darkgreen', s=30, marker='x', label='NE')
    ax_right.scatter(ne_x, ne_y, c='darkgreen', s=30, marker='x', label='NE')
    
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


def _sample_random_adversaries(n_agents: int, fraction: float, rng: "np.random.Generator") -> set:
    """Pick roughly ``fraction * n_agents`` distinct agent indices (at least one if fraction > 0)."""
    if fraction <= 0 or n_agents <= 0:
        return set()
    n_adv = min(n_agents, max(1, round(fraction * n_agents)))
    chosen = rng.choice(n_agents, size=n_adv, replace=False)
    return set(int(x) for x in chosen.tolist())


def _axis_limits_from_positions(pos_records, n_agents: int):
    all_x, all_y = [], []
    for pos in pos_records:
        p = np.reshape(pos, (-1,))
        for i in range(n_agents):
            all_x.append(p[2 * i])
            all_y.append(p[2 * i + 1])
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    margin = 0.1 * max(x_max - x_min, y_max - y_min, 1)
    return x_min - margin, x_max + margin, y_min - margin, y_max + margin


def animate_dual_grid_trajectories(
    num_iter=500,
    save_path=None,
    frame_skip=10,
    seed=None,
    adversarial_fraction=0.3,
    grid_original=13,
    grid_improved=32,
    improved_base_lr=0.05,
    improved_momentum=0.9,
):
    """
    Side-by-side GIF: fixed-step on a small grid vs accelerated GRANE on a larger grid,
    each with ~``adversarial_fraction`` random (noisy) adversarial agents.
    """
    if seed is not None:
        rng_adv = np.random.default_rng(int(seed))
    else:
        rng_adv = np.random.default_rng()

    n_left = grid_original**2 - 4
    n_right = grid_improved**2 - 4
    adv_left = _sample_random_adversaries(n_left, adversarial_fraction, rng_adv)
    adv_right = _sample_random_adversaries(n_right, adversarial_fraction, rng_adv)

    sim_left = simulation_config(num_iter=num_iter)
    sim_right = simulation_config(
        num_iter=num_iter,
        step_method='accelerated',
        adaptive_base_lr=improved_base_lr,
        accelerated_momentum=improved_momentum,
    )

    game_left = Resilient(
        sim_left,
        grid_width=grid_original,
        random_agents=adv_left,
        constant_agents=None,
        l_inf_ball=1,
        D=1,
        corner_size=1,
    )
    game_right = Resilient(
        sim_right,
        grid_width=grid_improved,
        random_agents=adv_right,
        constant_agents=None,
        l_inf_ball=1,
        D=1,
        corner_size=1,
    )

    if seed is not None:
        np.random.seed(int(seed))
    init_left = -7 + 14 * np.random.rand(game_left.dim_state, 1)
    if seed is not None:
        np.random.seed(int(seed) + 1_000_003)
    init_right = -7 + 14 * np.random.rand(game_right.dim_state, 1)

    print(f"Running original (grid={grid_original}, constant step={sim_left.step_size}, |adv|={len(adv_left)})...")
    if seed is not None:
        np.random.seed(int(seed) + 2_000_003)
    err_left, pos_left, _ = game_left.iterate_algo(init_left.copy(), verbose=False)

    print(f"Running improved (grid={grid_improved}, accelerated α={improved_base_lr}, |adv|={len(adv_right)})...")
    if seed is not None:
        np.random.seed(int(seed) + 3_000_003)
    err_right, pos_right, _ = game_right.iterate_algo(init_right.copy(), verbose=False)

    indices = list(range(0, num_iter + 1, frame_skip))
    if indices[-1] != num_iter:
        indices.append(num_iter)

    adv_l = game_left.random_agents | game_left.constant_agents
    adv_r = game_right.random_agents | game_right.constant_agents
    honest_l = [i for i in range(game_left.N) if i not in adv_l]
    honest_r = [i for i in range(game_right.N) if i not in adv_r]

    xl0, xl1, yl0, yl1 = _axis_limits_from_positions(pos_left, game_left.N)
    xr0, xr1, yr0, yr1 = _axis_limits_from_positions(pos_right, game_right.N)

    ne_l = np.ravel(game_left.NE)
    ne_r = np.ravel(game_right.NE)
    ne_x_l = [float(ne_l[2 * i]) for i in range(game_left.N)]
    ne_y_l = [float(ne_l[2 * i + 1]) for i in range(game_left.N)]
    ne_x_r = [float(ne_r[2 * i]) for i in range(game_right.N)]
    ne_y_r = [float(ne_r[2 * i + 1]) for i in range(game_right.N)]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))

    scat_ll_h = ax_left.scatter([], [], c='blue', s=20, alpha=0.7, label='Honest')
    scat_ll_a = ax_left.scatter([], [], c='red', s=30, alpha=0.8, marker='s', label='Adversarial')
    scat_lr_h = ax_right.scatter([], [], c='orange', s=8, alpha=0.55, label='Honest')
    scat_lr_a = ax_right.scatter([], [], c='red', s=12, alpha=0.75, marker='s', label='Adversarial')
    ax_left.scatter(ne_x_l, ne_y_l, c='darkgreen', s=30, marker='x', label='NE')
    ax_right.scatter(ne_x_r, ne_y_r, c='darkgreen', s=18, marker='x', label='NE')

    ax_left.set_xlim(xl0, xl1)
    ax_left.set_ylim(yl0, yl1)
    ax_right.set_xlim(xr0, xr1)
    ax_right.set_ylim(yr0, yr1)

    for ax in (ax_left, ax_right):
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='upper right', fontsize=8)

    ax_left.set_title('Original algorithm')
    ax_right.set_title('Our Improved Algorithm')

    title_text = fig.suptitle('', fontsize=12)

    def init():
        scat_ll_h.set_offsets(np.empty((0, 2)))
        scat_ll_a.set_offsets(np.empty((0, 2)))
        scat_lr_h.set_offsets(np.empty((0, 2)))
        scat_lr_a.set_offsets(np.empty((0, 2)))
        return scat_ll_h, scat_ll_a, scat_lr_h, scat_lr_a, title_text

    def update(frame_idx):
        k = indices[min(frame_idx, len(indices) - 1)]
        pl = np.reshape(pos_left[k], (-1,))
        pr = np.reshape(pos_right[k], (-1,))

        xh_l = [pl[2 * i] for i in honest_l]
        yh_l = [pl[2 * i + 1] for i in honest_l]
        xa_l = [pl[2 * i] for i in sorted(adv_l)]
        ya_l = [pl[2 * i + 1] for i in sorted(adv_l)]
        xh_r = [pr[2 * i] for i in honest_r]
        yh_r = [pr[2 * i + 1] for i in honest_r]
        xa_r = [pr[2 * i] for i in sorted(adv_r)]
        ya_r = [pr[2 * i + 1] for i in sorted(adv_r)]

        scat_ll_h.set_offsets(np.c_[xh_l, yh_l] if honest_l else np.empty((0, 2)))
        scat_ll_a.set_offsets(np.c_[xa_l, ya_l] if adv_l else np.empty((0, 2)))
        scat_lr_h.set_offsets(np.c_[xh_r, yh_r] if honest_r else np.empty((0, 2)))
        scat_lr_a.set_offsets(np.c_[xa_r, ya_r] if adv_r else np.empty((0, 2)))

        title_text.set_text(
            f'Iteration {k}  |  Original error: {err_left[k]:.2e}  |  Improved error: {err_right[k]:.2e}'
        )
        return scat_ll_h, scat_ll_a, scat_lr_h, scat_lr_a, title_text

    n_frames = len(indices)
    anim = FuncAnimation(
        fig, update, init_func=init, frames=n_frames, interval=80, blit=True
    )

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
        err_record, pos_record, last_iter = game.iterate_algo(init_state)

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
    parser.add_argument('--mode', choices=['original', 'compare', 'adaptive', 'animate', 'animate-dual-grid'], default='original',
                        help='Run mode: original, compare, adaptive, animate, or animate-dual-grid (13 vs 32 grid GIF)')
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
    parser.add_argument('--seed', type=int, default=None,
                        help='NumPy RNG seed for reproducible runs (original, adaptive, compare, animate, animate-dual-grid)')
    parser.add_argument('--adversarial-fraction', type=float, default=0.3,
                        help='animate-dual-grid: fraction of random adversarial agents on each grid (default 0.3)')
    parser.add_argument('--dual-improved-lr', type=float, default=0.05,
                        help='animate-dual-grid: accelerated step size on the large grid (default 0.05)')
    parser.add_argument('--dual-improved-momentum', type=float, default=0.9,
                        help='animate-dual-grid: Nesterov momentum on the large grid (default 0.9)')
    args = parser.parse_args()
    
    continue_run = False
    
    if not continue_run:
        if os.path.exists("error_data.txt"):
            os.remove("error_data.txt")
        if os.path.exists("position_data.txt"):
            os.remove("position_data.txt")
        if os.path.exists("last_state.txt"):
            os.remove("last_state.txt")

    sim_config = simulation_config(num_iter=args.num_iter)
    
    selected = [0,1,2,3,4,5,6,7,8,9,10,11]
    random_agents = []
    constant_agents = []
    adversarial = random_agents + constant_agents

    if args.mode == 'animate-dual-grid':
        if not args.save_animation:
            parser.error('--save-animation is required for animate-dual-grid mode')
        print("\n" + "="*60)
        print("ANIMATING: Dual-grid (original vs improved)")
        print("="*60)
        animate_dual_grid_trajectories(
            num_iter=args.num_iter,
            save_path=args.save_animation,
            frame_skip=args.frame_skip,
            seed=args.seed,
            adversarial_fraction=args.adversarial_fraction,
            improved_base_lr=args.dual_improved_lr,
            improved_momentum=args.dual_improved_momentum,
        )
    else:
        if args.seed is not None:
            np.random.seed(int(args.seed))
        # Create game instance
        if args.grid_width == 15:
            game = Resilient(sim_config, grid_width=15, 
                            random_agents=set([5, 71, 8, 74, 10, 78, 17, 87, 28, 95, 46, 61]), 
                            constant_agents=None, l_inf_ball=2, D=3, corner_size=1)
        elif args.grid_width == 10:
            game = Resilient(sim_config, grid_width=10,
                            random_agents=set([4, 6, 11, 19, 26, 32, 38, 41]),
                            constant_agents=None, l_inf_ball=2, D=2, corner_size=1)
        elif args.grid_width == 6:
            game = Resilient(sim_config, grid_width=6,
                            random_agents=set([4, 6, 11, 16, 24, 25, 28, 30]),
                            constant_agents=None, l_inf_ball=2, D=2, corner_size=1)
        else:
            game = Resilient(sim_config, grid_width=args.grid_width, 
                            random_agents=None, constant_agents=None, 
                            l_inf_ball=1, D=1, corner_size=1)

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
            
            plot_save_file_data(selected, adversarial)
#!/usr/bin/env python
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
    step_method: Literal['constant', 'adagrad', 'rmsprop', 'adam'] = 'constant'
    adaptive_base_lr: float = 0.1

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
        if self.sim_config.step_method == 'constant':
            self.step_controller = None
        else:
            self.step_controller = create_controller(
                dim_state=self.dim_state,
                method=self.sim_config.step_method,
                base_lr=self.sim_config.adaptive_base_lr
            )
    
    def set_step_method(self, method: str, base_lr: float = 0.1):
        """
        Change the step size method dynamically.
        
        Parameters
        ----------
        method : str
            One of 'constant', 'adagrad', 'rmsprop', 'adam'
        base_lr : float
            Base learning rate for adaptive methods
        """
        self.sim_config.step_method = method
        self.sim_config.adaptive_base_lr = base_lr
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
        
        for i in range(self.sim_config.num_iter):
            if verbose and (i%100) == 0:
                print(f"Iteration {i} of {self.sim_config.num_iter}, error: {records[-1]:.6f}")
            
            state_y = self.adversarial_communication(state_x)
            state_v = remove_d.filter_communicated_message(state_y, self.Go, self.offsets, self.neighbors, self.D)

            # Compute gradient
            temp1 = self.F.dot(state_v)
            temp2 = self.R.transpose().dot(temp1)
            gradients = temp2 + self.RB
            
            # Apply step size (constant or adaptive)
            if self.step_controller is None:
                # Use constant step size (original behavior)
                state_x = state_v - self.sim_config.step_size * gradients
            elif self.sim_config.step_method == 'adam':
                # Use Adam's full update (includes momentum)
                update = self.step_controller.get_adam_update(gradients)
                state_x = state_v - update
            else:
                # Use adaptive step sizes (AdaGrad, RMSProp)
                step_sizes = self.step_controller.get_step_sizes(gradients)
                state_x = state_v - step_sizes * gradients

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
    parser.add_argument('--mode', choices=['original', 'compare', 'adaptive'], default='original',
                        help='Run mode: original (fixed step), compare (benchmark methods), adaptive (use adaptive)')
    parser.add_argument('--step-method', choices=['constant', 'adagrad', 'rmsprop', 'adam'], default='adam',
                        help='Step method to use in adaptive mode')
    parser.add_argument('--base-lr', type=float, default=0.1, help='Base learning rate for adaptive methods')
    parser.add_argument('--num-iter', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--grid-width', type=int, default=15, help='Grid width for the network')
    parser.add_argument('--profile', action='store_true', help='Run with cProfile')
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

    # Create game instance
    if args.grid_width == 15:
        game = Resilient(sim_config, grid_width=15, 
                        random_agents=set([5, 71, 8, 74, 10, 78, 17, 87, 28, 95, 46, 61]), 
                        constant_agents=None, l_inf_ball=2, D=3, corner_size=1)
    elif args.grid_width == 10:
        game = Resilient(sim_config, grid_width=10,
                        random_agents=set([4, 6, 11, 19, 26, 32, 38, 41]),
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
        
        # Compare different methods
        methods = [
            ('constant', 0.025),   # Original fixed step size
            ('adagrad', 0.3),
            ('rmsprop', 0.05),
            ('adam', 0.05)
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
        
    elif args.mode == 'adaptive':
        # Run with specified adaptive method
        print(f"\nRunning with {args.step_method} (base_lr={args.base_lr})")
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
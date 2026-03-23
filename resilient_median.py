#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Running the more complex resilient simulation'''
import os

import random
from typing import List
from dataclasses import dataclass

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import json
import time
from datetime import datetime
from pathlib import Path
import csv

import info_robust_graph as irg

@dataclass
class simulation_config:
    '''A container just to hold simulation parameters'''
    step_size: float = 1/100.0
    num_iter: int = 10000
    num_rounds: int = 1
    init_state: np.ndarray = None
    step_schedule: str = "constant"

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

def save_plot(figure, name, out_dir=None):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    if out_dir is None:
        figure.savefig(f"{name}.pdf", format='pdf', bbox_inches='tight', pad_inches=0.01)
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(out_dir / f"{name}.pdf", format='pdf', bbox_inches='tight', pad_inches=0.01)

class PlotErrorFigure(object):
    def __init__(self, name, out_dir=None):
        self.name = name
        self.out_dir = out_dir
        self.figure = None

    def __enter__(self):
        self.figure = plt.figure()
        return self.figure

    def __exit__(self, exc_type, exc_value, exc_traceback):
        plt.yscale("log")
        plt.xlabel('Iteration')
        plt.ylabel(r'$||x_{k} - x^{*}||$')
        # plt.show()
        save_plot(self.figure, self.name, out_dir=self.out_dir)
        plt.close()

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def make_run_dir(base_dir="runs", game=None, sim_config=None):
    """
    Create a unique directory for this run so nothing is overwritten.
    Name includes key params for quick browsing.
    """
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Try to embed some key params into folder name (short but informative)
    if game is not None and sim_config is not None:
        agg = getattr(game, "aggregation_method", "unknown")
        ra = len(getattr(game, "random_agents", []))
        ca = len(getattr(game, "constant_agents", []))
        step_sched = getattr(sim_config, "step_schedule", "constant")
        step_size = getattr(sim_config, "step_size", None)
        D = getattr(game, "D", None)
        linf = getattr(game, "grid_width", None)  # (grid width we will add too)
        l_inf_ball = getattr(game, "l_inf_ball", None) if hasattr(game, "l_inf_ball") else None

        # Make it filesystem-friendly
        step_str = f"{step_size:.2e}" if isinstance(step_size, (int, float)) else "na"
        parts = [
            ts,
            f"grid{game.grid_width}",
            f"{agg}",
            f"ra{ra}",
            f"ca{ca}",
            f"step{step_sched}",
            step_str,
            f"D{D}",
            f"linf{getattr(game, 'l_inf_ball', 'na')}",
        ]
        run_name = "_".join(str(p) for p in parts)
    else:
        run_name = ts

    run_dir = Path(base_dir) / run_name

    # If collision (rare), append a counter
    if run_dir.exists():
        k = 1
        while (Path(base_dir) / f"{run_name}_{k}").exists():
            k += 1
        run_dir = Path(base_dir) / f"{run_name}_{k}"

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def append_csv_row(path: Path, fieldnames, row_dict):
    """
    Append a row to a CSV; if file doesn't exist, write header.
    """
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def compute_run_stats(err_record, thresholds=(1e-1, 1e-2, 1e-3)):
    """
    Compute final/best error and first-iteration hitting certain thresholds.
    err_record is a list of floats of length (num_iter + 1).
    """
    errs = np.array(err_record, dtype=float)
    final_error = float(errs[-1])
    best_error = float(errs.min())
    iter_best = int(errs.argmin())

    first_below = {}
    for thr in thresholds:
        idx = np.where(errs <= thr)[0]
        first_below[f"first_below_{thr:.0e}"] = int(idx[0]) if idx.size > 0 else None

    # Optional: estimate slope of log10(error) over last 20% of iterations (rough rate)
    # Useful to compare "slowly converging" vs "stalled".
    slope = None
    try:
        start = int(0.8 * len(errs))
        y = np.log10(np.maximum(errs[start:], 1e-300))
        x = np.arange(len(y), dtype=float)
        # simple least squares slope
        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom > 0:
            slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    except Exception:
        slope = None

    stats = {
        "final_error": final_error,
        "best_error": best_error,
        "iter_best_error": iter_best,
        "log10_slope_last20pct": slope,
    }
    stats.update(first_below)
    return stats

def gather_config_dict(game, sim_config, extra=None):
    """
    Record everything needed to interpret results & reproduce.
    """
    cfg = {
        "timestamp_local": datetime.now().isoformat(),
        "sim_config": {
            "step_size": _safe_float(sim_config.step_size),
            "num_iter": int(sim_config.num_iter),
            "num_rounds": int(sim_config.num_rounds),
            "step_schedule": str(sim_config.step_schedule),
        },
        "game": {
            "grid_width": int(game.grid_width),
            "corner_size": int(game.corner_size),
            "aggregation_method": str(game.aggregation_method),
            "D": int(game.D),
            "l_inf_ball": int(game.l_inf_ball) if hasattr(game, "l_inf_ball") else None,
            "num_agents_N": int(game.N),
            "random_agents": sorted(list(game.random_agents)),
            "constant_agents": sorted(list(game.constant_agents)),
        },
    }
    if extra:
        cfg.update(extra)
    return cfg

def geometric_median(points: np.ndarray, max_iter: int = 50, tol: float = 1e-6, eps: float = 1e-12):
    """
    Weiszfeld algorithm for geometric median.
    points: (m, d) array
    Returns: (d,) array
    """
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("points must be shape (m, d) with m > 0")

    y = points.mean(axis=0)  # initialize at mean
    for _ in range(max_iter):
        diffs = points - y
        dists = np.linalg.norm(diffs, axis=1)
        # If y coincides with a point, that's a median (avoid divide by 0)
        if np.any(dists < eps):
            return points[np.argmin(dists)].copy()

        w = 1.0 / np.maximum(dists, eps)
        y_next = (points * w[:, None]).sum(axis=0) / w.sum()

        if np.linalg.norm(y_next - y) <= tol:
            return y_next
        y = y_next

    return y

class Resilient:
    """Simulation for the resilient algorithm"""
    def __init__(self, sim_config, grid_width, random_agents=None, constant_agents=None, 
                 l_inf_ball = 1, D = 1, corner_size = 1, aggregation_method = "trim",
                 median_window = None, use_geometric_median = False):
        self.sim_config = sim_config        
        self.grid_width = grid_width
        self.corner_size = corner_size
        self.l_inf_ball = l_inf_ball
        self.aggregation_method = aggregation_method
        if aggregation_method not in ("trim", "median", "median_window"):
            raise ValueError(f"Unknown aggregation_method '{aggregation_method}'. "
                             f"Use 'trim', 'median', or 'median_window'.")
        self.aggregation_method = aggregation_method
        self.median_window = median_window
        self.use_geometric_median = use_geometric_median

        self.dim_action_i = 2
        self.N = grid_width**2 - 4*corner_size**2
        self.dim_action_list = self.dim_action_i * np.ones(self.N, dtype=int)
        self.dim_action = self.dim_action_i * self.N
        self.dim_state = self.N * self.dim_action
        self.random_agents = set(random_agents) if random_agents else set()
        self.constant_agents = set(constant_agents) if constant_agents else set()

        self.D = D

        self.Gc = irg.grid_l_inf_to_adj_matrix(grid_width, l_inf_ball)
        self.corners = irg.get_corners(self.Gc, corner_size)
        self.Gc = irg.remove_nodes_from_adj_matrix(self.Gc, self.corners)
        self.Go = self.Gc + np.eye(self.N, dtype=int)
        self.adj_list_gc = irg.adj_matrix_to_adj_in_set(self.Gc, self_loop=False)

        self.R = action_select_matrix(self.dim_action_list)
        self.A, self.b = self.get_gradient()

        self.F = block_diag(*[self.A[2*i:2*(i+1),:] for i in range(self.N)])
        # self.RF = self.R.transpose().dot(self.F)
        self.RB = self.R.transpose().dot(self.b)
        self.NE = -np.linalg.inv(self.A).dot(self.b)

        self.example = 'position_plot'

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
    
    def median_aggregate(self, received_values, agent_value):
        '''
        Filter than uses the median across messages received
        '''
        all_values = list(received_values)
        all_values.append(float(agent_value))

        return float(np.median(all_values))
    
    def median_window_aggregate(self, received_values, agent_value):
        """
        Windowed median ("median-trimmed mean") for a single scalar coordinate.

        1. Compute the median m of neighbors + own value.
        2. Keep only those values with |val - m| <= median_window.
        3. Return their average.

        If median_window is None or if no values fall in the window,
        falls back to pure median.
        """

        vals = np.array(received_values + [float(agent_value)], dtype = float)
        m = np.median(vals)

        # If no window specified, just use the median
        if self.median_window is None:
            return float(m)

        # Select values within the window
        if self.median_window == "mad":
            mad = np.median(np.abs(vals - m))
            w = 1.4826 * mad + 1e-6
        else:
            w = float(self.median_window)
        mask = np.abs(vals - m) <= w
        selected = vals[mask]

        if selected.size == 0:
            # Degenerate case: nothing within window, use median
            return float(m)

        # Average of in-window values
        return float(selected.mean())
    
    def robust_geom_median_2d(self, points_2d: np.ndarray, D: int):
        """
        2D robust aggregation using geometric median as the center.
        Optional improvements:
        - adaptive MAD-like radius in 2D (median of distances to center)
        - k-closest guarantee (based on distance to center)
        Returns (agg_point(2,), frac_kept, window_used, k_used).
        """
        pts = points_2d.astype(float)
        total = pts.shape[0]

        center = geometric_median(pts)
        dists = np.linalg.norm(pts - center[None, :], axis=1)

        # ---- window selection in 2D ----
        window_used = None
        if self.median_window == "mad":
            mad2 = float(np.median(np.abs(dists - np.median(dists))))
            # Use the same multiplier but applied to a robust distance scale
            w = 1.4826 * mad2 + 1e-12
            window_used = w
            mask = dists <= (np.median(dists) + w)
            kept = pts[mask]
            kept_dists = dists[mask]
        else:
            w = float(self.median_window)
            window_used = w
            mask = dists <= w
            kept = pts[mask]
            kept_dists = dists[mask]

        if kept.shape[0] == 0:
            # fall back to geometric median of all points
            return center #, 0.0, window_used, 0

        # ---- K-closest guarantee ----
        # if self.use_kclosest:
        #     if self.k_rule == "total_minus_2D":
        #         K = max(1, total - 2 * D)
        #     elif self.k_rule == "fixed":
        #         K = max(1, min(self.k_fixed, total))
        #     else:
        #         raise ValueError(f"Unknown k_rule: {self.k_rule}")

        #     order = np.argsort(kept_dists)
        #     K_eff = min(K, kept.shape[0])
        #     kept = kept[order[:K_eff]]
        #     k_used = int(K_eff)
        # else:
        #     k_used = int(kept.shape[0])

        # frac = float(kept.shape[0]) / float(total)

        # Return mean of kept points (stable) OR geometric median of kept points (more robust but slower).
        # Here: mean is fine because outliers were filtered by geom-median centering.
        return kept.mean(axis=0)# , frac, window_used, k_used

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
                if self.aggregation_method == "median" and self.use_geometric_median:
                    # indices for x,y components
                    offset_x = self.dim_action_i * state_j + 0
                    offset_y = self.dim_action_i * state_j + 1
                    idx_i_x = self.dim_action * agent_i + offset_x
                    idx_i_y = self.dim_action * agent_i + offset_y

                    if self.Go[agent_i, state_j] == 1:
                        idx_j_x = self.dim_action * state_j + offset_x
                        idx_j_y = self.dim_action * state_j + offset_y
                        state_v[idx_i_x] = state_y[idx_j_x, state_j]
                        state_v[idx_i_y] = state_y[idx_j_y, state_j]
                    else:
                        # gather neighbor 2D points (neighbors' claims about agent_i)
                        pts = []
                        for X in self.adj_list_gc[agent_i]:
                            px = float(state_y[self.dim_action * X + offset_x, agent_i])
                            py = float(state_y[self.dim_action * X + offset_y, agent_i])
                            pts.append([px, py])

                        # include own value
                        ownx = float(state_y[idx_i_x, agent_i])
                        owny = float(state_y[idx_i_y, agent_i])
                        pts.append([ownx, owny])

                        pts = np.array(pts, dtype=float)
                        agg2 = self.robust_geom_median_2d(pts, D=self.D)
                        state_v[idx_i_x] = float(agg2[0])
                        state_v[idx_i_y] = float(agg2[1])

                    continue  # skip scalar loop since we handled both components

                for component_k in range(self.dim_action_i):
                    offset = self.dim_action_i*state_j+component_k
                    state_index_i = self.dim_action*agent_i + offset
                    if self.Go[agent_i, state_j] == 1:
                        state_index_j = self.dim_action*state_j + offset
                        state_v[state_index_i] = state_y[state_index_j,state_j]
                    else:
                        agent_i_in_messages = [
                            state_y[self.dim_action*X + offset, agent_i]
                            for X in self.adj_list_gc[agent_i]
                        ]
                        own_value = state_y[state_index_i, agent_i]

                        # Choose aggregation method
                        if self.aggregation_method == "median":
                            state_v[state_index_i] = self.median_aggregate(
                                agent_i_in_messages,
                                own_value
                            )
                        elif self.aggregation_method == "median_window":
                            state_v[state_index_i] = self.median_window_aggregate(
                                agent_i_in_messages,
                                own_value
                            )
                        else:  # "trim"
                            state_v[state_index_i] = self.remove_extreme_D_average(
                                agent_i_in_messages,
                                own_value
                            )

        return state_v

    def adversarial_communication(self, state_x):
        dim = self.dim_action
        state_y = np.kron(state_x, np.ones([1,self.N]))

        for agent in self.random_agents:
            state_y[agent*dim:(agent + 1)*dim,:] = state_y[agent*dim:(agent + 1)*dim,:] + np.random.normal(0,10,[dim,self.N])
            state_y[agent*dim:(agent + 1)*dim,agent] = state_x[agent*dim:(agent+1)*dim].transpose()[0]
        for agent in self.constant_agents:
            state_y[agent*dim:(agent + 1)*dim,:] = np.ones([dim,self.N]) * (100)
            state_y[agent*dim:(agent + 1)*dim,agent] = state_x[agent*dim:(agent+1)*dim].transpose()[0]

        return state_y

    def iterate_algo(self, init_state):
        state_x = init_state

        err_record = [np.linalg.norm(self.NE - self.R.dot(state_x), 2)]
        pos_record = [self.R.dot(state_x)]

        diag = {
            "alpha": [],
            "grad_norm": [],
            "consensus_err": [],
            "y_corrupt_norm": [],
            "v_corrupt_norm": [],
        }

        state_v_prev = None
        beta = 0.9

        for i in range(self.sim_config.num_iter):
            if (i % 100) == 0:
                print(f"Iteration {i} of {self.sim_config.num_iter}")

            # step size schedule
            if self.sim_config.step_schedule == "sqrt":
                alpha_k = self.sim_config.step_size / np.sqrt(i + 1)
            else:
                alpha_k = self.sim_config.step_size
            diag["alpha"].append(float(alpha_k))

            # "honest broadcast" baseline
            baseline = np.kron(state_x, np.ones([1, self.N]))

            # adversarial communication + filtering
            state_y = self.adversarial_communication(state_x)
            state_v = self.filter_communicated_message(state_y)

            # compare with honest baseline (attack magnitude) and post-filter residual
            diag["y_corrupt_norm"].append(float(np.linalg.norm(state_y - baseline)))

            # best baseline for state_v: what filtering would do if there were no adversaries
            state_v_honest = self.filter_communicated_message(baseline)
            diag["v_corrupt_norm"].append(float(np.linalg.norm(state_v - state_v_honest)))

            # update
            if state_v_prev is None:
                y = state_v
            else:
                y = state_v + beta * (state_v - state_v_prev)

            Fv = self.F.dot(y)
            RFv = self.R.transpose().dot(Fv)
            state_x = state_v - alpha_k * (RFv + self.RB)
            state_v_prev = state_v

            # record realized action & error
            pos = self.R.dot(state_x)
            pos_record.append(pos)
            err_record.append(float(np.linalg.norm(self.NE - pos, 2)))

            # gradient norm at realized action
            g = self.A.dot(pos) + self.b
            diag["grad_norm"].append(float(np.linalg.norm(g)))

            # consensus error: agents' global estimates disagreement
            X = state_x.reshape(self.N, self.dim_action)     # (N, dim_action)
            Xbar = X.mean(axis=0, keepdims=True)
            diag["consensus_err"].append(float(np.sqrt(np.mean(np.sum((X - Xbar)**2, axis=1)))))

        return err_record, pos_record, state_x, diag

    def position_plot(self, pos_record, save=False, index_set=None, adversarial=None, out_dir=None, title=None, stats_text=None):
        figure = plt.figure()
        pos_record = np.reshape(pos_record, [-1, self.dim_action])

        indexs = range(self.N) if index_set is None else index_set
        adversarial = adversarial if adversarial is not None else []

        for i in indexs:
            if i in adversarial:
                plt.plot(pos_record[:, 2*i], pos_record[:, 2*i+1], '--')
            else:
                plt.plot(pos_record[:, 2*i], pos_record[:, 2*i+1])

        for i in range(self.N):
            plt.plot(self.NE[2*i], self.NE[2*i+1], marker='.', markersize=3, color="red")

        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        if title:
            plt.title(title)

        if stats_text:
            plt.gca().text(
                0.02, 0.98, stats_text,
                transform=plt.gca().transAxes,
                va='top', ha='left',
                bbox=dict(boxstyle="round", alpha=0.85)
            )

        # plt.show()
        if save:
            save_plot(figure, f"{self.example}", out_dir=out_dir)
        plt.close()

def plot_save_file_data(game, selected, adversarial, run_dir: Path):
    pos_records = []
    with open(run_dir / 'position_data.txt') as f:
        for line in f:
            data = [float(num) for num in line.split(',')]
            pos_records.append(np.array([data]).T)

    game.example = "position_plot_from_file"
    game.position_plot(pos_records, save=True, index_set=selected, adversarial=adversarial, out_dir=run_dir)

    err_record = []
    with open(run_dir / 'error_data.txt') as f:
        for line in f:
            err_record.append(float(line))

    with PlotErrorFigure('error_plot_from_file', out_dir=run_dir) as error_fig:
        plt.plot(err_record)

def plot_error_with_stats(err_record, run_dir: Path, stats_text: str, name="error_plot"):
    with PlotErrorFigure(name, out_dir=run_dir) as fig:
        plt.plot(err_record)
        plt.gca().text(
            0.02, 0.98, stats_text,
            transform=plt.gca().transAxes,
            va='top', ha='left',
            bbox=dict(boxstyle="round", alpha=0.85)
        )

def plot_mechanism(diag, run_dir: Path, name="diag_mechanism"):
    """
    consensus_err and grad_norm over time.
    """
    fig = plt.figure()
    it = np.arange(1, len(diag["grad_norm"]) + 1)

    plt.plot(it, diag["consensus_err"], label="consensus_err")
    plt.plot(it, diag["grad_norm"], label="grad_norm")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Value (log scale)")
    plt.legend()
    # plt.show()
    save_plot(fig, name, out_dir=run_dir)
    plt.close()

def plot_attack_filter(diag, run_dir: Path, name="diag_attack_filter"):
    """
    Corruption magnitude before filtering (y_corrupt_norm)
    and residual corruption after filtering (v_corrupt_norm).
    """
    fig = plt.figure()
    it = np.arange(1, len(diag["y_corrupt_norm"]) + 1)

    plt.plot(it, diag["y_corrupt_norm"], label="||state_y - baseline||")
    plt.plot(it, diag["v_corrupt_norm"], label="||state_v - state_v_honest||")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Corruption (log scale)")
    plt.legend()
    # plt.show()
    save_plot(fig, name, out_dir=run_dir)
    plt.close()

def main(base_dir, game, sim_config, thresholds=(1e-1, 1e-2, 1e-3)):
    init_state = -7 + 14*np.random.rand(game.dim_state, 1)

    # Create unique run directory
    run_dir = make_run_dir(base_dir=base_dir, game=game, sim_config=sim_config)

    # Save config snapshot
    cfg = gather_config_dict(game, sim_config)
    save_json(cfg, run_dir / "config.json")

    # Run rounds
    for r in range(sim_config.num_rounds):
        print(f'Executing round: {r} / {sim_config.num_rounds}')
        err_record, pos_record, last_iter, diag = game.iterate_algo(init_state)

        # Save raw data inside this run folder
        with open(run_dir / 'position_data.txt', 'a') as f:
            for data in pos_record:
                record = data.T
                record = list(record[0])
                f.write("%s\n" % ",".join([str(num) for num in record]))

        with open(run_dir / 'error_data.txt', 'a') as f:
            for data in err_record:
                f.write("%s\n" % data)

        with open(run_dir / 'last_state.txt', 'a') as f:
            data = last_iter.T
            f.write("%s\n" % ",".join([str(num) for num in data[0]]))

        # Compute stats and save
        stats = compute_run_stats(err_record, thresholds=thresholds)
        save_json(stats, run_dir / "stats.json")

        # Also append to a global summary CSV (across all runs)
        summary_row = {}
        summary_row.update(cfg["sim_config"])
        summary_row.update({
            "grid_width": cfg["game"]["grid_width"],
            "corner_size": cfg["game"]["corner_size"],
            "aggregation_method": cfg["game"]["aggregation_method"],
            "D": cfg["game"]["D"],
            "l_inf_ball": cfg["game"]["l_inf_ball"],
            "num_agents_N": cfg["game"]["num_agents_N"],
            "num_random_agents": len(cfg["game"]["random_agents"]),
            "num_constant_agents": len(cfg["game"]["constant_agents"]),
            "run_dir": str(run_dir),
        })
        summary_row.update(stats)

        # Keep field order stable
        fieldnames = list(summary_row.keys())
        append_csv_row(Path("runs") / "summary.csv", fieldnames, summary_row)

        # Prepare a compact stats text box for plots
        stats_text = (
            f"agg={cfg['game']['aggregation_method']}  D={cfg['game']['D']}  l_inf={cfg['game']['l_inf_ball']}\n"
            f"grid={cfg['game']['grid_width']}  N={cfg['game']['num_agents_N']}\n"
            f"random={len(cfg['game']['random_agents'])}  constant={len(cfg['game']['constant_agents'])}\n"
            f"step={cfg['sim_config']['step_schedule']}  alpha0={cfg['sim_config']['step_size']:.2e}\n"
            f"final={stats['final_error']:.3e}  best={stats['best_error']:.3e} @ {stats['iter_best_error']}\n"
            + " ".join([f"{k}:{v}" for k, v in stats.items() if k.startswith("first_below_")])
        )

        # Plots (saved in run folder)
        plot_error_with_stats(err_record, run_dir, stats_text, name="error_plot")

        # Position plot
        game.example = "position_plot"
        game.position_plot(
            pos_record,
            save=True,
            index_set=None,
            adversarial=sorted(list(game.random_agents | game.constant_agents)),
            out_dir=run_dir,
            title="Trajectories",
            stats_text=stats_text
        )

        # New diagnostic plots
        plot_mechanism(diag, run_dir, name="diag_mechanism")
        plot_attack_filter(diag, run_dir, name="diag_attack_filter")

        print(last_iter)
        init_state = last_iter

    return err_record, pos_record, last_iter, run_dir

if __name__ == "__main__":
    seed = 10
    random.seed(seed)
    np.random.seed(seed)

    sim_config = simulation_config()
    # Use decaying step size
    sim_config.step_size = 0.4
    sim_config.step_schedule = "sqrt"
    sim_config.num_iter = 40000
    selected = [0,1,2,3,4,5]
    aggregation_method = "median"
    median_window = 0.0
    use_geometric_median = True
    num_agents = 4
    base_dir = f"runs_constant_agents_value100_geom_med_only_seed_{seed}"
    loop = False

    random_agents = []
    constant_agents = []
    if loop:
        for agent_type in ["constant"]:
            for i in range(0, num_agents):
                if agent_type == "random":
                    random_agents.append(i)
                else:
                    constant_agents.append(i)
                adversarial = random_agents + constant_agents

                game = Resilient(
                    sim_config,
                    grid_width = 4,
                    random_agents = random_agents,
                    constant_agents = constant_agents,
                    l_inf_ball = 2,
                    D = 1,
                    corner_size = 1,
                    aggregation_method = aggregation_method,   # or "median" or "trim"
                    median_window = median_window,                   # tune this
                    use_geometric_median = use_geometric_median
                )

                # These are the old examples I used.
                # game = Resilient(sim_config, grid_width = 4, random_agents=None, constant_agents=None, l_inf_ball=1)
                # game = Resilient(sim_config, grid_width = 10, random_agents=set([4, 6, 11, 19, 26, 32, 38, 41]), constant_agents=None, l_inf_ball=2) # Fails
                # game = Resilient(sim_config, grid_width = 10, random_agents=set([5, 71, 8, 74, 10, 78, 17, 87, 28, 95, 46, 61]), constant_agents=None, l_inf_ball=2, D=3, corner_size = 1)
                # game = Resilient(sim_config, grid_width = 4, random_agents=random_agents, constant_agents=constant_agents, l_inf_ball=1, aggregation_method="median")

                # These are other examples where I wanted to make Gc != Go. 
                # grid_width = 10; game = Resilient(sim_config, grid_width = grid_width, random_agents=set([0,3,6,28,31,34,37,58,61,64,67,89,92,95]), constant_agents=None, l_inf_ball=1, D=1, corner_size = 1)
                # grid_width = 4; game = Resilient(sim_config, grid_width = grid_width, random_agents=set([0,11]), constant_agents=None, l_inf_ball=1, D=1, corner_size = 1)
                # grid_width = 5; game = Resilient(sim_config, grid_width = grid_width, random_agents=set([0,7,14]), constant_agents=None, l_inf_ball=1, D=1, corner_size = 1)
                # game.Go = irg.grid_l_one_to_adj_matrix(grid_width,1)
                # game.Go[1,grid_width], game.Go[grid_width,1] = 1, 1
                # game.Go[grid_width-2,2*grid_width - 1], game.Go[2*grid_width-1, grid_width-2] = 1, 1
                # game.Go[(grid_width-2)*grid_width, (grid_width-1)*grid_width + 1], game.Go[(grid_width-1)*grid_width + 1, (grid_width-2)*grid_width] = 1,1
                # game.Go[grid_width**2 - 2, grid_width**2 - grid_width -1], game.Go[grid_width**2 - grid_width -1, grid_width**2 - 2] = 1,1
                # game.Go = irg.remove_nodes_from_adj_matrix(game.Go, irg.get_corners(game.Go, 1))
                
                err_record, pos_record, last_iter, run_dir = main(base_dir, game, sim_config, thresholds=(1e-1, 1e-2, 1e-3))

                plot_save_file_data(game, selected, adversarial, run_dir)

            random_agents = []
            constant_agents = []
    else:
        constant_agents = [3,9]
        random_agents = [0,6,11]
        adversarial = random_agents + constant_agents
        base_dir = "runs"

        game = Resilient(
            sim_config,
            grid_width = 4,
            random_agents = random_agents,
            constant_agents = constant_agents,
            l_inf_ball = 2,
            D = 1,
            corner_size = 1,
            aggregation_method = "median",   # or "median" or "trim"
            median_window = median_window,                   # tune this
            use_geometric_median = use_geometric_median
        )

        err_record, pos_record, last_iter, run_dir = main(base_dir, game, sim_config, thresholds=(1e-1, 1e-2, 1e-3))

        plot_save_file_data(game, selected, adversarial, run_dir)
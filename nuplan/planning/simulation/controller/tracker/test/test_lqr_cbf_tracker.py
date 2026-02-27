"""
Standalone test that verifies the nuplan LQRCBFTracker against our example scenarios.

This mimics what nuplan's simulation loop does:
  1. Construct EgoState, trajectory, and neighbor TrackedObjects from raw scenario data
  2. Call track_trajectory() in a loop
  3. Propagate ego state via kinematic bicycle model
  4. Compare outputs with the standalone modular LQRTrackerCBF

Usage:
    cd /home/eku/yiwei/nuplan-devkit
    python -m nuplan.planning.simulation.controller.tracker.test.test_lqr_cbf_tracker
"""

import sys
import numpy as np
import torch
from typing import List, Tuple

from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.tracker.lqr_cbf import LQRCBFTracker
from nuplan.planning.simulation.controller.tracker.tracker_utils import (
    get_interpolated_reference_trajectory_poses,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

# Also import standalone modular tracker for comparison
sys.path.insert(0, '/home/eku/yiwei/AV_Diffusion_Planner_CBF')
from diffusion_planner.model.cbf_simple.lqr_tracker_cbf_modular import LQRTrackerCBF as ModularLQRTrackerCBF
from diffusion_planner.model.cbf_simple.test_safety import (
    get_simple_scenario_0, get_simple_scenario_1, get_simple_scenario_2,
    get_simple_scenario_3, get_simple_scenario_4,
    duration, v,
)
from diffusion_planner.model.cbf.MPC.vehicle_params import EGO_LENGTH, EGO_WIDTH

VEHICLE = get_pacifica_parameters()
DT = 0.1


def make_ego_state(x, y, heading, velocity, steering_angle, time_us):
    """Build an EgoState from raw values."""
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(x, y, heading),
        rear_axle_velocity_2d=StateVector2D(velocity, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_angle=steering_angle,
        time_point=TimePoint(time_us),
        vehicle_parameters=VEHICLE,
        is_in_auto_mode=True,
    )


def make_trajectory_from_poses(poses_xy_heading, dt, start_time_us=0):
    """Build an InterpolatedTrajectory from [N, 3] (x, y, heading) array."""
    ego_states = []
    for i, (x, y, h) in enumerate(poses_xy_heading):
        t_us = start_time_us + int(i * dt * 1e6)
        ego_states.append(make_ego_state(x, y, h, 0.0, 0.0, t_us))
    return InterpolatedTrajectory(ego_states)


def make_detections(neighbor_states, time_us):
    """Build TrackedObjects from a list of (x, y, heading, length, width) tuples."""
    agents = []
    for i, (nx, ny, nh, nl, nw) in enumerate(neighbor_states):
        box = OrientedBox(StateSE2(nx, ny, nh), nl, nw, 1.5)
        metadata = SceneObjectMetadata(
            timestamp_us=time_us, token=f"nbr_{i}", track_id=i, track_token=f"track_{i}",
        )
        agent = Agent(
            tracked_object_type=TrackedObjectType.VEHICLE,
            oriented_box=box,
            velocity=StateVector2D(0.0, 0.0),
            metadata=metadata,
        )
        agents.append(agent)
    return TrackedObjects(agents)


def propagate_bicycle(ego_x, ego_y, ego_heading, ego_v, ego_steering,
                      accel, steering_rate, dt, wheelbase):
    """Simple Euler bicycle integration (matches our standalone tracker)."""
    ego_x += ego_v * np.cos(ego_heading) * dt
    ego_y += ego_v * np.sin(ego_heading) * dt
    ego_heading += ego_v * np.tan(ego_steering) / wheelbase * dt
    ego_v += accel * dt
    ego_v = max(ego_v, 0.0)
    ego_steering += steering_rate * dt
    return ego_x, ego_y, ego_heading, ego_v, ego_steering


def run_nuplan_tracker_on_scenario(x_np, v0_val, nei_sizes=None, cbf_alpha=0.07):
    """
    Run the nuplan LQRCBFTracker on a scenario.

    Args:
        x_np: [B, P, H, 4] numpy array (x, y, cos, sin)
        v0_val: initial ego velocity
        nei_sizes: list of (L, W) per neighbor
    Returns:
        out_traj: [H, 4] tracked ego trajectory
        u_hist: [H, 2] (accel, steering_rate)
    """
    B, P, H, _ = x_np.shape
    dt = DT

    if nei_sizes is None:
        nei_sizes = [(EGO_LENGTH, EGO_WIDTH)] * (P - 1)

    # Build reference trajectory from ego poses
    ref = x_np[0, 0]  # [H, 4]
    ref_heading = np.arctan2(ref[:, 3], ref[:, 2])
    ref_poses = np.column_stack([ref[:, :2], ref_heading])

    trajectory = make_trajectory_from_poses(ref_poses, dt)

    # Create tracker
    tracker = LQRCBFTracker(
        q_longitudinal=[10.0],
        r_longitudinal=[1.0],
        q_lateral=[1.0, 10.0, 0.0],
        r_lateral=[1.0],
        discretization_time=dt,
        tracking_horizon=10,
        jerk_penalty=1e-4,
        curvature_rate_penalty=1e-2,
        stopping_proportional_gain=0.5,
        stopping_velocity=0.2,
        cbf_alpha=cbf_alpha,
    )

    # Initialize ego state
    ego_x = float(ref[0, 0])
    ego_y = float(ref[0, 1])
    ego_heading = float(ref_heading[0])
    ego_v = float(v0_val)
    ego_steering = 0.0
    wheelbase = VEHICLE.wheel_base

    out_traj = np.zeros((H, 4), dtype=np.float64)
    out_traj[0] = ref[0]
    u_hist = np.zeros((H, 2), dtype=np.float64)

    for k in range(H - 1):
        time_us = int(k * dt * 1e6)
        next_time_us = int((k + 1) * dt * 1e6)

        # Build ego state
        ego_state = make_ego_state(ego_x, ego_y, ego_heading, ego_v, ego_steering, time_us)

        # Build neighbor detections
        neighbor_states_current = []
        neighbor_states_next = []
        for m in range(P - 1):
            nbr_k = min(k, H - 1)
            nbr_k1 = min(k + 1, H - 1)

            nbr_heading = np.arctan2(
                float(x_np[0, m + 1, nbr_k, 3]),
                float(x_np[0, m + 1, nbr_k, 2]),
            )
            nbr_heading1 = np.arctan2(
                float(x_np[0, m + 1, nbr_k1, 3]),
                float(x_np[0, m + 1, nbr_k1, 2]),
            )
            nl, nw = nei_sizes[m]

            neighbor_states_current.append((
                float(x_np[0, m + 1, nbr_k, 0]),
                float(x_np[0, m + 1, nbr_k, 1]),
                nbr_heading, float(nl), float(nw),
            ))
            neighbor_states_next.append((
                float(x_np[0, m + 1, nbr_k1, 0]),
                float(x_np[0, m + 1, nbr_k1, 1]),
                nbr_heading1, float(nl), float(nw),
            ))

        current_detections = make_detections(neighbor_states_current, time_us)
        next_detections = make_detections(neighbor_states_next, next_time_us)
        tracker.update_observations(current_detections, next_detections)

        # Call track_trajectory
        current_iter = SimulationIteration(TimePoint(time_us), k)
        next_iter = SimulationIteration(TimePoint(next_time_us), k + 1)

        dynamic_state = tracker.track_trajectory(current_iter, next_iter, ego_state, trajectory)

        accel = dynamic_state.rear_axle_acceleration_2d.x
        steering_rate = dynamic_state.tire_steering_rate

        u_hist[k, 0] = accel
        u_hist[k, 1] = steering_rate

        # Propagate with simple bicycle (matching standalone tracker)
        ego_x, ego_y, ego_heading, ego_v, ego_steering = propagate_bicycle(
            ego_x, ego_y, ego_heading, ego_v, ego_steering,
            accel, steering_rate, dt, wheelbase,
        )

        out_traj[k + 1, 0] = ego_x
        out_traj[k + 1, 1] = ego_y
        out_traj[k + 1, 2] = np.cos(ego_heading)
        out_traj[k + 1, 3] = np.sin(ego_heading)

    return out_traj, u_hist


def run_modular_tracker_on_scenario(x_np, v0_val, nei_sizes=None, cbf_alpha=0.07):
    """Run the standalone modular tracker for comparison."""
    tracker = ModularLQRTrackerCBF(dt=DT, cbf_alpha=cbf_alpha)
    x_opt, u_hist, status = tracker(x_np, [v0_val], nei_sizes=nei_sizes)
    return x_opt[0, 0], u_hist[0]  # ego traj [H, 4], ego controls [H, 2]


def compare_results(name, nuplan_traj, modular_traj, nuplan_u, modular_u):
    """Compare nuplan and modular tracker outputs."""
    pos_diff = np.linalg.norm(nuplan_traj[:, :2] - modular_traj[:, :2], axis=1)
    max_pos_diff = np.max(pos_diff)
    mean_pos_diff = np.mean(pos_diff)

    accel_diff = np.abs(nuplan_u[:, 0] - modular_u[:, 0])
    max_accel_diff = np.max(accel_diff)

    sr_diff = np.abs(nuplan_u[:, 1] - modular_u[:, 1])
    max_sr_diff = np.max(sr_diff)

    passed = max_pos_diff < 1.0  # Allow some divergence due to nuplan vs standalone differences

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}:")
    print(f"    Position: max_diff={max_pos_diff:.4f}m, mean_diff={mean_pos_diff:.4f}m")
    print(f"    Accel:    max_diff={max_accel_diff:.4f}m/s^2")
    print(f"    SteerR:   max_diff={max_sr_diff:.6f}rad/s")
    return passed


if __name__ == "__main__":
    N = int(duration / DT)
    num_simple_scene = 4
    cbf_alpha = 0.07

    scenarios = {
        0: ("Stationary blocker", get_simple_scenario_0),
        1: ("T-bone crossing", get_simple_scenario_1),
        2: ("Left turn intersection", get_simple_scenario_2),
        3: ("Lane change", get_simple_scenario_3),
        4: ("Adjacent lane stopped", get_simple_scenario_4),
    }

    all_passed = True
    print("=" * 60)
    print("Testing nuplan LQRCBFTracker vs modular LQRTrackerCBF")
    print("=" * 60)

    for idx, (name, get_scenario) in scenarios.items():
        x = get_scenario(N, DT)
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x

        P = x_np.shape[1]
        if idx == 0:
            v0 = v
        else:
            v0 = v

        nei_sizes = [(EGO_LENGTH, EGO_WIDTH)] * (P - 1)

        print(f"\n--- Scenario {idx}: {name} (P={P}) ---")

        # Run both trackers
        nuplan_traj, nuplan_u = run_nuplan_tracker_on_scenario(
            x_np, v0, nei_sizes=nei_sizes, cbf_alpha=cbf_alpha,
        )
        modular_traj, modular_u = run_modular_tracker_on_scenario(
            x_np, v0, nei_sizes=nei_sizes, cbf_alpha=cbf_alpha,
        )

        passed = compare_results(name, nuplan_traj, modular_traj, nuplan_u, modular_u)
        all_passed = all_passed and passed

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL SCENARIOS PASSED")
    else:
        print("SOME SCENARIOS FAILED")
    print("=" * 60)

"""
Tracker that replays pre-computed (accel, steering_rate) actions from
CBF modules (lqr_tracker_cbf_modular, QP_solver, etc.) through nuplan's
motion model.

Any CBF module can call set_latest_action() to store its first-step action.
"""

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.planning.simulation.controller.tracker.abstract_tracker import AbstractTracker
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

# Shared action variable: numpy [2] (accel, steering_rate)
_latest_action = None


def set_latest_action(action):
    """Set the latest action. Called by CBF modules after solving."""
    global _latest_action
    _latest_action = action


def get_latest_action():
    """Get the latest action."""
    return _latest_action


class ActionReplayTracker(AbstractTracker):
    """Replays a single pre-computed (accel, steering_rate) action per tick."""

    def track_trajectory(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        initial_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> DynamicCarState:
        """Inherited, see superclass."""
        global _latest_action
        assert _latest_action is not None, "_latest_action is not set"

        accel = float(_latest_action[0])
        steering_rate = float(_latest_action[1])

        # Consume the action to prevent stale reuse
        _latest_action = None

        return DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=initial_state.car_footprint.rear_axle_to_center_dist,
            rear_axle_velocity_2d=initial_state.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=StateVector2D(accel, 0),
            tire_steering_rate=steering_rate,
        )
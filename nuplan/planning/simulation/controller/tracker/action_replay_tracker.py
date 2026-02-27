"""
Tracker that replays pre-computed (accel, steering_rate) actions from the
LQR-CBF module through nuplan's motion model.

Reads the latest action from diffusion_planner.model.cbf_simple.lqr_tracker_cbf_modular.latest_action.
"""

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.planning.simulation.controller.tracker.abstract_tracker import AbstractTracker
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


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
        from diffusion_planner.model.cbf_simple.lqr_tracker_cbf_modular import get_latest_action

        action = get_latest_action()
        assert action is not None

        accel = float(action[0])
        steering_rate = float(action[1])

        return DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=initial_state.car_footprint.rear_axle_to_center_dist,
            rear_axle_velocity_2d=initial_state.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=StateVector2D(accel, 0),
            tire_steering_rate=steering_rate,
        )

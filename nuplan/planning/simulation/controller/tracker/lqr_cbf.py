"""
LQR tracker with CBF (Control Barrier Function) collision avoidance.

Subclasses nuplan's LQRTracker. The CBF layer solves a lightweight QP at each
timestep to modify the longitudinal acceleration for safety, while steering
rate passes through from LQR unchanged.

Usage:
    The tracker needs neighbor observations to compute CBF constraints.
    Call `update_observations(detections)` before each `track_trajectory()` call.
    This is done automatically by `LQRCBFTwoStageController`, or manually
    in standalone testing.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.tracker.tracker_utils import (
    _generate_profile_from_initial_condition_and_derivatives,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

logger = logging.getLogger(__name__)

# Lazy import of CasADi + CBF filter — only loaded when the class is instantiated.
_CBFAccelFilter = None


def _get_cbf_filter_class():
    """Lazy import to avoid hard CasADi dependency in nuplan unless CBF tracker is used."""
    global _CBFAccelFilter
    if _CBFAccelFilter is None:
        from diffusion_planner.model.cbf_simple.lqr_tracker_cbf_modular import CBFAccelFilter
        _CBFAccelFilter = CBFAccelFilter
    return _CBFAccelFilter


class LQRCBFTracker(LQRTracker):
    """
    LQR tracker augmented with a CBF safety filter on longitudinal acceleration.

    Inherits all LQR logic from LQRTracker. Overrides `track_trajectory` to:
    1. Compute LQR accel + steering_rate (via parent methods)
    2. Apply CBF QP to get safe accel
    3. Return safe accel + LQR steering_rate

    Neighbor observations must be supplied via `update_observations()` before
    each call to `track_trajectory()`.
    """

    def __init__(
        self,
        # LQR params (forwarded to parent)
        q_longitudinal: npt.NDArray[np.float64],
        r_longitudinal: npt.NDArray[np.float64],
        q_lateral: npt.NDArray[np.float64],
        r_lateral: npt.NDArray[np.float64],
        discretization_time: float,
        tracking_horizon: int,
        jerk_penalty: float,
        curvature_rate_penalty: float,
        stopping_proportional_gain: float,
        stopping_velocity: float,
        vehicle: VehicleParameters = get_pacifica_parameters(),
        # CBF params
        cbf_alpha: float = 0.07,
        cbf_slack_weight: float = 1e4,
        cbf_max_neighbors: int = 10,
        cbf_safety_margin: float = 0.0,
    ):
        super().__init__(
            q_longitudinal=q_longitudinal,
            r_longitudinal=r_longitudinal,
            q_lateral=q_lateral,
            r_lateral=r_lateral,
            discretization_time=discretization_time,
            tracking_horizon=tracking_horizon,
            jerk_penalty=jerk_penalty,
            curvature_rate_penalty=curvature_rate_penalty,
            stopping_proportional_gain=stopping_proportional_gain,
            stopping_velocity=stopping_velocity,
            vehicle=vehicle,
        )

        CBFAccelFilter = _get_cbf_filter_class()
        self._cbf = CBFAccelFilter(
            dt=discretization_time,
            wheelbase=vehicle.wheel_base,
            cbf_alpha=cbf_alpha,
            slack_weight=cbf_slack_weight,
            max_neighbors=cbf_max_neighbors,
            safety_margin=cbf_safety_margin,
        )

        # Current neighbor observations (set externally before each track_trajectory call)
        self._current_detections: Optional[TrackedObjects] = None
        self._next_detections: Optional[TrackedObjects] = None

    def update_observations(
        self,
        current_detections: TrackedObjects,
        next_detections: Optional[TrackedObjects] = None,
    ) -> None:
        """
        Provide neighbor observations for the CBF constraint computation.

        :param current_detections: Tracked objects at the current timestep.
        :param next_detections: Tracked objects at the next timestep (for CBF lookahead).
            If None, current_detections is used (constant-velocity assumption is implicit
            in the barrier linearization).
        """
        self._current_detections = current_detections
        self._next_detections = next_detections if next_detections is not None else current_detections

    def track_trajectory(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        initial_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> DynamicCarState:
        """
        Compute LQR controls then apply CBF safety filter on acceleration.

        If no observations have been provided via update_observations(), falls
        back to pure LQR (no CBF filtering).
        """
        # --- Step 1: Compute LQR accel and steering rate (reuse parent internals) ---
        initial_velocity, initial_lateral_state_vector = self._compute_initial_velocity_and_lateral_state(
            current_iteration, initial_state, trajectory
        )

        reference_velocity, curvature_profile = self._compute_reference_velocity_and_curvature_profile(
            current_iteration, trajectory
        )

        should_stop = reference_velocity <= self._stopping_velocity and initial_velocity <= self._stopping_velocity

        if should_stop:
            accel_cmd, steering_rate_cmd = self._stopping_controller(initial_velocity, reference_velocity)
        else:
            accel_cmd = self._longitudinal_lqr_controller(initial_velocity, reference_velocity)
            velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
                initial_condition=initial_velocity,
                derivatives=np.ones(self._tracking_horizon, dtype=np.float64) * accel_cmd,
                discretization_time=self._discretization_time,
            )[: self._tracking_horizon]
            steering_rate_cmd = self._lateral_lqr_controller(
                initial_lateral_state_vector, velocity_profile, curvature_profile
            )

        # --- Step 2: Apply CBF safety filter ---
        if self._current_detections is not None and not should_stop:
            accel_cmd = self._apply_cbf_filter(
                accel_des=accel_cmd,
                initial_state=initial_state,
            )

        return DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=initial_state.car_footprint.rear_axle_to_center_dist,
            rear_axle_velocity_2d=initial_state.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=StateVector2D(accel_cmd, 0),
            tire_steering_rate=steering_rate_cmd,
        )

    def _apply_cbf_filter(
        self,
        accel_des: float,
        initial_state: EgoState,
    ) -> float:
        """
        Extract neighbor geometry from TrackedObjects, build CBF constraints,
        and solve QP to get safe acceleration.
        """
        ego_x = initial_state.rear_axle.x
        ego_y = initial_state.rear_axle.y
        ego_heading = initial_state.rear_axle.heading
        ego_v = initial_state.dynamic_car_state.rear_axle_velocity_2d.x
        ego_steering = initial_state.tire_steering_angle

        # Extract vehicle neighbors from current and next detections
        neighbors_current = self._extract_neighbors(self._current_detections)
        neighbors_next = self._extract_neighbors(self._next_detections)

        # Match neighbors by count (pad or trim next to match current)
        n = min(len(neighbors_current), len(neighbors_next))
        neighbors_current = neighbors_current[:n]
        neighbors_next = neighbors_next[:n]

        if n == 0:
            return accel_des

        accel_safe, n_slack, status = self._cbf.filter_accel(
            accel_des, ego_x, ego_y, ego_heading, ego_v, ego_steering,
            neighbors_current, neighbors_next,
        )

        if 'Succeeded' not in status and 'Success' not in status:
            logger.warning(f"CBF QP status: {status}, falling back to LQR accel")
            return accel_des

        return accel_safe

    # Q. Why is this necessary?
    @staticmethod
    def _extract_neighbors(detections: TrackedObjects) -> List[Tuple[float, float, float, float, float]]:
        """
        Extract neighbor (x, y, heading, half_spine, radius) tuples from TrackedObjects.

        Uses the box center as the capsule center, length/2 as half_spine, width/2 as radius.
        Only considers VEHICLE type objects.
        """
        neighbors = []
        for obj in detections.tracked_objects:
            if obj.tracked_object_type != TrackedObjectType.VEHICLE:
                continue
            center = obj.box.center
            neighbors.append((
                float(center.x),
                float(center.y),
                float(center.heading),
                float(obj.box.length) / 2.0,
                float(obj.box.width) / 2.0,
            ))
        return neighbors

"""
Two-stage controller that feeds tracked-object observations to an LQRCBFTracker.

Drop-in replacement for TwoStageController when the tracker is LQRCBFTracker.
The only difference is that `update_state` also forwards the current detections
to the tracker via `update_observations()` before computing control actions.
"""

import logging
import os
import time
from typing import Optional

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.controller.motion_model.abstract_motion_model import AbstractMotionModel
from nuplan.planning.simulation.controller.tracker.lqr_cbf import LQRCBFTracker
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


class LQRCBFTwoStageController(AbstractEgoController):
    """
    Two-stage controller (tracker + motion model) that supplies neighbour
    detections to an LQRCBFTracker before each tracking step.
    """

    def __init__(
        self,
        scenario: AbstractScenario,
        tracker: LQRCBFTracker,
        motion_model: AbstractMotionModel,
        observations: AbstractObservation,
    ):
        self._scenario = scenario
        self._tracker = tracker
        self._motion_model = motion_model
        self._observations = observations
        self._current_state: Optional[EgoState] = None
        self._verbose = os.environ.get("VERBOSE") == "1"
        self._logger = logging.getLogger(__name__)

    def reset(self) -> None:
        """Inherited, see superclass."""
        self._current_state = None

    def get_state(self) -> EgoState:
        """Inherited, see superclass."""
        if self._current_state is None:
            self._current_state = self._scenario.initial_ego_state
        return self._current_state

    def update_state(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        ego_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> None:
        """Inherited, see superclass."""
        t0 = time.perf_counter() if self._verbose else None

        # Feed current detections to the CBF tracker
        observation = self._observations.get_observation()
        if isinstance(observation, DetectionsTracks):
            self._tracker.update_observations(observation.tracked_objects)

        sampling_time = next_iteration.time_point - current_iteration.time_point

        dynamic_state = self._tracker.track_trajectory(
            current_iteration, next_iteration, ego_state, trajectory
        )

        self._current_state = self._motion_model.propagate_state(
            state=ego_state, ideal_dynamic_state=dynamic_state, sampling_time=sampling_time
        )

        if t0 is not None:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._logger.info(f"[LQR-CBF Controller] update_state: {elapsed_ms:.1f} ms")

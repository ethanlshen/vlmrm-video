from gymnasium import utils
from typing import Dict, Optional, Tuple
from numpy.typing import NDArray

from sofa_env.scenes.deflect_spheres.deflect_spheres_env import DeflectSpheresEnv, ObservationType, Mode, RenderMode


class CLIPRewardedDeflectSpheresEnv(DeflectSpheresEnv):
    def __init__(
        self,
        episode_length: int,
        num_deflect_to_win: int = 1,
        mode: Mode = Mode.WITH_REPLACEMENT,
        observation_type: ObservationType = ObservationType.STATE,
        image_shape: Tuple[int, int] = (480, 480),
        **kwargs,
    ) -> None:

        gt_reward_config = {
            "action_violated_cartesian_workspace": -0.0,
            "action_violated_state_limits": -0.0,
            "tool_collision": -0.0,
            "distance_to_active_sphere": -0.0,
            "delta_distance_to_active_sphere": -5.0,
            "deflection_of_inactive_spheres": -0.005,
            "deflection_of_active_sphere": 0.0,
            "delta_deflection_of_active_sphere": 1.0,
            "done_with_active_sphere": 10.0,
            "successful_task": 100.0,
        }

        utils.EzPickle.__init__(
            self,
            episode_length=episode_length,
            num_deflect_to_win=num_deflect_to_win,
            mode=mode,
            observation_type=observation_type,
            image_shape=image_shape,
            reward_amount_dict=gt_reward_config,
            **kwargs,
        )

        # Remove render_mode from kwargs
        kwargs.pop("render_mode", None)
        super().__init__(
            num_deflect_to_win=num_deflect_to_win,
            mode=mode,
            observation_type=observation_type,
            image_shape=image_shape,
            render_mode=RenderMode.HEADLESS,
            reward_amount_dict=gt_reward_config,
            **kwargs,
        )

        self.episode_length = episode_length
        self.num_steps = 0

    def step(self, action) -> Tuple[NDArray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = super().step(action)

        self.num_steps += 1
        terminated = self.num_steps >= self.episode_length

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.num_steps = 0
        return super().reset(seed=seed, options=options)

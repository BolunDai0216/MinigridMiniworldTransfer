import numpy as np
from gymnasium import spaces, utils
from miniworld.entity import Box
from miniworld.manual_control import ManualControl
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS


class MiniworldGridEnv(MiniWorldEnv, utils.EzPickle):
    def __init__(self, size=10, max_episode_steps=180, **kwargs):
        assert size >= 2
        self.size = size

        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set("forward_step", 1.0)
        params.set("turn_step", 90)

        MiniWorldEnv.__init__(
            self, params=params, max_episode_steps=max_episode_steps, **kwargs
        )
        utils.EzPickle.__init__(
            self, size=size, max_episode_steps=max_episode_steps, **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)

        self.box = self.place_entity(
            Box(color="red"), pos=np.array([1.5, 0.5, 5.5]), dir=0.0
        )
        self.place_agent(pos=np.array([1.5, 0.5, 1.5]), dir=0.0)

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info


def main():
    env = MiniworldGridEnv(view="top", render_mode="human")

    manual_control = ManualControl(env, True, True)
    manual_control.run()


if __name__ == "__main__":
    main()

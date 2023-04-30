from pdb import set_trace

import numpy as np
from gymnasium import spaces, utils
from miniworld.entity import COLOR_NAMES, Box
from miniworld.manual_control import ManualControl
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS


class MiniworldGridEnv(MiniWorldEnv, utils.EzPickle):
    def __init__(self, size=9, max_episode_steps=100, **kwargs):
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

        colors = np.random.choice(COLOR_NAMES, size=4, replace=False)

        self.box_left = self.place_entity(
            Box(color=colors[0]), pos=np.array([0.5, 0.5, 4.5]), dir=0.0
        )
        self.box_bottom = self.place_entity(
            Box(color=colors[1]), pos=np.array([4.5, 0.5, 8.5]), dir=0.0
        )
        self.box_right = self.place_entity(
            Box(color=colors[2]), pos=np.array([8.5, 0.5, 4.5]), dir=0.0
        )
        self.box_up = self.place_entity(
            Box(color=colors[3]), pos=np.array([4.5, 0.5, 0.5]), dir=0.0
        )

        self.boxes = [self.box_left, self.box_bottom, self.box_right, self.box_up]

        # Select a random target door
        boxIdx = np.random.choice(4, size=1)[0]
        self.target_box = self.boxes[boxIdx]
        self.target_color = colors[boxIdx]

        # Generate the mission string
        self.mission = f"go to the {self.target_color} box"

        self.place_agent(pos=np.array([4.5, 0.5, 4.5]), dir=0.0)

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        ax, ay = self.agent.pos[0], self.agent.pos[2]
        tx, ty = self.target_box.pos[0], self.target_box.pos[2]

        if action == self.actions.done:
            if (ax == tx and int(abs(ay - ty)) == 1) or (
                ay == ty and int(abs(ax - tx)) == 1
            ):
                reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info


def main():
    env = MiniworldGridEnv(view="top", render_mode="human")

    manual_control = ManualControl(env, True, True)
    manual_control.run()


if __name__ == "__main__":
    main()

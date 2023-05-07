from pdb import set_trace

import numpy as np
from gymnasium import spaces, utils
from miniworld.entity import COLOR_NAMES, Ball, Box, Key
from miniworld.manual_control import ManualControl
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS


class MiniworldGoToObjEnv(MiniWorldEnv, utils.EzPickle):
    def __init__(self, size=9, max_episode_steps=100, fast=True, **kwargs):
        assert size >= 2
        self.size = size

        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()

        if fast:
            params.set("forward_step", 0.9)
            params.set("turn_step", 90)
        else:
            params.set("forward_step", 0.3)
            params.set("turn_step", 30)

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
        ObjList = [
            {
                "name": "ball",
                "obj": Ball,
            },
            {
                "name": "box",
                "obj": Box,
            },
            {
                "name": "key",
                "obj": Key,
            },
        ]
        _objs = np.random.choice(ObjList, size=4, replace=True)
        positions = [
            np.array([0.9, 0.5, 4.5]),
            np.array([4.5, 0.5, 8.1]),
            np.array([8.1, 0.5, 4.5]),
            np.array([4.5, 0.5, 0.9]),
        ]

        self.objs = []
        for i in range(4):
            if _objs[i]["name"] == "key":
                self.objs.append(
                    self.place_entity(
                        _objs[i]["obj"](color=colors[i]),
                        pos=positions[i],
                        dir=i * np.pi / 2,
                    )
                )
            else:
                self.objs.append(
                    self.place_entity(
                        _objs[i]["obj"](color=colors[i], size=0.5),
                        pos=positions[i],
                        dir=i * np.pi / 2,
                    )
                )

        # Select a random target object
        ObjIdx = np.random.choice(4, size=1)[0]
        self.target_obj = self.objs[ObjIdx]
        self.target_obj_name = _objs[ObjIdx]["name"]
        self.target_color = colors[ObjIdx]

        # Generate the mission string
        self.mission = f"go to the {self.target_color} {self.target_obj_name}"

        self.place_agent(pos=np.array([4.5, 0.5, 4.5]), dir=0.0)

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        ax, ay = self.agent.pos[0], self.agent.pos[2]
        tx, ty = self.target_obj.pos[0], self.target_obj.pos[2]

        next_ax = ax + 0.8 * np.cos(self.agent.dir)
        next_ay = ay - 0.8 * np.sin(self.agent.dir)

        _dis = np.sqrt((next_ax - tx) ** 2 + (next_ay - ty) ** 2)

        if _dis <= 0.2:
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info


def main():
    env = MiniworldGoToObjEnv(view="top", render_mode="human", manual_control=True)

    manual_control = ManualControl(env, True, True)
    manual_control.run()


if __name__ == "__main__":
    main()

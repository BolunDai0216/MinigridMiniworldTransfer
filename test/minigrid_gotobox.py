from __future__ import annotations

from pdb import set_trace

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class GoToBoxEnv(MiniGridEnv):
    def __init__(
        self,
        size=11,
        agent_start_pos=(5, 5),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.size = size
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES],
        )

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str):
        return f"go to the {color} box"

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate the 4 doors at random positions
        boxPos = []
        boxPos.append((5, 1))
        boxPos.append((1, 5))
        boxPos.append((9, 5))
        boxPos.append((5, 9))

        # Generate the door colors
        boxColors = []
        while len(boxColors) < len(boxPos):
            color = self._rand_elem(COLOR_NAMES)
            if color in boxColors:
                continue
            boxColors.append(color)

        # Place the doors in the grid
        for idx, pos in enumerate(boxPos):
            color = boxColors[idx]
            self.grid.set(*pos, Box(color))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Select a random target door
        boxIdx = self._rand_int(0, len(boxPos))
        self.target_pos = boxPos[boxIdx]
        self.target_color = boxColors[boxIdx]

        # Generate the mission string
        self.mission = f"go to the {self.target_color} box"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        if self.agent_dir == 0:
            next_ax = ax + 1
            next_ay = ay
        elif self.agent_dir == 1:
            next_ax = ax
            next_ay = ay + 1
        elif self.agent_dir == 2:
            next_ax = ax - 1
            next_ay = ay
        elif self.agent_dir == 3:
            next_ax = ax
            next_ay = ay - 1

        # Don't let the agent open any of the doors
        if action == self.actions.toggle:
            terminated = True

        # Reward performing done action in front of the target door
        # if action == self.actions.done:
        if next_ax == tx and next_ay == ty:
            reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info


def main():
    env = GoToBoxEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()


if __name__ == "__main__":
    main()

"""Baseline embodiment with all body scales fixed at one."""

import mujoco

from myoacross.embodiment.model import EmbodimentModel


class BaselineModel(EmbodimentModel):
    """Keep the original MyoFullBody dimensions."""

    def solve(self) -> dict[str, float]:
        """Return unit scales for every non-world baseline body."""

        body_scales: dict[str, float] = {}

        for body_id in range(1, self.baseline_model.nbody):
            body_name = mujoco.mj_id2name(
                self.baseline_model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )
            if not body_name:
                raise ValueError(f"MyoFullBody body {body_id} does not have a name")

            body_scales[body_name] = 1.0

        return body_scales

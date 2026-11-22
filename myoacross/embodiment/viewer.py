"""Display an embodiment."""

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer


def view_embodiment(
    xml_path: str | Path,
    *,
    fps: float = 60.0,
) -> None:
    """Display the model at its default configuration."""
    xml_path = Path(xml_path).expanduser().resolve()

    if not xml_path.is_file():
        raise FileNotFoundError(f"Model XML not found: {xml_path}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Display FPS must be finite and positive")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    reference_qpos = model.qpos0.copy()
    mujoco.mj_forward(model, data)

    frame_interval = 1.0 / fps

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            frame_start = time.perf_counter()

            with viewer.lock():
                data.qpos[:] = reference_qpos
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)

            viewer.sync()

            remaining = frame_interval - (time.perf_counter() - frame_start)

            if remaining > 0:
                time.sleep(remaining)

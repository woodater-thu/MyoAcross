"""Display embodiments, retargeted motions, or policy rollouts."""

import argparse
import math
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model",
        type=Path,
        help="Path to the embodiment model.xml.",
    )
    parser.add_argument(
        "--motion",
        type=Path,
        help="Path to motion.npz; requires --model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="RL checkpoint for policy-controlled playback.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for policy inference; default: cuda:0.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Display refresh rate in Hz.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Reference playback speed; policy speed uses viewer controls.",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Loop reference playback; policies automatically reset episodes.",
    )

    parser.add_argument(
        "--viewer", choices=("native", "viser"), default="native",
        help="Policy viewer backend; viser serves a browser UI.",
    )
    parser.add_argument(
        "--camera", choices=("front-side",), default=None,
        help="Optional policy camera preset; otherwise retain the viewer defaults.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Viser listen address; default: loopback for SSH forwarding.",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Viser HTTP/WebSocket port; default: 8080.",
    )

    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.viewer == "viser" and args.checkpoint is None:
        parser.error("--viewer viser currently requires --checkpoint")

    # All stages require an explicit model.
    if args.model is None:
        parser.error("--model is required")

    # Policy rollout also requires a reference motion.
    if args.checkpoint is not None and args.motion is None:
        parser.error("--checkpoint requires --motion")

    # Validate display and playback options.
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if args.checkpoint is not None and (args.speed != 1.0 or not args.loop):
        parser.error(
            "Policy playback uses automatic episode resets and the mjlab "
            "viewer's interactive speed controls; --speed and --no-loop "
            "currently apply to reference playback."
        )

    # Resolve input paths before loading stage-specific dependencies.
    for name in ("model", "motion", "checkpoint"):
        path = getattr(args, name)
        if path is None:
            continue

        path = path.expanduser().resolve()
        if not path.is_file():
            parser.error(f"--{name}: file not found: {path}")

        setattr(args, name, path)

    # Policy-controlled simulation.
    if args.checkpoint is not None:
        from myoacross.rl.viewer import view_policy
        view_policy(
            xml_path=args.model,
            motion_path=args.motion,
            checkpoint_path=args.checkpoint,
            fps=args.fps,
            device=args.device,
            viewer=args.viewer,
            camera=args.camera,
            host=args.host,
            port=args.port,
        )

    # Kinematic motion playback.
    elif args.motion is not None:
        from myoacross.retarget.viewer import view_motion
        view_motion(
            xml_path=args.model,
            motion_path=args.motion,
            fps=args.fps,
            speed=args.speed,
            loop=args.loop,
        )

    # Static embodiment inspection.
    else:
        from myoacross.embodiment.viewer import view_embodiment
        view_embodiment(
            xml_path=args.model,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()

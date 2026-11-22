"""Unified command-line entry point for MyoAcross."""

from __future__ import annotations

import argparse
from importlib import import_module


COMMANDS = {
    "measure": (
        "myoacross.embodiment.smplh_to_measurements",
        "Generate subject measurements from SMPL shape with SKEL.",
    ),
    "embodiment": (
        "myoacross.embodiment.generate",
        "Generate subject-specific embodiments.",
    ),
    "retarget": (
        "myoacross.retarget.run",
        "Retarget human motions to MyoFullBody.",
    ),
    "rl": (
        "myoacross.rl.run",
        "Generate embodiments, train policies, or evaluate policies.",
    ),
    "play": (
        "myoacross.play",
        "Inspect embodiments or play motions.",
    ),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="myoacross",
        description="Subject-specific embodiment and morphology-conditioned control.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    for name, (_, description) in COMMANDS.items():
        # Each module owns its arguments and detailed help.
        subparsers.add_parser(
            name,
            help=description,
            add_help=False,
        )

    args, remaining = parser.parse_known_args(argv)

    module_name, _ = COMMANDS[args.command]
    module = import_module(module_name)
    module.main(remaining)
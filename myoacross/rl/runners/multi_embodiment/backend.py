"""Copy (never modify) installed mujoco_warp and adapt touch volumes per world.

Run with the server venv Python, then prepend output directory to PYTHONPATH.
The shared (1-D) site_size input remains supported for the unchanged K=1 path.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from .zero_tendon import MARKER as TENDON_MARKER, ZERO_FIELDS


MARKER = "MYOACROSS_BATCHED_SITE_SIZE"


def adapt_sensor_source(text: str) -> str:
    """Fail closed if any expected declaration, index or call site has changed."""
    if MARKER in text:
        raise ValueError("Backend source is already adapted; use the original installed package")
    replacements = (
        ("site_size: wp.array[wp.vec3]", "site_size: wp.array2d[wp.vec3]", 3),
        ("site_size[objid]", "site_size[worldid % site_size.shape[0], objid]", 2),
        ("site_size[refid]", "site_size[worldid % site_size.shape[0], refid]", 1),
        ("m.site_size,", "m.site_size.reshape((1, m.nsite)) if m.site_size.ndim == 1 else m.site_size,", 3),
    )
    for old, new, count in replacements:
        if text.count(old) != count:
            raise ValueError(f"Unsupported MuJoCo-Warp sensor layout: expected {count} occurrences of {old!r}")
        text = text.replace(old, new)
    text += f"\n{MARKER} = True\n"
    ast.parse(text)
    return text


_PHYSICS_KERNELS = {
    "derivative": {"_qderiv_tendon_damping": "_myoacross_zero_tendon_damping"},
    "smooth": {
        "_tendon_armature": "_myoacross_zero_tendon_armature",
        "_tendon_bias_coef": "_myoacross_zero_tendon_armature",
        "_tendon_bias_qfrc": "_myoacross_zero_tendon_armature",
    },
    "passive": {"_spring_damper_tendon_passive": "_myoacross_zero_tendon_passive"},
}


def adapt_physics_source(text: str, module: str) -> str:
    """Guard the five verified zero-contribution launches; preserve all other source."""
    if TENDON_MARKER in text:
        raise ValueError("Physics backend source is already adapted")
    targets = _PHYSICS_KERNELS[module]
    tree = ast.parse(text)
    matches = {name: [] for name in targets}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "wp" and node.value.func.attr == "launch"
                and node.value.args and isinstance(node.value.args[0], ast.Name)
                and node.value.args[0].id in targets):
            name = node.value.args[0].id
            inputs = next((kw.value for kw in node.value.keywords if kw.arg == "inputs"), None)
            fields = {n.attr for n in ast.walk(inputs) if isinstance(n, ast.Attribute)
                      and isinstance(n.value, ast.Name) and n.value.id == "m"} if inputs else set()
            if not set(ZERO_FIELDS[targets[name]]).issubset(fields):
                raise ValueError(f"Unsupported MuJoCo-Warp physics inputs for {name}")
            matches[name].append(node)
    for name, nodes in matches.items():
        if len(nodes) != 1:
            raise ValueError(f"Unsupported MuJoCo-Warp physics layout: expected one launch of {name}")
    lines = text.splitlines(keepends=True)
    for name, node in sorted(((name, nodes[0]) for name, nodes in matches.items()),
                             key=lambda item: item[1].lineno, reverse=True):
        start, end = node.lineno - 1, node.end_lineno
        indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
        lines[start:end] = [f"{indent}if not getattr(m, {targets[name]!r}, False):\n"] + [
            "  " + line if line.strip() else line for line in lines[start:end]
        ]
    adapted = "".join(lines) + f"\n{TENDON_MARKER} = True\n"
    ast.parse(adapted)
    return adapted


def _adapted_sources(source: Path) -> dict[str, str]:
    return {
        "sensor": adapt_sensor_source((source / "_src/sensor.py").read_text()),
        **{name: adapt_physics_source((source / f"_src/{name}.py").read_text(), name)
           for name in _PHYSICS_KERNELS},
    }


def prepare_backend(source: Path, output: Path) -> Path:
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source):
        raise ValueError("Adapter output must be outside the installed package")
    destination = output / "mujoco_warp"
    if destination.exists():
        raise FileExistsError(destination)
    adapted = _adapted_sources(source)
    output.mkdir(parents=True, exist_ok=True)
    # Never leave a partial package on PYTHONPATH after validation/copy failure.
    with TemporaryDirectory(prefix=".backend-", dir=output) as temporary:
        staged = Path(temporary) / "mujoco_warp"
        shutil.copytree(source, staged, ignore=shutil.ignore_patterns("__pycache__"))
        for name, text in adapted.items():
            (staged / f"_src/{name}.py").write_text(text)
        staged.rename(destination)
    return destination


def run_with_backend(command: list[str], cache_root: Path) -> int:
    """Run a child process with a validated adapter on its import path."""
    import fcntl
    import hashlib
    import os
    import subprocess

    spec = importlib.util.find_spec("mujoco_warp")
    if spec is None or spec.origin is None:
        raise RuntimeError("mujoco_warp is not installed in this Python environment")
    source = Path(spec.origin).parent
    raw = (source / "_src/sensor.py").read_text()
    if MARKER in raw:
        raise RuntimeError("Start backend preparation from the original installed package")
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    adapted = _adapted_sources(source)
    for name, text in sorted(adapted.items()):
        digest.update(name.encode())
        digest.update(text.encode())
    output = cache_root / digest.hexdigest()[:16]
    sensor = output / "mujoco_warp/_src/sensor.py"
    cache_root.mkdir(parents=True, exist_ok=True)
    # Each torchrun rank can arrive here before its process group exists.
    with (cache_root / f"{output.name}.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not sensor.exists():
            prepare_backend(source, output)
        for name, text in adapted.items():
            cached = output / f"mujoco_warp/_src/{name}.py"
            if not cached.is_file() or cached.read_text() != text:
                raise RuntimeError(f"Cached {name} adapter differs from the installed backend")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(output.resolve())] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    return subprocess.run(command, env=environment).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    spec = importlib.util.find_spec("mujoco_warp")
    if spec is None or spec.origin is None:
        raise RuntimeError("mujoco_warp is not installed in this Python environment")
    print(prepare_backend(Path(spec.origin).parent, args.output))


if __name__ == "__main__":
    main()

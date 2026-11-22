"""Run subject-specific motion retargeting and export."""

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from queue import Empty
import multiprocessing
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import os
from time import perf_counter
import json
import argparse

from myoacross.utils.datalists import get_target_subject_ids, get_motion_specs
from myoacross.utils.paths import (
    SMPLH_MODEL_DIR,
    EMBODIMENTS_DIR,
    RETARGETS_DIR,
    DATASET_ROOT,
)

from myoacross.retarget.config import load_config
from myoacross.retarget.export import (
    annotate_exported_result,
)
from myoacross.retarget.retargeter import retarget_motion
from myoacross.retarget.export import (
    get_mjlab_layout,
    load_model,
    build_mjlab_motion,
    build_report,
    save_result,
)


def run_motion(
    motion_path: str | Path,
    robot_xml_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    smplh_model_dir: str | Path = SMPLH_MODEL_DIR,
    overwrite: bool = False,
    identity: dict | None = None,
) -> Path:
    """
    Retarget one motion and save mjlab motion.npz plus report.json.

    Source motion identity is independent of the target embodiment XML.
    """

    motion_path = Path(motion_path).expanduser().resolve()
    robot_xml_path = Path(robot_xml_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().absolute()

    # Validate paths
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion not found: {motion_path}")
    if not robot_xml_path.is_file():
        raise FileNotFoundError(f"Model XML not found: {robot_xml_path}")
    if os.path.lexists(output_dir):
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise ValueError(f"Output must be a regular directory: {output_dir}")
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}")

    # Load configs
    config = load_config(config_path)
    start = perf_counter()

    print(f"Motion: {motion_path}", flush=True)
    print(f"Model: {robot_xml_path}", flush=True)

    # Retarget one motion
    result = retarget_motion(
        motion_path=motion_path,
        robot_xml_path=robot_xml_path,
        config=config,
        smplh_model_dir=smplh_model_dir,
    )

    result["metadata"]["identity"] = identity

    # Compile once per process; each motion uses its own MjData.
    model = load_model(robot_xml_path)

    # Use the official Entity ordering to build motion.npz
    layout = get_mjlab_layout(robot_xml_path)
    motion = build_mjlab_motion(
        model,
        result,
        joint_names=layout["joint_names"],
        body_names=layout["body_names"],
    )

    # Build report
    report = build_report(
        result,
        motion,
        joint_names=layout["joint_names"],
        body_names=layout["body_names"],
    )

    report["software"] = {
        "mjlab": layout["mjlab_version"],
        "mujoco": layout["mujoco_version"],
        "general_motion_retargeting": version("general-motion-retargeting"),
        "mink": version("mink"),
    }
    report["processing_seconds_excluding_save"] = perf_counter() - start

    # Save results
    saved_dir = save_result(motion, report, output_dir, overwrite=overwrite)

    annotate_exported_result(saved_dir)

    print(f"Saved: {saved_dir / 'motion.npz'}", flush=True)
    print(f"Report: {saved_dir / 'report.json'}", flush=True)
    print(
        f"Frames: {len(motion['joint_pos'])}, FPS: {float(motion['fps'].item()):g}",
        flush=True,
    )

    return saved_dir


@dataclass(frozen=True)
class RetargetJob:
    target_subject_id: str | None
    source_subject_id: str
    motion_id: str
    category: str
    motion_set: str
    embodiment: str
    target_split: str | None
    motion_path: Path
    robot_xml_path: Path
    output_dir: Path

    def identity(self) -> dict:
        return {
            name: getattr(self, name)
            for name in (
                "target_subject_id",
                "source_subject_id",
                "motion_id",
                "category",
                "motion_set",
                "embodiment",
                "target_split",
            )
        }


def collect_jobs(
    embodiment: str,
    *,
    split: str = "train",
    subject_id: str | None = None,
    motion_set: str = "kit_flat_locomotion_72",
    motion_ids: list[str] | None = None,
    output_root: str | Path = RETARGETS_DIR,
    overwrite: bool = False,
) -> list[RetargetJob]:
    """Pair accepted target embodiments with the shared training motions."""
    if embodiment not in ("baseline", "height_scale", "segment_scale"):
        raise ValueError(f"Unknown embodiment: {embodiment!r}")
    if (
        not motion_set
        or Path(motion_set).name != motion_set
        or motion_set in (".", "..")
    ):
        raise ValueError("motion_set must be a single directory name")
    targets = get_target_subject_ids(split=split, embodiment=embodiment)
    if subject_id is not None:
        subject_id = subject_id.strip()
        if subject_id not in targets:
            raise ValueError(f"Target subject not in {split} split: {subject_id!r}")
        targets = (subject_id,)
    motions = get_motion_specs(motion_set=motion_set)
    if motion_ids is not None:
        requested = set(motion_ids)
        unknown = requested - {motion.motion_id for motion in motions}
        if unknown or not requested:
            raise ValueError(f"Invalid motion selection: {sorted(unknown)}")
        motions = tuple(m for m in motions if m.motion_id in requested)
    for motion in motions:
        if not (DATASET_ROOT / motion.source_path).is_file():
            raise FileNotFoundError(DATASET_ROOT / motion.source_path)

    # A baseline is one shared body, not one body per source performer.
    target_models = (None,) if embodiment == "baseline" and targets else targets
    output_root = Path(output_root).expanduser().absolute()
    jobs = []
    skipped = []
    for target in target_models:
        model_dir = EMBODIMENTS_DIR / embodiment
        if target is not None:
            model_dir /= target
        model_path = model_dir / "model.xml"
        report_path = model_dir / "report.json"
        with report_path.open(encoding="utf-8") as file:
            report = json.load(file)
        if (
            not isinstance(report, dict)
            or report.get("strategy") != embodiment
            or "subject_id" not in report
            or report["subject_id"] != target
        ):
            raise ValueError(f"Embodiment identity mismatch: {report_path}")
        if report.get("status") == "failed":
            skipped.append(target)
            print(f"Skipping failed target {target}: {report.get('error')}", flush=True)
            continue
        if report.get("status") != "passed":
            raise ValueError(f"Unknown embodiment status: {report_path}")
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        for motion in motions:
            model_output = output_root / embodiment
            if target is not None:
                model_output /= target
            output_dir = model_output / motion.motion_id
            if os.path.lexists(output_dir):
                if output_dir.is_symlink() or not output_dir.is_dir():
                    raise ValueError(
                        f"Output must be a regular directory: {output_dir}"
                    )
                if not overwrite:
                    raise FileExistsError(f"Output already exists: {output_dir}")
            jobs.append(
                RetargetJob(
                    target_subject_id=target,
                    source_subject_id=motion.source_subject_id,
                    motion_id=motion.motion_id,
                    category=motion.category,
                    motion_set=motion_set,
                    embodiment=embodiment,
                    target_split=split if target is not None else None,
                    motion_path=(DATASET_ROOT / motion.source_path).resolve(),
                    robot_xml_path=model_path.resolve(),
                    output_dir=output_dir,
                )
            )
    print(
        f"Collected {len(jobs)} target-motion jobs; skipped {len(skipped)} failed targets.",
        flush=True,
    )
    return jobs


def group_jobs(jobs: list[RetargetJob]) -> list[list[RetargetJob]]:
    """Keep a target's motions in one process so fitted models can be reused."""
    groups: dict[Path, list[RetargetJob]] = {}
    for job in jobs:
        groups.setdefault(job.output_dir.parent, []).append(job)
    return list(groups.values())


_STOP_EVENT = None
_RESULT_QUEUE = None


def _init_worker(stop_event, result_queue) -> None:
    global _STOP_EVENT, _RESULT_QUEUE
    _STOP_EVENT = stop_event
    _RESULT_QUEUE = result_queue
    import torch

    torch.set_num_threads(1)


def _execute_job(job, config_path, config, overwrite, skip_completed) -> dict:
    """Common sequential/parallel validation, execution and status reporting."""
    record = {**job.identity(), "output_dir": str(job.output_dir)}
    if skip_completed and job.output_dir.exists():
        try:
            existing = json.loads((job.output_dir / "report.json").read_text())
            if not (job.output_dir / "motion.npz").is_file():
                raise ValueError("Missing motion arrays")
            if (
                existing["metadata"]["identity"] != job.identity()
                or existing["metadata"]["config"] != config
            ):
                raise ValueError("Result identity/config differs")
            if "quality" not in existing:
                raise ValueError("Missing quality screening")
        except (OSError, ValueError, KeyError) as error:
            if not overwrite:
                raise ValueError(
                    f"Stale/incomplete result {job.output_dir}; use --overwrite"
                ) from error
        else:
            return {
                **record,
                "status": "skipped_complete",
                "quality": existing["quality"],
            }
    run_motion(
        job.motion_path,
        job.robot_xml_path,
        config_path,
        job.output_dir,
        overwrite=overwrite,
        identity=job.identity(),
    )
    report = json.loads((job.output_dir / "report.json").read_text())
    return {**record, "status": "exported", "quality": report["quality"]}


def _run_group(jobs, config_path, config, overwrite, skip_completed, continue_on_error):
    """Send each completed motion to the parent, including before a later failure."""
    records = []
    for job in jobs:
        if _STOP_EVENT.is_set():
            break
        try:
            result = _execute_job(job, config_path, config, overwrite, skip_completed)
        except Exception as error:
            result = {
                **job.identity(),
                "output_dir": str(job.output_dir),
                "status": "failed",
                "error": repr(error),
            }
            if not continue_on_error:
                _STOP_EVENT.set()
        records.append(result)
        _RESULT_QUEUE.put(result)
        if result["status"] == "failed" and not continue_on_error:
            break
    return records


def execute_jobs(
    jobs: list[RetargetJob],
    config_path: Path,
    *,
    workers: int = 1,
    overwrite: bool = False,
    skip_completed: bool = False,
    continue_on_error: bool = False,
    on_result=None,
) -> list[dict]:
    """
    Execute jobs with identical semantics for one or multiple workers.

    Only the parent invokes on_result, once per completed/skipped/failed motion.
    At most workers groups are submitted; after a failure, busy workers finish
    their current motion and no further motion is started unless requested.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    config = load_config(config_path)
    records = []
    seen = set()

    def publish(result):
        # Results arrive per motion and again with the completed future. The
        # latter ensures a flushed manifest even if queue delivery was delayed.
        if result["output_dir"] in seen:
            return
        seen.add(result["output_dir"])
        records.append(result)
        if on_result is not None:
            on_result(result)
        detail = f"; {result['error']}" if result["status"] == "failed" else ""
        print(
            f"[{len(records)}/{len(jobs)}] {result['status']}: "
            f"{result['output_dir']}{detail}",
            flush=True,
        )

    if workers == 1:
        for job in jobs:
            try:
                result = _execute_job(
                    job, config_path, config, overwrite, skip_completed
                )
            except Exception as error:
                result = {
                    **job.identity(),
                    "output_dir": str(job.output_dir),
                    "status": "failed",
                    "error": repr(error),
                }
            publish(result)
            if result["status"] == "failed" and not continue_on_error:
                break
        return records

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    queue = deque(sorted(group_jobs(jobs), key=len, reverse=True))
    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    result_queue = context.Queue()
    running = {}

    def drain():
        while True:
            try:
                result = result_queue.get_nowait()
            except Empty:
                return
            publish(result)

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(stop_event, result_queue),
        ) as pool:
            try:
                while queue or running:
                    if stop_event.is_set():
                        queue.clear()
                    while queue and len(running) < workers and not stop_event.is_set():
                        group = queue.popleft()
                        future = pool.submit(
                            _run_group,
                            group,
                            config_path,
                            config,
                            overwrite,
                            skip_completed,
                            continue_on_error,
                        )
                        running[future] = group
                    if not running:
                        break
                    done, _ = wait(running, timeout=0.1, return_when=FIRST_COMPLETED)
                    drain()
                    for future in done:
                        group = running.pop(future)
                        try:
                            results = future.result()
                        except BaseException:
                            # Infrastructure failures cannot safely continue.
                            stop_event.set()
                            raise
                        for result in results:
                            publish(result)
                drain()
            except BaseException:
                stop_event.set()
                # Drain while joining workers, otherwise a full result pipe
                # could prevent a worker from exiting after interruption.
                while running:
                    done, _ = wait(running, timeout=0.1, return_when=FIRST_COMPLETED)
                    while True:
                        try:
                            result_queue.get_nowait()
                        except Empty:
                            break
                    for future in done:
                        running.pop(future)
                raise
    finally:
        result_queue.close()
        result_queue.join_thread()
    return records


def main(argv: list[str] | None = None) -> None:
    """Retarget shared motions using accepted target embodiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Retarget YAML configuration.",
    )
    parser.add_argument(
        "--embodiment",
        choices=("baseline", "height_scale", "segment_scale"),
        required=True,
        help="Generated embodiment strategy.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Target embodiment split; all selected motions are training motions.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Process one target subject; otherwise process all targets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing motion.npz and report.json.",
    )
    parser.add_argument("--motion-set", default="kit_flat_locomotion_72")
    parser.add_argument(
        "--motion-id",
        action="append",
        dest="motion_ids",
        help="Select a motion ID; may be repeated.",
    )
    parser.add_argument("--output-root", type=Path, default=RETARGETS_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list jobs without fitting or retargeting.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip complete results with matching identity, configuration and content hashes.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failures and continue other motions.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Write this invocation's task results and quality status.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes; motions within a target stay sequential.",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    # Load configs
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)

    # Collect retarget jobs
    jobs = collect_jobs(
        args.embodiment,
        split=args.split,
        subject_id=args.subject,
        motion_set=args.motion_set,
        motion_ids=args.motion_ids,
        output_root=args.output_root,
        overwrite=args.overwrite or args.skip_completed,
    )

    if args.dry_run:
        for job in jobs:
            print(json.dumps({**job.identity(), "output_dir": str(job.output_dir)}))
        return

    if not jobs:
        print("No executable retargeting jobs.", flush=True)
        return

    records = []

    def record(result):
        records.append(result)
        if args.manifest is not None:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.manifest.with_suffix(".tmp")
            temporary.write_text(json.dumps(records, indent=2) + "\n")
            temporary.replace(args.manifest)

    execute_jobs(
        jobs,
        config_path,
        workers=args.workers,
        overwrite=args.overwrite,
        skip_completed=args.skip_completed,
        continue_on_error=args.continue_on_error,
        on_result=record,
    )
    failures = sum(r["status"] == "failed" for r in records)
    saved = sum(r["status"] == "exported" for r in records)
    print(
        f"Done: {len(records)}/{len(jobs)} tasks, {saved} motions saved, "
        f"{failures} execution failures.",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

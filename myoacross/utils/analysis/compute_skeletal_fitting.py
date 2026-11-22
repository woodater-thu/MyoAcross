"""Calculate skeletal fitting statistics and figures from model reports."""

import json
from pathlib import Path
import numpy as np
from myoacross.utils.datalists import get_target_subject_ids

from myoacross.utils.paths import EMBODIMENTS_DIR, PROJECT_ROOT
import argparse





def _read_model_report(path: Path, sources: list[str]) -> dict:
    raw = path.read_bytes()
    sources.append(str(path))
    report = json.loads(raw)
    assert report["measurement_stage"] == "final_compiled_model", path
    assert report["status"] == "passed", path
    return report


def collect_fitting_rows(root: Path):
    """Validate model reports and collect errors for each subject and method."""
    sources = []

    baseline = _read_model_report(root / "baseline/report.json", sources)
    names = [n for n in baseline["measurements"] if n != "height"]
    assert len(names) == 17
    rows = []
    for split in ("train", "test"):
        for subject in get_target_subject_ids(split=split, embodiment="segment_scale"):
            reports = {
                m: _read_model_report(root / m / subject / "report.json", sources)
                for m in ("height_scale", "segment_scale")
            }
            targets = np.array(
                [reports["segment_scale"]["measurements"][n]["target_m"] for n in names]
            )
            for method, report in [("baseline", baseline), *reports.items()]:
                actual = np.array(
                    [report["measurements"][n]["actual_m"] for n in names]
                )
                if method != "baseline":
                    assert report["subject_id"] == subject
                    other = np.array(
                        [report["measurements"][n]["target_m"] for n in names]
                    )
                    np.testing.assert_allclose(other, targets, rtol=0, atol=1e-12)
                    recorded = np.array(
                        [report["measurements"][n]["error_m"] for n in names]
                    )
                    np.testing.assert_allclose(
                        actual - targets, recorded, rtol=0, atol=1e-12
                    )
                error = actual - targets
                assert np.isfinite(error).all()
                rows.append(
                    dict(
                        subject=subject,
                        split=split,
                        method=method,
                        mae_m=float(np.abs(error).mean()),
                        signed_errors_m=dict(zip(names, error.tolist())),
                    )
                )

    return names, rows, sources


def summarize_fitting(names, rows) -> dict:
    """Reduce subject errors using the established split and method order."""
    summary = {}
    for split in ("all", "train", "test"):
        summary[split] = {}
        for method in ("baseline", "height_scale", "segment_scale"):
            selected = [
                r
                for r in rows
                if r["method"] == method and (split == "all" or r["split"] == split)
            ]
            values = np.array([r["mae_m"] for r in selected])
            summary[split][method] = dict(
                n=len(values),
                mae_m=float(values.mean()),
                subject_sd_m=float(values.std(ddof=1)),
                median_m=float(np.median(values)),
                per_measurement_mae_m={
                    n: float(np.mean([abs(r["signed_errors_m"][n]) for r in selected]))
                    for n in names
                },
            )
        regional = summary[split]["segment_scale"]["mae_m"]
        summary[split]["regional_reduction_pct"] = {
            m: 100 * (1 - regional / summary[split][m]["mae_m"])
            for m in ("baseline", "height_scale")
        }
    assert summary["all"]["baseline"]["n"] == 141
    return summary






def main(argv: list[str] | None = None):
    from myoacross.utils.plotting.plot_skeletal_fitting import plot_fitting_results
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embodiments-root", type=Path, default=EMBODIMENTS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/experiments/skeletal_fitting/evaluations/report",
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = args.embodiments_root.resolve()
    names, rows, sources = collect_fitting_rows(root)
    summary = summarize_fitting(names, rows)
    (output / "results.json").write_text(
        json.dumps(
            dict(
                metric="Mean absolute error over 17 measurements, then equally over humans; meters",
                baseline_note="height_scale denotes global scaling; source reports identify the fitting method",
                summary=summary,
                rows=rows,
                source_reports=sources,
            ),
            indent=2,
        )
    )
    plot_fitting_results(names, rows, summary, output)
    print(output / "results.json")


if __name__ == "__main__":
    main()

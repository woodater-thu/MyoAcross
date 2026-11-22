"""Calculate muscle-level body-mass associations."""

import json, numpy as np, pandas as pd
from scipy.stats import pearsonr, t as student_t
from .comparison_paths import parse_analysis_args


def run(root, gait_root, output):
    W = root
    O = output
    O.mkdir(parents=True, exist_ok=False)
    meta = {
        r["subject_id"]: r["metadata"]
        for r in json.loads((W / "manifest.json").read_text())["subjects"]
    }
    rows = []
    for cond in ["height_scale", "regional_scale"]:
        P = W / "evaluations" / cond / "profiles"
        for rec in json.loads((P / "signal_audit.json").read_text()):
            if not rec["completed"]:
                continue
            sid = rec["subject"]
            motion = rec["motion"]
            events = json.loads((P / sid / f"{motion}_events.json").read_text())
            with np.load(P / sid / "signals" / f"{motion}.npz") as d:
                time = d["time_episode_sec"]
                act = d["activation"]
                pos = d["root_position_m"][:, :2]
                names = list(d["muscle_names"])
                for ci, event in enumerate(events["cycles"]):
                    a = event["start"] * 0.01
                    b = event["end"] * 0.01
                    grid = np.r_[a, time[(time > a) & (time < b)], b]
                    assert b <= time[-1] + 1e-5
                    v = np.linalg.norm(
                        [
                            np.interp(b, time, pos[:, j])
                            - np.interp(a, time, pos[:, j])
                            for j in range(2)
                        ]
                    ) / (b - a)
                    for j, name in enumerate(names):
                        av = np.trapezoid(np.interp(grid, time, act[:, j]), grid) / (
                            b - a
                        )
                        rows.append(
                            dict(
                                condition=cond,
                                subject=sid,
                                motion=motion,
                                cycle=ci,
                                muscle=name,
                                activation=av,
                                speed=v,
                                mass=meta[sid]["body_mass_kg"],
                                height=meta[sid]["height_m"],
                            )
                        )
    df = pd.DataFrame(rows)
    m = (
        df.groupby(["condition", "subject", "motion", "muscle"])[
            ["activation", "speed", "mass", "height"]
        ]
        .mean()
        .reset_index()
    )
    m.to_csv(O / "motions.csv", index=False)
    s = (
        m.groupby(["condition", "subject", "muscle"])[
            ["activation", "speed", "mass", "height"]
        ]
        .mean()
        .reset_index()
    )
    s.to_csv(O / "subjects.csv", index=False)

    def residual(y, z):
        X = np.column_stack([np.ones(len(y)), z])
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    def score(x, y, z):
        return pearsonr(residual(x, z), residual(y, z)).statistic

    rng = np.random.default_rng(42)
    rr = []
    for (cond, muscle), g in s.groupby(["condition", "muscle"]):
        x = g.mass.to_numpy()
        y = g.activation.to_numpy()
        z = g[["height", "speed"]].to_numpy()
        raw = pearsonr(x, y).statistic
        r = score(x, y, z)
        boot = []
        slopes = []
        X = np.column_stack([np.ones(len(x)), x, z])
        slope = np.linalg.lstsq(X, y, rcond=None)[0][1]
        for _ in range(10000):
            ix = rng.integers(len(x), size=len(x))
            boot.append(score(x[ix], y[ix], z[ix]))
            slopes.append(np.linalg.lstsq(X[ix], y[ix], rcond=None)[0][1])
        p = 2 * student_t.sf(abs(r) * np.sqrt((len(x) - 4) / (1 - r * r)), len(x) - 4)
        rr.append(
            dict(
                condition=cond,
                muscle=muscle,
                n=len(x),
                raw_r=raw,
                partial_r=r,
                ci_low=np.quantile(boot, 0.025),
                ci_high=np.quantile(boot, 0.975),
                p=p,
                slope_per_10kg=slope * 10,
                slope_ci_low=np.quantile(slopes, 0.025) * 10,
                slope_ci_high=np.quantile(slopes, 0.975) * 10,
                mean_activation=y.mean(),
            )
        )
    r = pd.DataFrame(rr)
    for c, ix in r.groupby("condition").groups.items():
        vals = r.loc[ix, "p"].to_numpy()
        order = np.argsort(vals)
        q = np.minimum.accumulate(
            (vals[order] * len(vals) / np.arange(1, len(vals) + 1))[::-1]
        )[::-1]
        back = np.empty(len(vals))
        back[order] = np.minimum(1, q)
        r.loc[ix, "fdr_q"] = back
    r.to_csv(O / "muscle_statistics.csv", index=False)
    r["signed_share_of_total_slope"] = r.slope_per_10kg / r.groupby(
        "condition"
    ).slope_per_10kg.transform("sum")
    r.to_csv(O / "muscle_statistics.csv", index=False)
    per = []
    for (c, motion, muscle), g in m.groupby(["condition", "motion", "muscle"]):
        per.append(
            dict(
                condition=c,
                motion=motion,
                muscle=muscle,
                n=len(g),
                partial_r=score(
                    g.mass.to_numpy(),
                    g.activation.to_numpy(),
                    g[["height", "speed"]].to_numpy(),
                ),
            )
        )
    pd.DataFrame(per).to_csv(O / "per_motion_partial_correlations.csv", index=False)
    (O / "protocol.json").write_text(
        json.dumps(
            dict(
                activation="time-averaged individual muscle activation over existing complete gait cycles; equal cycles per motion then equal completed motions per subject",
                controls=["height", "simulated speed"],
                n=120,
                bootstrap=10000,
                multiple_testing="BH FDR across 11 muscles within each model condition",
                slope="OLS activation ~ intercept + mass + height + speed; coefficient per 10 kg; cross-subject association",
                signed_slope_share="muscle slope divided by summed muscle slope; a decomposition of the unweighted recorded activation sum, not force or physiological contribution",
                source="Complete gait cycles from saved simulation signals",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    args = parse_analysis_args("Calculate muscle-level body-mass associations.")
    run(args.root, args.gait_root, args.output)

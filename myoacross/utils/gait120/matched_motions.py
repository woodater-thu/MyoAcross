"""Compare model conditions on matched completed motions."""

import json, numpy as np, pandas as pd
from myoacross.utils.gait120.analysis import aggregate_subject
from myoacross.utils.gait120.signals import pearson
from .comparison_paths import parse_analysis_args


def run(root, gait_root, output):
    W = root
    A = output
    G = gait_root
    conds = ["template", "height_scale", "regional_scale"]
    df = pd.read_csv(A / "per_subject.csv")
    audit = json.loads((A / "comparison_audit.json").read_text())
    counts = json.loads((A / "effective_counts.json").read_text())
    recs = {
        c: json.loads(
            (W / "evaluations" / c / "profiles/signal_audit.json").read_text()
        )
        for c in conds
    }
    for sid in audit["subjects_with_different_valid_motion_sets"]:
        ids = set.intersection(
            *[set(r["valid_motion_ids"]) for r in counts if r["subject"] == sid]
        )
        with np.load(G / "references" / f"{sid}.npz") as h:
            for c in conds:
                rr = [
                    r
                    for r in recs[c]
                    if (c == "template" or r["subject"] == sid) and r["motion"] in ids
                ]
                v = aggregate_subject(
                    W
                    / "evaluations"
                    / c
                    / "profiles"
                    / ("shared" if c == "template" else sid),
                    rr,
                )
                metrics = {
                    "emg_r": np.mean(
                        [pearson(x, y) for x, y in zip(v["activation"], h["emg"])]
                    ),
                    "grf_r": np.mean(
                        [pearson(x, y) for x, y in zip(v["grf_bw"], h["grf_bw"])]
                    ),
                    "grf_rmse": np.sqrt(
                        ((v["grf_bw"] - h["grf_bw"]) ** 2).mean(1)
                    ).mean(),
                }
                for k, x in metrics.items():
                    df.loc[(df.subject == sid) & (df.condition == c), k] = x
                df.loc[(df.subject == sid) & (df.condition == c), "valid"] = len(ids)
    df.to_csv(A / "per_subject_matched_motions.csv", index=False)
    rng = np.random.default_rng(42)
    out = []
    for metric in ["grf_rmse", "grf_r", "emg_r"]:
        p = df.pivot(index="subject", columns="condition", values=metric)
        for old, new in [
            ("height_scale", "regional_scale"),
            ("template", "regional_scale"),
        ]:
            x = p[[old, new]].dropna()
            d = (
                x[old] - x[new] if metric == "grf_rmse" else x[new] - x[old]
            ).to_numpy()
            b = d[rng.integers(len(d), size=(10000, len(d)))].mean(1)
            out.append(
                dict(
                    metric=metric,
                    comparison=new + " vs " + old,
                    n=len(d),
                    improved=int((d > 0).sum()),
                    mean_improvement=d.mean(),
                    lo=np.quantile(b, 0.025),
                    hi=np.quantile(b, 0.975),
                )
            )
    pd.DataFrame(out).to_csv(A / "paired_matched_motions.csv", index=False)
    print(pd.DataFrame(out).to_string(index=False))
    e = pd.read_csv(A / "between_subject_emg.csv")
    print("EMG FDR q<.05:")
    print(e[e.fdr_q < 0.05].to_string(index=False))
    print(
        "Mean individual deviation errors:",
        pd.read_csv(A / "individual_deviation_errors.csv")
        .groupby("condition")
        .residual_grf_rmse.mean()
        .to_dict(),
    )


if __name__ == "__main__":
    args = parse_analysis_args("Compare model conditions on matched completed motions.")
    run(args.root, args.gait_root, args.output)

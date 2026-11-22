"""Summarize paired and between-subject comparisons."""

import numpy as np, pandas as pd, json
from scipy.stats import rankdata
from .comparison_paths import parse_analysis_args


def run(root, gait_root, output):
    A = output
    rng = np.random.default_rng(20261027)

    def adjust(p, method):
        p = np.asarray(p)
        order = np.argsort(p)
        v = p[order]
        n = len(v)
        z = (
            np.maximum.accumulate(v * (n - np.arange(n)))
            if method == "holm"
            else np.minimum.accumulate((v * n / np.arange(1, n + 1))[::-1])[::-1]
        )
        out = np.empty(n)
        out[order] = np.minimum(1, z)
        return out

    p = pd.read_csv(A / "paired_comparisons.csv")
    p["primary_holm_p"] = np.nan
    mask = (p.comparison == "regional_scale vs height_scale") & p.metric.isin(
        ["grf_rmse", "emg_r"]
    )
    p.loc[mask, "primary_holm_p"] = adjust(p.loc[mask, "wilcoxon_p"], "holm")
    p.to_csv(A / "paired_comparisons.csv", index=False)
    f = pd.read_csv(A / "grf_feature_values.csv")
    rows = []

    def rowcorr(x, y):
        x = rankdata(x, axis=1)
        y = rankdata(y, axis=1)
        x -= x.mean(1, keepdims=True)
        y -= y.mean(1, keepdims=True)
        return (x * y).sum(1) / np.sqrt((x * x).sum(1) * (y * y).sum(1))

    for name, g in f.groupby("feature"):
        p = g.pivot(index="subject", columns="condition", values="simulated")
        y = g.groupby("subject").measured.first().reindex(p.index).to_numpy()
        ix = rng.integers(len(y), size=(10000, len(y)))
        vals = {}
        for c in ["height_scale", "regional_scale"]:
            x = p[c].to_numpy()
            r = rowcorr(x[ix], y[ix])
            vals[c] = r
            rows.append(
                dict(
                    feature=name,
                    quantity=c,
                    rho=rowcorr(x[None], y[None])[0],
                    lo=np.quantile(r, 0.025),
                    hi=np.quantile(r, 0.975),
                )
            )
        d = vals["regional_scale"] - vals["height_scale"]
        rows.append(
            dict(
                feature=name,
                quantity="regional_minus_height",
                rho=rows[-1]["rho"] - rows[-2]["rho"],
                lo=np.quantile(d, 0.025),
                hi=np.quantile(d, 0.975),
            )
        )
    pd.DataFrame(rows).to_csv(A / "between_subject_grf_bootstrap.csv", index=False)
    e = pd.read_csv(A / "between_subject_emg.csv")
    e["fdr_q"] = np.nan
    ok = np.isfinite(e.p)
    e.loc[ok, "fdr_q"] = adjust(e.loc[ok, "p"], "bh")
    e.to_csv(A / "between_subject_emg.csv", index=False)
    counts = json.loads((A / "effective_counts.json").read_text())
    by = {}
    for r in counts:
        by.setdefault(r["subject"], {})[r["condition"]] = r
    mismatch = []
    for sid, c in by.items():
        sets = [
            set(c[x]["valid_motion_ids"])
            for x in ["template", "height_scale", "regional_scale"]
        ]
        if not (sets[0] == sets[1] == sets[2]):
            mismatch.append(sid)
    (A / "comparison_audit.json").write_text(
        json.dumps(
            dict(
                subjects=len(by),
                subjects_with_different_valid_motion_sets=mismatch,
                grf_motion_counts=[
                    r for r in counts if r["grf_valid_motions"] != [8, 8]
                ],
            ),
            indent=2,
        )
    )
    print("Valid-motion set mismatches:", mismatch)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    args = parse_analysis_args("Summarize paired and between-subject comparisons.")
    run(args.root, args.gait_root, args.output)

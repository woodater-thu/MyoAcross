"""Compare Template, Global scale and Regional scale."""

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from myoacross.utils.gait120.analysis import aggregate_subject, summarize_metrics
from myoacross.utils.gait120.signals import pearson, EMG_CHANNELS
from .comparison_paths import parse_analysis_args


def run(root, gait_root, output):
    W = root
    G = gait_root
    O = output
    O.mkdir(parents=True, exist_ok=False)
    conds = ["template", "height_scale", "regional_scale"]
    m = json.loads((W / "manifest.json").read_text())
    sids = [r["subject_id"] for r in m["subjects"]]
    rng = np.random.default_rng(42)
    audits = {
        c: json.loads(
            (W / "evaluations" / c / "profiles/signal_audit.json").read_text()
        )
        for c in conds
    }
    for c in conds[1:]:
        dest = O / f"original_protocol_{c}"
        if not dest.exists():
            summarize_metrics(
                W / "evaluations" / c / "profiles", G / "references", dest
            )
    template = aggregate_subject(
        W / "evaluations/template/profiles/shared", audits["template"]
    )
    human = []
    sim = {c: [] for c in conds}
    rows = []
    counts = []
    for sid in sids:
        with np.load(G / "references" / f"{sid}.npz") as d:
            h = {k: d[k] for k in ["emg", "grf_bw"]}
        human.append(h)
        for c in conds:
            rec = (
                audits[c]
                if c == "template"
                else [r for r in audits[c] if r["subject"] == sid]
            )
            a = (
                template
                if c == "template"
                else aggregate_subject(W / "evaluations" / c / "profiles" / sid, rec)
            )
            sim[c].append(a)
            er = np.array([pearson(x, y) for x, y in zip(a["activation"], h["emg"])])
            gr = np.array([pearson(x, y) for x, y in zip(a["grf_bw"], h["grf_bw"])])
            rm = np.sqrt(np.mean((a["grf_bw"] - h["grf_bw"]) ** 2, axis=1))
            rows.append(
                dict(
                    subject=sid,
                    condition=c,
                    emg_r=np.mean(er),
                    grf_r=np.mean(gr),
                    grf_rmse=np.mean(rm),
                    completed=a["completed_motions"],
                    valid=a["valid_motions"],
                )
            )
            counts.append(
                dict(
                    subject=sid,
                    condition=c,
                    valid_motion_ids=a["valid_motion_ids"],
                    grf_valid_motions=a["grf_valid_motions"],
                )
            )
    pd.DataFrame(rows).to_csv(O / "per_subject.csv", index=False)
    (O / "effective_counts.json").write_text(json.dumps(counts, indent=2))
    df = pd.DataFrame(rows)
    summary = []
    for c in conds:
        q = df[df.condition == c]
        control = json.loads((W / "evaluations" / c / "summary.json").read_text())
        row = dict(
            condition=c,
            completion=control["completed"],
            coverage=control["coverage"],
            tracking_mm=1000 * control["body_position_error_m"],
        )
        for metric in ["grf_rmse", "grf_r", "emg_r"]:
            v = q[metric].dropna().to_numpy()
            b = v[rng.integers(len(v), size=(10000, len(v)))].mean(1)
            row.update(
                {
                    metric: float(v.mean()),
                    metric + "_n": len(v),
                    metric + "_lo": float(np.quantile(b, 0.025)),
                    metric + "_hi": float(np.quantile(b, 0.975)),
                }
            )
        summary.append(row)
    pd.DataFrame(summary).to_csv(O / "summary.csv", index=False)
    paired = []
    for metric in ["grf_rmse", "grf_r", "emg_r"]:
        p = df.pivot(index="subject", columns="condition", values=metric)
        for old, new in [
            ("template", "height_scale"),
            ("height_scale", "regional_scale"),
            ("template", "regional_scale"),
        ]:
            x = p[[old, new]].dropna()
            d = (
                x[old] - x[new] if metric == "grf_rmse" else x[new] - x[old]
            ).to_numpy()
            b = d[rng.integers(len(d), size=(10000, len(d)))].mean(1)
            paired.append(
                dict(
                    metric=metric,
                    comparison=new + " vs " + old,
                    n=len(d),
                    improved=int((d > 0).sum()),
                    mean_improvement=d.mean(),
                    ci_low=np.quantile(b, 0.025),
                    ci_high=np.quantile(b, 0.975),
                    wilcoxon_p=wilcoxon(d).pvalue,
                )
            )
    pd.DataFrame(paired).to_csv(O / "paired_comparisons.csv", index=False)
    Hgrf = np.stack([h["grf_bw"] for h in human])
    Hemg = np.stack([h["emg"] for h in human])
    Sgrf = {c: np.stack([a["grf_bw"] for a in sim[c]]) for c in conds}
    Semg = {c: np.stack([a["activation"] for a in sim[c]]) for c in conds}
    mask = np.isfinite(Hgrf).all((1, 2))
    pop = []
    for c in conds:
        hg = Hgrf[mask].mean(0)
        sg = Sgrf[c][mask].mean(0)
        pop.append(
            dict(
                condition=c,
                grf_r=np.mean([pearson(a, b) for a, b in zip(sg, hg)]),
                grf_rmse=np.sqrt(((sg - hg) ** 2).mean(1)).mean(),
                emg_r=np.mean(
                    [pearson(a, b) for a, b in zip(Semg[c].mean(0), Hemg.mean(0))]
                ),
            )
        )
    pd.DataFrame(pop).to_csv(O / "population_mean_waveform_metrics.csv", index=False)

    def features(g):
        a = g[:, :, :50].max(2)
        b = g[:, :, 50:].max(2)
        v = g[:, :, 25:75].min(2)
        ta = g[:, :, :50].argmax(2) / 99 * 100
        tb = (50 + g[:, :, 50:].argmax(2)) / 99 * 100
        return np.round(np.stack([a, b, v, ta, tb], axis=-1).mean(1), 8)

    fnames = [
        "First peak (BW)",
        "Second peak (BW)",
        "Valley (BW)",
        "First peak time (%)",
        "Second peak time (%)",
    ]
    HF = features(Hgrf[mask])
    SF = {c: features(Sgrf[c][mask]) for c in conds}
    between = []
    for j, name in enumerate(fnames):
        for c in conds:
            x = SF[c][:, j]
            y = HF[:, j]
            rho = np.nan if c == "template" else spearmanr(x, y).statistic
            between.append(
                dict(
                    feature=name,
                    condition=c,
                    n=len(y),
                    spearman=rho,
                    rmse=np.sqrt(np.mean((x - y) ** 2)),
                    measured_sd=np.std(y, ddof=1),
                    simulated_sd=np.std(x, ddof=1),
                )
            )
    pd.DataFrame(between).to_csv(O / "between_subject_grf.csv", index=False)
    fr = []
    for c in conds:
        for i, sid in enumerate(np.asarray(sids)[mask]):
            for j, n in enumerate(fnames):
                fr.append(
                    dict(
                        subject=sid,
                        condition=c,
                        feature=n,
                        simulated=SF[c][i, j],
                        measured=HF[i, j],
                    )
                )
    pd.DataFrame(fr).to_csv(O / "grf_feature_values.csv", index=False)

    def shares(a):
        total = a.sum(2)
        return np.stack(
            [
                a[:, :, k * 25 : (k + 1) * 25].sum(2)
                / np.where(total > 1e-12, total, np.nan)
                for k in range(4)
            ],
            axis=2,
        )

    HE = shares(Hemg)
    erows = []
    heat = {}
    for c in conds[1:]:
        E = shares(Semg[c])
        arr = np.full((12, 4), np.nan)
        for j, ch in enumerate(EMG_CHANNELS):
            for k in range(4):
                ok = np.isfinite(E[:, j, k]) & np.isfinite(HE[:, j, k])
                x = E[ok, j, k]
                y = HE[ok, j, k]
                r, p = spearmanr(x, y)
                arr[j, k] = r
                erows.append(
                    dict(
                        condition=c,
                        channel=ch,
                        quarter=k + 1,
                        n=len(x),
                        spearman=r,
                        p=p,
                    )
                )
        heat[c] = arr
    pd.DataFrame(erows).to_csv(O / "between_subject_emg.csv", index=False)
    res = []
    for c in conds:
        h = Hgrf[mask]
        s = Sgrf[c][mask]
        hr = h - h.mean(0)
        sr = s - s.mean(0)
        rm = np.sqrt(((sr - hr) ** 2).mean((1, 2)))
        for sid, value in zip(np.asarray(sids)[mask], rm):
            res.append(dict(subject=sid, condition=c, residual_grf_rmse=value))
    pd.DataFrame(res).to_csv(O / "individual_deviation_errors.csv", index=False)


if __name__ == "__main__":
    args = parse_analysis_args("Compare Template, Global scale and Regional scale.")
    run(args.root, args.gait_root, args.output)

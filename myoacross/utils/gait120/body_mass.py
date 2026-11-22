"""Calculate body-mass associations of mean muscle activation."""

import numpy as np, pandas as pd, json
from scipy.stats import pearsonr, t as student_t
from .comparison_paths import parse_analysis_args


def run(root, gait_root, output):
    O = output
    O.mkdir(parents=True, exist_ok=False)
    s = pd.read_csv(root / "subjects.csv")

    def residual(y, z):
        X = np.column_stack([np.ones(len(y)), z])
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    def partial(x, y, z):
        return float(pearsonr(residual(x, z), residual(y, z)).statistic)

    rng = np.random.default_rng(42)
    rows = []
    resrows = []
    for k, (c, label, color) in enumerate(
        [
            ("height_scale", "Global scale", "#dc9639"),
            ("regional_scale", "Regional scale", "#247baf"),
        ]
    ):
        d = s[s.condition == c]
        x = d.mass.to_numpy()
        y = d.activation_sum11.to_numpy()
        z = d[["height", "speed"]].to_numpy()
        for adj, controls in [("raw", np.empty((len(x), 0))), ("height_speed", z)]:
            xx = residual(x, controls)
            yy = residual(y, controls)
            r, p = pearsonr(xx, yy)
            boot = []
            for _ in range(10000):
                ix = rng.integers(len(x), size=len(x))
                boot.append(partial(x[ix], y[ix], controls[ix]))
            lo, hi = np.quantile(boot, [0.025, 0.975])
            dfree = len(x) - controls.shape[1] - 2
            p = 2 * student_t.sf(abs(r) * np.sqrt(dfree / (1 - r * r)), dfree)
            rows.append(
                dict(
                    condition=c,
                    adjustment=adj,
                    n=len(x),
                    pearson_r=r,
                    p_unadjusted=p,
                    ci_low=lo,
                    ci_high=hi,
                )
            )
            if adj != "raw":
                for sid, a, b in zip(d.subject, xx, yy):
                    resrows.append(
                        dict(
                            condition=c,
                            subject=sid,
                            mass_residual=a,
                            activation_residual=b,
                        )
                    )
        # Regression slope at fixed height and speed.
        X = np.column_stack([np.ones(len(x)), x, z])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        rows.append(
            dict(
                condition=c,
                adjustment="OLS slope per kg at fixed height and speed",
                n=len(x),
                pearson_r=np.nan,
                p_unadjusted=np.nan,
                ci_low=np.nan,
                ci_high=np.nan,
                slope_per_kg=beta[1],
            )
        )
    pd.DataFrame(rows).to_csv(O / "correlations.csv", index=False)
    pd.DataFrame(resrows).to_csv(O / "adjusted_subjects.csv", index=False)
    s[["condition", "subject", "mass", "height", "speed", "activation_sum11"]].to_csv(
        O / "subjects.csv", index=False
    )
    m = pd.read_csv(root / "motions.csv")
    mr = []
    for (c, motion), d in m.groupby(["condition", "motion"]):
        mr.append(
            dict(
                condition=c,
                motion=motion,
                n=len(d),
                raw_r=pearsonr(d.mass, d.activation_sum11).statistic,
                partial_r=partial(
                    d.mass.to_numpy(),
                    d.activation_sum11.to_numpy(),
                    d[["height", "speed"]].to_numpy(),
                ),
            )
        )
    pd.DataFrame(mr).to_csv(O / "per_motion_correlations.csv", index=False)
    (O / "protocol.json").write_text(
        json.dumps(
            dict(
                activation="time-averaged sum of 11 unique recorded right-leg activations; cycles within motion then equal completed motion weights",
                mass="recorded subject mass, matching calibrated individual model mass",
                controls=["recorded height", "simulated walking speed"],
                partial="Pearson correlation of OLS residuals; linear adjustment",
                bootstrap="10000 subject resamples, refitting adjustment in every resample",
                interpretation="cross-subject association; controlled height/speed does not constitute an isolated mass intervention",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    args = parse_analysis_args(
        "Calculate body-mass associations of mean muscle activation."
    )
    run(args.root, args.gait_root, args.output)

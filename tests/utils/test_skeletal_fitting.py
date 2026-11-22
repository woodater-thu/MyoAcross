"""Check fitting figure units and subject pairing."""

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np

from myoacross.utils.plotting import plot_skeletal_fitting as fitting


class FittingFigureTests(TestCase):
    def test_figures_use_millimeters_and_pair_subjects_by_identity(self):
        names = ["thigh_length_r", "thigh_length_l"]
        rows = [
            dict(subject=subject, method=method, mae_m=value)
            for method, samples in (
                ("baseline", [("A", 0.03), ("B", 0.04)]),
                ("height_scale", [("A", 0.01), ("B", 0.02)]),
                ("segment_scale", [("B", 0.002), ("A", 0.001)]),
            )
            for subject, value in samples
        ]
        summary = {"all": {
            method: {"per_measurement_mae_m": dict(zip(names, values))}
            for method, values in (
                ("baseline", [0.03, 0.04]),
                ("height_scale", [0.01, 0.02]),
                ("segment_scale", [0.001, 0.002]),
            )
        }}
        figures = {}
        exports = []

        def capture(figure, path, **kwargs):
            path = Path(path)
            figures[path.stem] = figure
            exports.append(path)

        with patch("matplotlib.figure.Figure.savefig", capture):
            fitting.plot_fitting_results(names, rows, summary, Path("unused"))

        self.assertEqual({p.suffix for p in exports}, {".svg", ".pdf", ".png", ".tiff"})
        distribution = figures["subject_mae"].axes[0]
        np.testing.assert_allclose(distribution.collections[1].get_offsets()[:, 1], [10, 20])
        measurements = figures["measurement_mae"].axes[0]
        np.testing.assert_allclose(measurements.collections[1].get_offsets()[:, 0], [10, 20])
        paired = figures["paired_subject_mae"].axes[0]
        np.testing.assert_allclose(paired.collections[0].get_offsets(), [[10, 1], [20, 2]])

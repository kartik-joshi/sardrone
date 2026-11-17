"""Precision-recall curves for every model version, simulated and real.

    python -m sardrone.train.curves        # -> docs/images/pr_curves.png

A single mAP number hides the shape of the trade-off. The curve shows, for
every confidence cut-off, how many of the detector's boxes are right
(precision) against how many of the people it finds (recall). A curve
hugging the top-right corner is good; one that collapses to the left finds
few people however low the cut-off is set.

The curves are drawn from predictions already stored in FiftyOne by
`make evaluate` (field `pred_<version>`); nothing is re-run. Each curve is
the precision envelope (best precision at that recall or higher), which is
the curve average precision integrates.
"""

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from sardrone.train.metrics import _curve, match_dataset  # noqa: E402

DEFAULT_OUT = env.PROJECT_ROOT / "docs" / "images" / "pr_curves.png"
VERSIONS = ("v0", "v1", "v2", "v3", "v4", "v5", "v6")


def envelope_curve(images) -> tuple[np.ndarray, np.ndarray]:
    """(recall, precision envelope) at IoU 0.5, from most to least confident."""
    matched = match_dataset(images, ious=(0.5,))
    precision, recall = _curve(
        matched.correct[0], np.ones(len(matched.confidence)), float(matched.truths_per_image.sum())
    )
    if len(precision) == 0:
        return np.zeros(1), np.zeros(1)
    return recall, np.maximum.accumulate(precision[::-1])[::-1]


def plot(curves: dict, scores: dict, path: Path) -> None:
    """`curves[(domain, version)] = (recall, precision)`; `scores[(domain, version)] = mAP@0.5`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    titles = {"sim": "Simulated test worlds (sim-test)", "real": "Real drone photos (sard-test)"}
    for axis, domain in zip(axes, ("sim", "real")):
        for version in VERSIONS:
            if (domain, version) not in curves:
                continue
            recall, precision = curves[(domain, version)]
            style = "--" if version == "v0" else "-"
            axis.plot(recall, precision, style, linewidth=1.6,
                      label=f"{version}  (mAP@0.5 {scores[(domain, version)]:.3f})")  # fmt: skip
        axis.set_title(titles[domain])
        axis.set_xlabel("recall: share of people found")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.3)
        axis.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("precision: share of boxes that are people")
    figure.suptitle("Precision-recall at IoU 0.5, one training run per version (v0 = pretrained, untrained)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    import fiftyone as fo

    from sardrone.fo import curate, real
    from sardrone.fo.frames import DATASET_NAME
    from sardrone.train.metrics import average_precision
    from sardrone.train.scorecard import images_from_view

    parser = argparse.ArgumentParser(description="Plot precision-recall curves for every version.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    views = {
        "sim": fo.load_dataset(DATASET_NAME).load_saved_view(curate.VIEW_TEST),
        "real": fo.load_dataset(real.DATASET_NAME).load_saved_view(real.VIEW_TEST),
    }
    curves, scores = {}, {}
    for domain, view in views.items():
        for version in VERSIONS:
            field = f"pred_{version}"
            if not view._dataset.has_field(field):
                continue
            recall, precision = envelope_curve(images_from_view(view, field))
            curves[(domain, version)] = (recall, precision)
            scores[(domain, version)] = average_precision(precision, recall)
            print(f"{domain:4s} {version}: mAP@0.5 {scores[(domain, version)]:.4f}, max recall {recall.max():.3f}")
    plot(curves, scores, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

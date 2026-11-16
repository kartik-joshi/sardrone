"""See and measure how different simulated and real images look.

    make gapmap        # -> docs/images/gap_map.png and docs/results/domain_gap.json

Two views of the same question, both from CLIP embeddings:

* THE MAP. Simulated test frames and real SARD frames are placed on one 2D
  map (UMAP). If the two colours sit in separate islands, a model can tell
  the domains apart at a glance, and what it learned on one may not carry
  over to the other.

* ONE NUMBER. A simple classifier (logistic regression) is trained to say
  "simulated or real?" from the embedding alone, and tested on images it
  has not seen (5-fold cross-validation). 50 % would mean the domains are
  indistinguishable; 100 % means they are trivially separable. This is a
  cheap proxy for the gap that needs no detector at all, so it can be
  checked BEFORE spending hours on training.
"""

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import fiftyone as fo  # noqa: E402
import numpy as np  # noqa: E402

from sardrone.fo import curate, real  # noqa: E402
from sardrone.fo.frames import DATASET_NAME  # noqa: E402

DEFAULT_MAP = env.PROJECT_ROOT / "docs" / "images" / "gap_map.png"
DEFAULT_JSON = env.PROJECT_ROOT / "docs" / "results" / "domain_gap.json"


def domain_separability(a: np.ndarray, b: np.ndarray, seed: int = 0) -> float:
    """Cross-validated accuracy of telling set `a` from set `b`.

    The two sets are cut to the same size first, so that guessing always
    scores exactly 50 %.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    rng = np.random.default_rng(seed)
    count = min(len(a), len(b))
    a = a[rng.choice(len(a), count, replace=False)]
    b = b[rng.choice(len(b), count, replace=False)]
    features = np.vstack([a, b])
    labels = np.array([0] * count + [1] * count)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(LogisticRegression(max_iter=2000), features, labels, cv=folds)
    return float(scores.mean())


def _random_halves(embeddings: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(len(embeddings))
    half = len(order) // 2
    return embeddings[order[:half]], embeddings[order[half:]]


def embed_degraded(view, seed: int = 51) -> np.ndarray:
    """Embeddings of the view's images AFTER the camera-realism treatment."""
    import fiftyone.zoo as foz

    from sardrone.train.realism import degrade_file

    with tempfile.TemporaryDirectory() as folder:
        samples = []
        for filepath in view.values("filepath"):
            target = Path(folder) / Path(filepath).name
            target.write_bytes(degrade_file(filepath, Path(filepath).name, seed)[0])
            samples.append(fo.Sample(filepath=str(target)))
        scratch = fo.Dataset()
        scratch.add_samples(samples, progress=False)
        raw = scratch.compute_embeddings(
            foz.load_zoo_model(curate.EMBEDDING_MODEL), batch_size=32, num_workers=0, progress=False
        )
        scratch.delete()
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def build(sim_count: int = 2000, seed: int = 51, map_path: Path = DEFAULT_MAP, json_path: Path = DEFAULT_JSON) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    frames = fo.load_dataset(DATASET_NAME)
    sard = fo.load_dataset(real.DATASET_NAME)
    curate.ensure_embeddings(sard)
    sim_view = frames.load_saved_view(curate.VIEW_TEST).take(sim_count, seed=seed)
    real_view = sard.load_saved_view(real.VIEW_TEST)
    sim = np.array(sim_view.values(curate.EMBEDDING_FIELD))
    real_embeddings = np.array(real_view.values(curate.EMBEDDING_FIELD))
    degraded = embed_degraded(sim_view.take(600, seed=seed), seed)

    summary = {
        "embedding_model": curate.EMBEDDING_MODEL,
        "sim_images": len(sim),
        "real_images": len(real_embeddings),
        "separability_sim_vs_real": round(domain_separability(sim, real_embeddings), 4),
        "separability_degraded_sim_vs_real": round(domain_separability(degraded, real_embeddings), 4),
        # Sanity checks: two RANDOM halves of one domain should not be
        # separable. (Splitting video frames into odd and even ones instead
        # scores BELOW 50 %: each frame's nearest neighbour is then in the
        # other half, which misleads the classifier.)
        "separability_sim_vs_sim": round(domain_separability(*_random_halves(sim, seed)), 4),
        "separability_real_vs_real": round(domain_separability(*_random_halves(real_embeddings, seed)), 4),
    }

    points = umap.UMAP(n_components=2, random_state=seed).fit_transform(np.vstack([sim, real_embeddings]))
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(points[: len(sim), 0], points[: len(sim), 1], s=4, alpha=0.5, label=f"simulated sim-test ({len(sim)})")
    axis.scatter(points[len(sim) :, 0], points[len(sim) :, 1], s=4, alpha=0.5, label=f"real sard-test ({len(real_embeddings)})")
    axis.set_title(
        "CLIP embeddings: simulated vs real\n"
        f"a linear classifier tells them apart {summary['separability_sim_vs_real']:.1%} of the time"
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(markerscale=4)
    figure.tight_layout()
    map_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(map_path, dpi=110)
    plt.close(figure)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Map and measure the sim-vs-real image gap.")
    parser.add_argument("--sim-count", type=int, default=2000)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.sim_count), indent=2))
    print(f"map: {DEFAULT_MAP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

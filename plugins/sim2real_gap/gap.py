"""Core logic of the sim2real-gap plugin. Plain functions, usable without the App.

Two questions, both cheap to answer before (or instead of) training:

1. How different do the simulated and real images look?
   Embed both sets with CLIP, then train a linear classifier to tell them
   apart (5-fold cross-validation on balanced sets). 50 % means the domains
   look alike; 100 % means they are trivially separable, and a detector
   trained on one may not carry over to the other. Two sanity checks
   (random halves of each domain) should come out near 50 %.

2. What does a detector mistake for its target on real images?
   Match predictions to ground truth (FiftyOne's own evaluation) and list
   the false alarms, most confident first.
"""

import numpy as np

DEFAULT_MODEL = "clip-vit-base32-torch"
DOMAIN_FIELD = "sim2real_domain"


def separability(a: np.ndarray, b: np.ndarray, seed: int = 0, folds: int = 5) -> float:
    """Cross-validated accuracy of a linear classifier telling `a` from `b`.
    Both sets are cut to the same size first, so guessing scores exactly 50 %."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    count = min(len(a), len(b))
    if count < folds:
        raise ValueError(f"need at least {folds} images in each set, got {len(a)} and {len(b)}")
    rng = np.random.default_rng(seed)
    a = a[rng.choice(len(a), count, replace=False)]
    b = b[rng.choice(len(b), count, replace=False)]
    features = np.vstack([a, b])
    labels = np.array([0] * count + [1] * count)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return float(cross_val_score(LogisticRegression(max_iter=2000), features, labels, cv=splitter).mean())


def random_halves(embeddings: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two random halves of one set. (Splitting video frames odd/even instead
    scores below 50 %: each frame's nearest neighbour lands in the other half.)"""
    order = np.random.default_rng(seed).permutation(len(embeddings))
    half = len(order) // 2
    return embeddings[order[:half]], embeddings[order[half:]]


def normalise(embeddings) -> np.ndarray:
    x = np.asarray(embeddings, dtype=float)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def ensure_embeddings(view, field: str, model_name: str = DEFAULT_MODEL) -> None:
    """Compute CLIP embeddings into `field` for samples that lack them."""
    import fiftyone as fo
    import fiftyone.zoo as foz

    missing = view.match(~fo.ViewField(field).exists()) if view.has_field(field) else view
    if len(missing):
        missing.compute_embeddings(foz.load_zoo_model(model_name), embeddings_field=field, batch_size=32,
                                   num_workers=0)  # no worker processes: safe in scripts and the App


def measure(dataset, sim_view, real_view, embeddings_field: str = "clip_embedding",
            model_name: str = DEFAULT_MODEL, brain_key: str | None = "sim2real_gap", seed: int = 0) -> dict:  # fmt: skip
    """Separability of simulated vs real images, with sanity checks.

    Writes `sim2real_domain` ("sim" or "real") on the samples, stores the
    summary in `dataset.info["sim2real_gap"]`, and (with `brain_key`) a 2D
    UMAP of both sets for the App's Embeddings panel, coloured by domain.
    """
    sim_ids, real_ids = set(sim_view.values("id")), set(real_view.values("id"))
    if sim_ids & real_ids:
        raise ValueError(f"{len(sim_ids & real_ids)} samples are in both views; the views must not overlap")
    for view in (sim_view, real_view):
        ensure_embeddings(view, embeddings_field, model_name)
    sim = normalise(sim_view.values(embeddings_field))
    real = normalise(real_view.values(embeddings_field))

    sim_view.set_values(DOMAIN_FIELD, ["sim"] * len(sim_view))  # creates the field if needed
    real_view.set_values(DOMAIN_FIELD, ["real"] * len(real_view))
    summary = {
        "embeddings_field": embeddings_field,
        "sim_images": len(sim),
        "real_images": len(real),
        "separability": round(separability(sim, real, seed), 4),
        "sanity_sim_vs_sim": round(separability(*random_halves(sim, seed), seed), 4),
        "sanity_real_vs_real": round(separability(*random_halves(real, seed), seed), 4),
    }
    summary["verdict"] = verdict(summary["separability"])
    if brain_key:
        import fiftyone.brain as fob

        both = dataset.select(list(sim_ids | real_ids))
        if brain_key in dataset.list_brain_runs():
            dataset.delete_brain_run(brain_key)
        fob.compute_visualization(both, embeddings=embeddings_field, brain_key=brain_key, method="umap",
                                  seed=seed)  # fmt: skip
        summary["brain_key"] = brain_key
    dataset.info["sim2real_gap"] = summary
    dataset.save()
    return summary


def verdict(score: float) -> str:
    # Wording is about the two image SETS, not about simulation: two real
    # datasets from different cameras or terrain can be just as separable.
    if score >= 0.95:
        return "trivially separable: the two sets look very different; a model trained on one may not transfer to the other"
    if score >= 0.75:
        return "clearly separable: the sets differ; check performance on the second set directly"
    if score >= 0.60:
        return "somewhat separable: modest differences"
    return "hard to tell apart: the sets look alike"


def false_alarms(view, pred_field: str, gt_field: str = "ground_truth", iou: float = 0.5,
                 eval_key: str = "s2r_eval", min_confidence: float = 0.0):  # fmt: skip
    """A patches view of the false alarms (predictions matching no ground
    truth at `iou`), most confident first. Uses FiftyOne's own matching."""
    from fiftyone import ViewField as F

    view.evaluate_detections(pred_field, gt_field=gt_field, eval_key=eval_key, iou=iou)
    wrong = view.filter_labels(pred_field, (F(eval_key) == "fp") & (F("confidence") >= min_confidence))
    return wrong.to_patches(pred_field).sort_by("confidence", reverse=True)

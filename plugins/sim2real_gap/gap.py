"""Core logic of the sim2real-gap plugin. Plain functions, usable without the App.

The question: how different are two image sets (say, simulated and real), and
WHERE is the difference? Everything works on embeddings (CLIP by default).

Set level
    separability   a classifier trained to tell the two sets apart, scored
                   with cross-validated balanced accuracy (a "classifier
                   two-sample test", Lopez-Paz and Oquab, ICLR 2017). 50 %:
                   indistinguishable; 100 %: trivially different. Reported
                   with its spread across folds, and with two sanity checks
                   (random halves of each set) that should come out near 50 %.
    frechet        the Frechet distance between Gaussians fitted to the two
                   sets of embeddings (as in FID, on CLIP features). 0 for
                   identical distributions; grows with the gap. Depends on
                   set size: compare runs of similar size only.

Sample (or object) level, written as fields you can sort and filter by
    <prefix>_realism   out-of-fold probability that the item is from the REAL
                       set. On simulated items: high = looks real (your best
                       synthetic data). On real items: low = looks synthetic.
    <prefix>_nearest   cosine distance to the nearest item of the OTHER set.
                       On real items: high = nothing like it in the simulated
                       set, i.e. what the simulator is missing.

Saved views (same dataset only): <prefix>-realistic-sim and
<prefix>-uncovered-real. A group breakdown (e.g. by world or weather) says
which simulated conditions are closest to real.
"""

import numpy as np

DEFAULT_MODEL = "clip-vit-base32-torch"
MODELS = ("clip-vit-base32-torch", "dinov2-vits14-torch")
DEFAULT_PREFIX = "s2r"
FOLDS = 5


# ---- statistics (no FiftyOne needed) ---------------------------------------------


def normalise(embeddings) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float32)  # half the memory of float64; plenty for cosine maths
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("expected a non-empty 2-D array of embeddings")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def _classifier():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=2000, class_weight="balanced")


def separability(a: np.ndarray, b: np.ndarray, seed: int = 0, folds: int = FOLDS) -> tuple[float, float]:
    """(mean, std) over folds of the balanced accuracy of telling `a` from `b`.
    Balanced accuracy weighs both sets equally, so guessing scores 50 % even
    when the sets differ in size."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    a, b = np.asarray(a, float), np.asarray(b, float)
    if min(len(a), len(b)) < folds:
        raise ValueError(f"need at least {folds} items in each set, got {len(a)} and {len(b)}")
    x = np.vstack([a, b])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(_classifier(), x, y, cv=splitter, scoring="balanced_accuracy")
    return float(scores.mean()), float(scores.std())


def realism_scores(a: np.ndarray, b: np.ndarray, seed: int = 0, folds: int = FOLDS) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold P(item is from `b`) for every item of `a` and of `b`.
    Out of fold: no item is scored by a classifier that saw it."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    x = np.vstack([a, b])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    p = cross_val_predict(_classifier(), x, y, cv=splitter, method="predict_proba")[:, 1]
    return p[: len(a)], p[len(a):]


def permutation_p_value(a: np.ndarray, b: np.ndarray, seed: int = 0, permutations: int = 100,
                        cap: int = 500) -> float:  # fmt: skip
    """p-value of the separability against shuffled labels (at most `cap` items
    per set, for speed). The smallest possible value is 1 / (permutations + 1):
    0.0099 with 100. A large p means the score could be chance."""
    from sklearn.model_selection import StratifiedKFold, permutation_test_score

    rng = np.random.default_rng(seed)
    a = a[rng.choice(len(a), min(cap, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(cap, len(b)), replace=False)]
    x = np.vstack([a, b])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    _, _, p = permutation_test_score(_classifier(), x, y, cv=splitter, n_permutations=permutations,
                                     scoring="balanced_accuracy", random_state=seed)  # fmt: skip
    return float(p)


def equal_size(a: np.ndarray, b: np.ndarray, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random subsets of exactly `n` items from each set. The Frechet distance
    is biased by set size, so it is only comparable at equal, fixed sizes."""
    rng = np.random.default_rng(seed)
    return a[rng.choice(len(a), n, replace=False)], b[rng.choice(len(b), n, replace=False)]


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to two embedding sets."""
    from scipy import linalg

    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root = linalg.sqrtm(cov_a @ cov_b)
    if not np.isfinite(root).all():  # nearly singular: regularise and retry
        eps = np.eye(len(mu_a)) * 1e-6
        root = linalg.sqrtm((cov_a + eps) @ (cov_b + eps))
    root = np.real(root)
    return float(max(0.0, np.sum((mu_a - mu_b) ** 2) + np.trace(cov_a + cov_b - 2 * root)))


def nearest_distance(query: np.ndarray, reference: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Cosine distance from each (normalised) query row to its nearest reference row."""
    out = np.empty(len(query))
    for start in range(0, len(query), chunk):
        sims = query[start:start + chunk] @ reference.T
        out[start:start + chunk] = 1.0 - sims.max(axis=1)
    return out


def random_halves(x: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two random halves of one set. (Odd/even frames of a video would score
    below 50 %: each frame's twin lands in the other half.)"""
    order = np.random.default_rng(seed).permutation(len(x))
    half = len(order) // 2
    return x[order[:half]], x[order[half:]]


def subsample(n: int, cap: int | None, seed: int = 0) -> np.ndarray:
    idx = np.arange(n)
    if cap and n > cap:
        idx = np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))
    return idx


def verdict(score: float) -> str:
    # About the two SETS, not about simulation: two real datasets from
    # different cameras or terrain can be just as separable.
    if score >= 0.95:
        return "trivially separable: the two sets look very different; a model trained on one may not transfer to the other"
    if score >= 0.75:
        return "clearly separable: the sets differ; check performance on the second set directly"
    if score >= 0.60:
        return "somewhat separable: modest differences"
    return "hard to tell apart: the sets look alike"


# ---- embeddings in FiftyOne -------------------------------------------------------


def ensure_embeddings(view, field: str, model_name: str = DEFAULT_MODEL, patches_field: str | None = None) -> None:
    """Compute embeddings where missing: per sample in `field`, or per object
    in the attribute `field` of each box in `patches_field`."""
    import fiftyone as fo
    import fiftyone.zoo as foz

    if patches_field:
        path = f"{patches_field}.detections.{field}"
        missing = [e is None for boxes in view.values(path) for e in (boxes or [])]
        if any(missing):
            view.compute_patch_embeddings(foz.load_zoo_model(model_name), patches_field, embeddings_field=field,
                                          batch_size=32, num_workers=0)  # fmt: skip
        return
    todo = view.match(~fo.ViewField(field).exists()) if view.has_field(field) else view
    if len(todo):
        # num_workers=0: no worker processes, which is safe in scripts and the App.
        todo.compute_embeddings(foz.load_zoo_model(model_name), embeddings_field=field, batch_size=32,
                                num_workers=0)  # fmt: skip


def _items(view, field: str, patches_field: str | None):
    """(ids, embeddings) of the samples or objects in a view."""
    if patches_field:
        ids, vecs = [], []
        for boxes_ids, boxes_vecs in zip(view.values(f"{patches_field}.detections.id"),
                                         view.values(f"{patches_field}.detections.{field}")):  # fmt: skip
            for i, v in zip(boxes_ids or [], boxes_vecs or []):
                if v is not None:
                    ids.append(i)
                    vecs.append(v)
    else:
        pairs = [(i, v) for i, v in zip(view.values("id"), view.values(field)) if v is not None]
        ids, vecs = [i for i, _ in pairs], [v for _, v in pairs]
    sizes = {len(v) for v in vecs}
    if len(sizes) > 1:
        raise ValueError(f"field {field!r} holds embeddings of different sizes {sorted(sizes)}: "
                         "it mixes models; use one embeddings field per model")  # fmt: skip
    return ids, (normalise(vecs) if vecs else np.zeros((0, 0)))


def clear_previous(dataset, prefix: str, patches_field: str | None) -> None:
    """Remove the per-item results of an earlier run with this prefix, so that
    fields, filters and saved views only ever show the latest run."""
    for name in ("domain", "realism", "nearest"):
        path = f"{patches_field}.detections.{prefix}_{name}" if patches_field else f"{prefix}_{name}"
        if dataset.has_field(path):
            dataset.set_field(path, None).save()


def _write(view, ids: list, values, path: str, patches_field: str | None) -> None:
    """Write one value per sample (or per object) id."""
    if not len(ids):
        return
    mapping = {i: (float(v) if isinstance(v, (float, np.floating)) else v) for i, v in zip(ids, values)}
    if patches_field:
        view.set_label_values(f"{patches_field}.detections.{path}", mapping)
    else:
        view.set_values(path, mapping, key_field="id")


# ---- the measurement --------------------------------------------------------------


MIN_GROUP = 50  # groups smaller than this are listed but not ranked
REPEATS = 3  # subsamples averaged for each group's scores


def frechet_with_baseline(a: np.ndarray, b: np.ndarray, n: int, seed: int = 0) -> dict:
    """The Frechet distance between `a` and `b` at `n` items each, next to the
    same distance between two disjoint halves of `a` and of `b` at the same n.
    At typical sizes most of a raw Frechet distance is small-sample bias; the
    baselines show what "no gap" scores, and `excess` is the gap above it.
    Needs at least 2n items in each set."""
    rng = np.random.default_rng(seed)
    pa, pb = rng.permutation(len(a)), rng.permutation(len(b))
    a1, a2, b1, b2 = a[pa[:n]], a[pa[n:2 * n]], b[pb[:n]], b[pb[n:2 * n]]
    gap_value = frechet_distance(a1, b1)
    base_a, base_b = frechet_distance(a1, a2), frechet_distance(b1, b2)
    return {"frechet": round(gap_value, 4), "baseline_sim": round(base_a, 4), "baseline_real": round(base_b, 4),
            "excess": round(gap_value - (base_a + base_b) / 2, 4), "n": n}  # fmt: skip


def measure(sim_view, real_view, *, embeddings_field: str = "clip_embedding", model_name: str = DEFAULT_MODEL,
            patches_field: str | None = None, group_field: str | None = None, prefix: str = DEFAULT_PREFIX,
            brain_key: str | None = "sim2real_gap", max_items: int | None = 5000, seed: int = 0,
            log=lambda message: None) -> dict:  # fmt: skip
    """Measure the gap between two views (from one dataset or two).

    Everything is computed first; only then are the previous run's results
    cleared and the new ones written, so a failure part-way leaves the
    previous results intact. Writes per-item fields (see the module
    docstring) on both sets, saves two views when both sets are in one
    dataset, stores the summary in `dataset.info[prefix]`, and finally (if
    asked) builds the embedding map, whose failure is recorded, not raised.
    """
    same = sim_view._dataset.name == real_view._dataset.name
    if same:
        overlap = set(sim_view.values("id")) & set(real_view.values("id"))
        if overlap:
            raise ValueError(f"{len(overlap)} samples are in both views; the views must not overlap")
    for view in (sim_view, real_view):
        log(f"embeddings for {view._dataset.name} ...")
        ensure_embeddings(view, embeddings_field, model_name, patches_field)
    sim_ids, sim = _items(sim_view, embeddings_field, patches_field)
    real_ids, real = _items(real_view, embeddings_field, patches_field)
    if min(len(sim_ids), len(real_ids)) < 2 * FOLDS:
        raise ValueError(f"need at least {2 * FOLDS} items in each set, got {len(sim_ids)} and {len(real_ids)}")
    if sim.shape[1] != real.shape[1]:
        raise ValueError(f"embedding sizes differ ({sim.shape[1]} vs {real.shape[1]}): use one model for both")

    # ---- compute everything ----
    si, ri = subsample(len(sim), max_items, seed), subsample(len(real), max_items, seed + 1)
    log("separability, sanity checks, permutation test ...")
    sep, sep_std = separability(sim[si], real[ri], seed)
    n_fd = min(len(si), len(ri)) // 2
    fd = frechet_with_baseline(sim[si], real[ri], n_fd, seed)
    summary = {
        "level": "objects" if patches_field else "images",
        "patches_field": patches_field,
        "embeddings_field": embeddings_field,
        "model": model_name,
        "sim_items": len(sim_ids),
        "real_items": len(real_ids),
        "separability": round(sep, 4),
        "separability_std": round(sep_std, 4),
        "sanity_sim_vs_sim": round(separability(*random_halves(sim[si], seed), seed)[0], 4),
        "sanity_real_vs_real": round(separability(*random_halves(real[ri], seed), seed)[0], 4),
        "p_value": round(permutation_p_value(sim[si], real[ri], seed), 4),
        "frechet": fd["frechet"],
        "frechet_baseline_sim": fd["baseline_sim"],
        "frechet_baseline_real": fd["baseline_real"],
        "frechet_excess": fd["excess"],
        "frechet_n": n_fd,
        "small_sets": min(len(si), len(ri)) < 50,
        "verdict": verdict(sep),
        "fields": {"domain": f"{prefix}_domain", "realism": f"{prefix}_realism", "nearest": f"{prefix}_nearest"},
    }

    # Per-item scores. Realism is out of fold on the subsample; items beyond it
    # are scored by a classifier fitted on the whole subsample.
    log("per-item realism and nearest-neighbour distances ...")
    p_sim, p_real = np.empty(len(sim)), np.empty(len(real))
    p_sim[si], p_real[ri] = realism_scores(sim[si], real[ri], seed)
    rest_s, rest_r = np.setdiff1d(np.arange(len(sim)), si), np.setdiff1d(np.arange(len(real)), ri)
    if len(rest_s) or len(rest_r):
        clf = _classifier().fit(np.vstack([sim[si], real[ri]]), np.r_[np.zeros(len(si)), np.ones(len(ri))])
        if len(rest_s):
            p_sim[rest_s] = clf.predict_proba(sim[rest_s])[:, 1]
        if len(rest_r):
            p_real[rest_r] = clf.predict_proba(real[rest_r])[:, 1]
    near_sim, near_real = nearest_distance(sim, real), nearest_distance(real, sim)
    summary["realism_sim_mean"] = round(float(p_sim.mean()), 4)
    summary["uncovered_real_p90"] = round(float(np.percentile(near_real, 90)), 4)
    if group_field:
        log(f"breakdown by {group_field} ...")
        summary["groups"] = groups(sim_view, real[ri], group_field, embeddings_field, patches_field, seed,
                                   realism_by_id=dict(zip(sim_ids, p_sim)))  # fmt: skip

    # ---- only now: replace the previous run's results ----
    datasets = {sim_view._dataset.name: sim_view._dataset, real_view._dataset.name: real_view._dataset}
    for dataset in datasets.values():
        clear_previous(dataset, prefix, patches_field)
    for view, ids, domain, p, near in ((sim_view, sim_ids, "sim", p_sim, near_sim),
                                       (real_view, real_ids, "real", p_real, near_real)):  # fmt: skip
        _write(view, ids, [domain] * len(ids), f"{prefix}_domain", patches_field)
        _write(view, ids, np.round(p, 4), f"{prefix}_realism", patches_field)
        _write(view, ids, np.round(near, 4), f"{prefix}_nearest", patches_field)
    if same:
        summary["views"] = save_views(sim_view, prefix, patches_field, sim_ids, real_ids)
        if not patches_field:  # label attributes are coloured differently in the App; leave those alone
            set_domain_colours(sim_view._dataset, f"{prefix}_domain")
    for dataset in datasets.values():
        dataset.info[prefix] = summary
        dataset.save()

    if same and brain_key:
        log("embedding map ...")
        try:
            summary["brain_key"] = embedding_map(sim_view, real_view, sim_ids + real_ids, np.vstack([sim, real]),
                                                 patches_field, brain_key, seed)  # fmt: skip
        except Exception as error:  # noqa: BLE001 - the measurement stands without the map
            summary["map_error"] = f"{type(error).__name__}: {error}"
        for dataset in datasets.values():
            dataset.info[prefix] = summary
            dataset.save()
    return summary


def groups(sim_view, real: np.ndarray, group_field: str, embeddings_field: str, patches_field: str | None,
           seed: int = 0, min_group: int = MIN_GROUP, repeats: int = REPEATS,
           realism_by_id: dict | None = None) -> list[dict]:  # fmt: skip
    """Each value of `group_field` in the simulated set against the real set.

    Groups with at least `min_group` items are ranked, most real-like first:
    all at ONE common size n (the smallest ranked group, capped by half the
    real set), each score the mean over `repeats` random subsamples, and the
    Frechet distance given as its excess over a real-vs-real baseline at the
    same n. Smaller groups are listed with their mean realism but not ranked:
    at a few dozen items the scores are mostly noise."""
    from fiftyone import ViewField as F

    members = [(v, sim_view.match(F(group_field) == v)) for v in sorted(sim_view.distinct(group_field), key=str)]
    missing = sim_view.match(~F(group_field).exists())
    if len(missing):
        members.append(("(none)", missing))
    data = [(value, *_items(view, embeddings_field, patches_field)) for value, view in members]
    ranked = [len(x) for _, _, x in data if len(x) >= min_group]
    n = min(ranked + [len(real) // 2]) if ranked else 0
    rows = []
    for k, (value, ids, x) in enumerate(data):
        row = {"value": value, "items": len(x), "n": None, "ranked": False,
               "separability": None, "frechet_excess": None, "realism": None}  # fmt: skip
        if realism_by_id:
            scores = [realism_by_id[i] for i in ids if i in realism_by_id]
            row["realism"] = round(float(np.mean(scores)), 4) if scores else None
        if len(x) >= min_group and n >= min_group:
            seps, excess = [], []
            for r in range(repeats):
                rng = np.random.default_rng(seed + 100 * k + r)
                xs = x[rng.choice(len(x), n, replace=False)]
                perm = rng.permutation(len(real))
                r1, r2 = real[perm[:n]], real[perm[n:2 * n]]
                seps.append(separability(xs, r1, seed + r)[0])
                excess.append(frechet_distance(xs, r1) - frechet_distance(r1, r2))
            row.update(n=n, ranked=True, separability=round(float(np.mean(seps)), 4),
                       frechet_excess=round(float(np.mean(excess)), 4))  # fmt: skip
        rows.append(row)
    big = float("inf")
    return sorted(rows, key=lambda r: (not r["ranked"], r["separability"] or 0, r["frechet_excess"] or big))


def save_views(sim_view, prefix: str, patches_field: str | None, sim_ids: list, real_ids: list) -> dict:
    """The two views you will want, saved on the dataset. Built from THIS run's
    items only (earlier runs' results are cleared first, too)."""
    from fiftyone import ViewField as F

    dataset = sim_view._dataset
    names = {"realistic_sim": f"{prefix}-realistic-sim", "uncovered_real": f"{prefix}-uncovered-real"}
    if patches_field:
        sim_p = dataset.select_labels(ids=sim_ids, fields=patches_field).to_patches(patches_field)
        real_p = dataset.select_labels(ids=real_ids, fields=patches_field).to_patches(patches_field)
        realistic = sim_p.sort_by(F(f"{patches_field}.{prefix}_realism"), reverse=True)
        uncovered = real_p.sort_by(F(f"{patches_field}.{prefix}_nearest"), reverse=True)
    else:
        realistic = dataset.select(sim_ids).sort_by(f"{prefix}_realism", reverse=True)
        uncovered = dataset.select(real_ids).sort_by(f"{prefix}_nearest", reverse=True)
    dataset.save_view(names["realistic_sim"], realistic, overwrite=True)
    dataset.save_view(names["uncovered_real"], uncovered, overwrite=True)
    return names


# Blue for simulated, orange for real: distinct for colour-blind viewers and
# readable on the App's dark background (checked with a CVD validator).
DOMAIN_COLOURS = {"sim": "#3987e5", "real": "#e8833a"}


def set_domain_colours(dataset, path: str) -> bool:
    """Give the domain field readable colours in the App, unless the user has
    already chosen colours for that field. Returns True if it set them."""
    import fiftyone as fo

    scheme = dataset.app_config.color_scheme or fo.ColorScheme()
    fields = list(scheme.fields or [])
    if any(f.get("path") == path for f in fields):
        return False
    fields.append({"path": path, "valueColors": [{"value": v, "color": c} for v, c in DOMAIN_COLOURS.items()]})
    scheme.fields = fields
    dataset.app_config.color_scheme = scheme
    dataset.save()
    return True


def embedding_map(sim_view, real_view, ids: list, embeddings: np.ndarray, patches_field: str | None,
                  brain_key: str, seed: int = 0) -> str:  # fmt: skip
    """A 2-D UMAP of both sets for the App's Embeddings panel, from the
    embeddings already in memory (`ids[i]` is the sample, or object, of row i),
    so the Brain never has to look them up or recompute them."""
    import fiftyone.brain as fob

    dataset = sim_view._dataset
    if brain_key in dataset.list_brain_runs():
        dataset.delete_brain_run(brain_key)
    vectors = dict(zip(ids, embeddings))
    if patches_field:  # the Brain wants, per sample, an array of its objects' embeddings
        both = dataset.select(sim_view.values("id") + real_view.values("id"), ordered=True)
        both = both.select_labels(ids=ids, fields=patches_field)  # only objects that have an embedding
        per_sample = [np.array([vectors[i] for i in boxes]) for boxes in both.values(f"{patches_field}.detections.id")]
        fob.compute_visualization(both, embeddings=per_sample, patches_field=patches_field, brain_key=brain_key,
                                  method="umap", seed=seed, verbose=False)  # fmt: skip
    else:
        both = dataset.select(ids, ordered=True)
        fob.compute_visualization(both, embeddings=np.array([vectors[i] for i in both.values("id")]),
                                  brain_key=brain_key, method="umap", seed=seed, verbose=False)  # fmt: skip
    return brain_key


# ---- detector review --------------------------------------------------------------


def review(view, pred_field: str, gt_field: str = "ground_truth", iou: float = 0.5, eval_key: str | None = None,
           min_confidence: float = 0.0, kind: str = "false_alarms"):  # fmt: skip
    """A patches view, worst first, using FiftyOne's own matching:
    kind="false_alarms": predictions that match no ground truth, most confident first;
    kind="misses": ground-truth objects that no prediction found.

    The evaluation is stored under `eval_key` (default: s2r_<pred field>), one
    per predictions field, and replaced when that field is reviewed again."""
    import re

    from fiftyone import ViewField as F

    eval_key = eval_key or "s2r_" + re.sub(r"[^A-Za-z0-9_]", "_", pred_field)
    dataset = view._dataset
    if eval_key in dataset.list_evaluations():
        dataset.delete_evaluation(eval_key)
    view.evaluate_detections(pred_field, gt_field=gt_field, eval_key=eval_key, iou=iou)
    if kind == "misses":
        return view.filter_labels(gt_field, F(eval_key) == "fn").to_patches(gt_field)
    wrong = view.filter_labels(pred_field, (F(eval_key) == "fp") & (F("confidence") >= min_confidence))
    return wrong.to_patches(pred_field).sort_by("confidence", reverse=True)


def false_alarms(view, pred_field: str, gt_field: str = "ground_truth", iou: float = 0.5,
                 eval_key: str | None = None, min_confidence: float = 0.0):  # fmt: skip
    """Kept for compatibility: review(..., kind="false_alarms")."""
    return review(view, pred_field, gt_field, iou, eval_key, min_confidence, "false_alarms")

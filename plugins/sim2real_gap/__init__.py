"""sim2real-gap: how different are two image sets, and where is the difference?

Operators:
    measure_sim2real_gap   two views (one dataset or two) -> separability with
                           its spread, sanity checks, Frechet distance, a
                           per-sample (or per-object) realism score and
                           coverage gap, a breakdown by any field, two saved
                           views and an embedding map
    review_detector        a detector's false alarms (most confident first)
                           or misses, as patches
"""

import importlib.util
import os

import fiftyone.operators as foo
import fiftyone.operators.types as types


def _core():
    """Load gap.py from this folder (the plugin loader does not import as a package)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gap.py")
    spec = importlib.util.spec_from_file_location("sim2real_gap_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _detection_fields(dataset):
    import fiftyone as fo

    return sorted(dataset.get_field_schema(embedded_doc_type=fo.Detections))


def _group_fields(dataset):
    import fiftyone as fo

    kinds = (fo.StringField, fo.IntField, fo.BooleanField)
    skip = {"id", "filepath", "media_type"}
    return sorted(n for n, f in dataset.get_field_schema().items() if isinstance(f, kinds) and n not in skip)


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f} %"


def report(summary: dict) -> str:
    """The run as Markdown, for the App and for logs."""
    level = "objects" if summary["level"] == "objects" else "images"
    lines = [
        f"### Separability: **{pct(summary['separability'])}** (± {pct(summary['separability_std'])} across folds)",
        f"_{summary['verdict']}_",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| Sanity: simulated vs simulated | {pct(summary['sanity_sim_vs_sim'])} (should be near 50 %) |",
        f"| Sanity: real vs real | {pct(summary['sanity_real_vs_real'])} (should be near 50 %) |",
        f"| Frechet distance ({summary['model']}) | {summary['frechet']:.3f} |",
        f"| {level.capitalize()}: simulated / real | {summary['sim_items']} / {summary['real_items']} |",
        f"| Mean realism of simulated {level} | {pct(summary['realism_sim_mean'])} |",
    ]
    if summary.get("groups"):
        lines += ["", "**Closest to real first** (by separability, then Frechet distance):", "",
                  "| Value | Items | Separability | Frechet | Mean realism |", "| --- | --- | --- | --- | --- |"]
        lines += [f"| {g['value']} | {g['items']} | {pct(g['separability'])} | "
                  f"{'-' if g.get('frechet') is None else format(g['frechet'], '.3f')} | {pct(g.get('realism'))} |"
                  for g in summary["groups"]]  # fmt: skip
    f = summary["fields"]
    lines += ["", f"Per-item fields: `{f['realism']}` (probability of looking real), "
              f"`{f['nearest']}` (distance to the nearest item of the other set), `{f['domain']}`."]  # fmt: skip
    if summary.get("views"):
        v = summary["views"]
        lines += [f"Saved views: `{v['realistic_sim']}` (best synthetic data first), "
                  f"`{v['uncovered_real']}` (what the simulation is missing first)."]  # fmt: skip
    if summary.get("brain_key"):
        lines += [f"Embedding map: brain key `{summary['brain_key']}`, colour by `{f['domain']}`."]
    return "\n".join(lines)


class MeasureSim2RealGap(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="measure_sim2real_gap",
            label="Measure the sim-to-real gap",
            description="How different are two image sets, and which images make the difference?",
            icon="compare_arrows",
            dynamic=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
        )

    def resolve_input(self, ctx):
        import fiftyone as fo

        inputs = types.Object()
        views = ctx.dataset.list_saved_views() if ctx.dataset else []
        if not views:
            inputs.md("Save a view of your simulated (or first) image set first.", name="help")
            return types.Property(inputs)
        inputs.enum("sim_view", views, label="Simulated images (saved view)", required=True)

        datasets = fo.list_datasets()
        inputs.enum("real_dataset", datasets, default=ctx.dataset.name, label="Real images: dataset",
                    description="This dataset, or another one (no embedding map across datasets)")  # fmt: skip
        other = ctx.params.get("real_dataset") or ctx.dataset.name
        real_views = fo.load_dataset(other).list_saved_views() if other in datasets else []
        inputs.enum("real_view", real_views, label="Real images (saved view)", required=True)

        inputs.enum("level", ["images", "objects"], default="images", label="Compare",
                    description="Whole images, or object crops from a detections field")  # fmt: skip
        if ctx.params.get("level") == "objects":
            inputs.enum("patches_field", _detection_fields(ctx.dataset), label="Objects (detections field)",
                        required=True)  # fmt: skip

        core = _core()
        inputs.enum("model", list(core.MODELS), default=core.DEFAULT_MODEL, label="Embedding model")
        default_field = "clip_embedding" if "clip" in (ctx.params.get("model") or core.DEFAULT_MODEL) else "dinov2_embedding"
        inputs.str("embeddings_field", default=default_field, label="Embeddings field",
                   description="Reused if present, computed where missing")  # fmt: skip
        inputs.enum("group_field", _group_fields(ctx.dataset), label="Break down by (optional)",
                    description="E.g. world, weather or camera: which simulated conditions are closest to real")  # fmt: skip
        inputs.str("prefix", default="s2r", label="Field prefix")
        inputs.str("brain_key", default="sim2real_gap", label="Brain key for the embedding map")
        inputs.int("max_items", default=5000, min=50, label="Max items per set for the set-level scores")
        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        return None  # let the user choose

    def execute(self, ctx):
        import fiftyone as fo

        core = _core()
        sim_view = ctx.dataset.load_saved_view(ctx.params["sim_view"])
        real_dataset = ctx.params.get("real_dataset") or ctx.dataset.name
        source = ctx.dataset if real_dataset == ctx.dataset.name else fo.load_dataset(real_dataset)
        real_view = source.load_saved_view(ctx.params["real_view"])
        summary = core.measure(
            sim_view, real_view,
            embeddings_field=ctx.params.get("embeddings_field") or "clip_embedding",
            model_name=ctx.params.get("model") or core.DEFAULT_MODEL,
            patches_field=ctx.params.get("patches_field") if ctx.params.get("level") == "objects" else None,
            group_field=ctx.params.get("group_field") or None,
            prefix=ctx.params.get("prefix") or core.DEFAULT_PREFIX,
            brain_key=ctx.params.get("brain_key") or None,
            max_items=ctx.params.get("max_items") or 5000,
            log=lambda m: ctx.set_progress(label=m) if hasattr(ctx, "set_progress") else None,
        )  # fmt: skip
        ctx.ops.reload_dataset()
        return {"report": report(summary), **{k: summary[k] for k in ("separability", "frechet")}}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("report", label="Result", view=types.MarkdownView())
        return types.Property(outputs)


class ReviewDetector(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="review_detector",
            label="Review a detector's false alarms or misses",
            description="Predictions that match no ground truth (most confident first), or objects it missed",
            icon="report",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        fields = _detection_fields(ctx.dataset) if ctx.dataset else []
        if len(fields) < 2:
            inputs.md("Needs a predictions field and a ground-truth field (Detections).", name="help")
            return types.Property(inputs)
        inputs.enum("pred_field", fields, label="Predictions", required=True)
        inputs.enum("gt_field", fields, label="Ground truth", required=True,
                    default="ground_truth" if "ground_truth" in fields else None)  # fmt: skip
        inputs.enum("kind", ["false_alarms", "misses"], default="false_alarms", label="Show")
        inputs.float("iou", default=0.5, min=0.1, max=0.95, label="IoU to count as a match")
        inputs.float("min_confidence", default=0.3, min=0.0, max=1.0, label="False alarms at or above")
        inputs.str("save_as", default="", label="Save the result as a view named (optional)")
        return types.Property(inputs)

    def execute(self, ctx):
        core = _core()
        kind = ctx.params.get("kind", "false_alarms")
        patches = core.review(
            ctx.view, ctx.params["pred_field"], gt_field=ctx.params.get("gt_field", "ground_truth"),
            iou=float(ctx.params.get("iou", 0.5)), min_confidence=float(ctx.params.get("min_confidence", 0.0)),
            kind=kind,
        )  # fmt: skip
        if ctx.params.get("save_as"):
            ctx.dataset.save_view(ctx.params["save_as"], patches, overwrite=True)
        ctx.ops.set_view(view=patches)
        return {"count": len(patches), "kind": kind.replace("_", " ")}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.int("count", label="Objects (now shown, worst first)")
        outputs.str("kind", label="Showing")
        return types.Property(outputs)


class ReviewFalseAlarms(ReviewDetector):
    """The 1.0 name, kept so existing scripts keep working."""

    @property
    def config(self):
        config = super().config
        config.name, config.label, config.unlisted = "review_false_alarms", "Review false alarms", True
        return config


def register(plugin):
    plugin.register(MeasureSim2RealGap)
    plugin.register(ReviewDetector)
    plugin.register(ReviewFalseAlarms)

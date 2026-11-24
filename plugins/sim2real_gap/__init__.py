"""sim2real-gap: measure how different simulated and real images look, and
review what a detector mistakes for its target on real images.

Operators:
    measure_sim2real_gap   two saved views (simulated, real) -> separability
                           score, sanity checks, and an embedding map
    review_false_alarms    a detector's false alarms, most confident first
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

    return sorted(
        name for name, field in dataset.get_field_schema(embedded_doc_type=fo.Detections).items()
    )


def for_display(summary: dict) -> dict:
    """The operator's output: scores as percentages with one decimal (the
    stored summary in dataset.info keeps fractions)."""
    shown = dict(summary)
    for key in ("separability", "sanity_sim_vs_sim", "sanity_real_vs_real"):
        shown[key] = f"{100 * summary[key]:.1f} %"
    return shown


class MeasureSim2RealGap(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="measure_sim2real_gap",
            label="Measure the sim-to-real gap",
            description="How easily can a linear classifier tell simulated images from real ones?",
            dynamic=True,
            allow_delegated_execution=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        views = ctx.dataset.list_saved_views() if ctx.dataset else []
        if len(views) < 2:
            inputs.md("Save two views first: one of simulated images and one of real images.", name="help")
            return types.Property(inputs)
        inputs.enum("sim_view", views, label="Simulated images (saved view)", required=True)
        inputs.enum("real_view", views, label="Real images (saved view)", required=True)
        inputs.str("embeddings_field", default="clip_embedding", label="Embeddings field",
                   description="Computed with CLIP for samples that lack it")  # fmt: skip
        inputs.str("brain_key", default="sim2real_gap", label="Brain key for the embedding map")
        return types.Property(inputs)

    def execute(self, ctx):
        core = _core()
        dataset = ctx.dataset
        summary = core.measure(
            dataset,
            dataset.load_saved_view(ctx.params["sim_view"]),
            dataset.load_saved_view(ctx.params["real_view"]),
            embeddings_field=ctx.params.get("embeddings_field") or "clip_embedding",
            brain_key=ctx.params.get("brain_key") or None,
        )
        return for_display(summary)

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("separability", label="Separability (50 % = alike, 100 % = trivially different)")
        outputs.str("verdict", label="Verdict")
        outputs.str("sanity_sim_vs_sim", label="Sanity: sim vs sim (should be near 50 %)")
        outputs.str("sanity_real_vs_real", label="Sanity: real vs real (should be near 50 %)")
        outputs.int("sim_images", label="Simulated images")
        outputs.int("real_images", label="Real images")
        outputs.str("brain_key", label="Embedding map (brain key): open the Embeddings panel, colour by sim2real_domain")
        return types.Property(outputs)


class ReviewFalseAlarms(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="review_false_alarms",
            label="Review false alarms",
            description="Show a detector's predictions that match no ground truth, most confident first",
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
        inputs.float("iou", default=0.5, min=0.1, max=0.95, label="IoU to count as a match")
        inputs.float("min_confidence", default=0.3, min=0.0, max=1.0, label="Only alarms at or above")
        return types.Property(inputs)

    def execute(self, ctx):
        core = _core()
        patches = core.false_alarms(
            ctx.view,
            ctx.params["pred_field"],
            gt_field=ctx.params.get("gt_field", "ground_truth"),
            iou=float(ctx.params.get("iou", 0.5)),
            min_confidence=float(ctx.params.get("min_confidence", 0.0)),
        )
        ctx.ops.set_view(view=patches)
        return {"false_alarms": len(patches)}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.int("false_alarms", label="False alarms (now shown, most confident first)")
        return types.Property(outputs)


def register(plugin):
    plugin.register(MeasureSim2RealGap)
    plugin.register(ReviewFalseAlarms)

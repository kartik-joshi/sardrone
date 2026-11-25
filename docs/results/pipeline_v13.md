# The first end-to-end pipeline run: v13 (issue #10)

One command, `runs/run_v13.sh` →
`python -m sardrone.pipeline --version v13 --export v13-6k --max-train 6000 --change "rich worlds (vehicle and rock meshes, walking and child bodies), realism, 300 real" --export-arg=--train --export-arg=train-v13-rich --export-arg=--realism --export-arg=--add-real --export-arg=300 --train-arg=--epochs --train-arg=10 --run`,
took the 30 new rich-world episodes to an updated scorecard. No stage was run by hand.

| Stage | What happened |
|---|---|
| wait | 09:14–10:05, for the shared GPU (a lock-file handshake with another project on this Mac) |
| ingest | 30 new episodes (rich-001 to rich-030) → `sar-frames` now holds 82,530 frames from 176 episodes, 113,063 boxes |
| curate | 12,345 new frames embedded (CLIP), 4,181 near-duplicates tagged; saved view `train-v13-rich` 11,986 frames |
| gate | **open**: 24,022 new hard frames since the last training (threshold 200); hard cases 45,723 against 21,701. (A dry run before ingest and curate had counted 16,985; the real run counts after curate, so the rich-world hard cases are included.) |
| export | `v13-6k`, hash `688cb09ad9b13984`: 5,999 rich-world frames sampled by episode, camera realism, plus 300 SARD pool images = 6,299 training images; validation `val-v1`, 2,861 |
| train | YOLO11n, 10 epochs, about 21 min each (Apple M4 GPU) |
| evaluate | scorecard row v13: simulation 0.909, **real SARD 0.614 (0.599 to 0.632)** mAP@0.5 |
| total | 10:05 → 13:47 (3 h 41 min after the GPU was free) |

The full log is `runs/logs/v13_pipeline.log` (runs are not committed). Excerpt:

```
== v13 pipeline starting Wed Nov 25 10:05:54 EST 2026
rich-001: 586 frames, 1379 boxes (7 tiny boxes dropped, 22 ground frames skipped)
...
dataset 'sar-frames': 82530 frames from 176 episodes, 113063 boxes
embedding 12345 frames with clip-vit-base32-torch ...
82530 frames; 4181 near-duplicates tagged (threshold 0.004; ...)
[gate] retrain=True: 24022 new hard frames (threshold 200); hard cases now 45723, at last training 21701
  "name": "v13-6k", "hash": "688cb09ad9b13984", ... "images": 6299 ...
ALL DONE Wed Nov 25 13:46:59 EST 2026
```

**One side effect, undone on purpose:** the curate stage redraws
`docs/images/phase2e_umap.png`, the Phase 2E figure that the report describes
with the pilot data. That file was restored to its committed version, so an
old figure does not silently change meaning.

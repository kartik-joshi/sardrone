"""Environment for everything that uses FiftyOne. Import this FIRST.

    from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

FiftyOne reads its settings when it is imported, so they must be in place
before that. Everything is kept inside this project folder:

* the database                 .fiftyone/db
* downloaded FiftyOne models   .cache/fiftyone_models
* Hugging Face model weights   .cache/huggingface

`setdefault` means anything you export yourself wins.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache"

os.environ.setdefault("VFF_MULTIMODAL", "1")
os.environ.setdefault("FIFTYONE_DATABASE_DIR", str(PROJECT_ROOT / ".fiftyone" / "db"))
os.environ.setdefault("FIFTYONE_DO_NOT_TRACK", "true")
os.environ.setdefault("FIFTYONE_MODEL_ZOO_DIR", str(CACHE_DIR / "fiftyone_models"))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
# Some operations are not implemented on Apple's GPU backend yet; let those
# fall back to the CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

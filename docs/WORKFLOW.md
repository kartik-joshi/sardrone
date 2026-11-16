# The data loop as a workflow

```
 episodes (.mcap)                                  ┌──────────── scorecard
      │                                            │
   ingest ──> curate ──> GATE ──> export ──> train ──> evaluate
 (frames,    (embed,     ≥ 200    (YOLO set   (YOLO11n)  (sim-test,
  labels)    de-dup,     new       + hash)               sard-test,
             views)      hard                            gap)
                         frames?
```

| Stage | Command | Reads | Writes |
| --- | --- | --- | --- |
| ingest | `make frames` | `data/episodes/*.mcap` | FiftyOne `sar-frames`, `data/frames/` |
| curate | `make curate` | `sar-frames` | embeddings, `duplicate` tags, saved views |
| gate | part of `make pipeline` | `sar-hard-cases`, `runs/pipeline_state.json` | a decision, printed |
| export | `make export` | views `train-v1`, `val-v1` | `runs/exports/<name>/` + data hash |
| train | `make train` | an export | `runs/train/<version>/` + `run.json` |
| evaluate | `make evaluate` | a model, `sim-test`, `sard-test` | `docs/results/scorecard.*` |

`make pipeline` prints what would run; `make pipeline RUN=1` runs it.

## The gate

Retraining takes hours. The gate lets it happen only when the data has
changed in a way that matters: at least 200 more frames in
`sar-hard-cases` (people lying, occluded, small, or at dusk) than at the
last training. The count at each training is kept in
`runs/pipeline_state.json`. Ingest and curate always run, since they are
cheap and the gate needs their result.

The threshold of 200 is the plan's number. It has not been tuned.

## Why this is not an Airflow DAG

The plan asks for a small Airflow DAG, or a reason for something lighter.
This is the lighter option, and the reasons are specific to this project:

- **One machine, one person, one fixed order.** Six stages that always run
  in sequence do not need a scheduler, a web server and a metadata
  database.
- **Dependencies.** Airflow pins a large set of packages and would not
  share an environment with FiftyOne, PyTorch and Ultralytics. It would
  need its own environment and would call these commands from outside
  anyway.
- **The useful properties are already here.** Each stage is a separate
  command that can be rerun on its own. Every run records its inputs (data
  hash, git commit). Batches resume after a crash. The gate's decision is
  printed and its state is a file.

What is missing without a scheduler: retries with back-off, alerting,
a history of runs in one place, and running stages on different machines.

## How it would map onto Airflow

Each stage is already a shell command, so each becomes one task. The gate
becomes a short-circuit task: when it returns false, everything after it
is skipped. This sketch has NOT been run; Airflow is not installed here.

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator

def gate_is_open() -> bool:
    from sardrone.pipeline import count_hard_cases, decide, read_state
    return decide(count_hard_cases(), read_state()).retrain

with DAG("sardrone_retrain", schedule="@daily", catchup=False) as dag:
    ingest = BashOperator(task_id="ingest", bash_command="make frames")
    curate = BashOperator(task_id="curate", bash_command="make curate")
    gate = ShortCircuitOperator(task_id="gate", python_callable=gate_is_open)
    export = BashOperator(task_id="export", bash_command="make export EXPORT={{ ds_nodash }}")
    train = BashOperator(task_id="train", bash_command="make train EXPORT={{ ds_nodash }} VERSION={{ ds_nodash }}")
    evaluate = BashOperator(task_id="evaluate", bash_command="make evaluate VERSION={{ ds_nodash }}")
    ingest >> curate >> gate >> export >> train >> evaluate
```

In production the same shape holds, with different machinery around it:
episodes arrive in object storage instead of a folder, training runs as a
container job on a GPU cluster, and evaluation runs against a registry of
frozen test sets. The plan lists storing episodes in S3 and running
evaluation as a container job as optional; neither is done here.

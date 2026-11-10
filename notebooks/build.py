"""Writes the six notebooks from plain cell lists, so they stay easy to review and diff.

    python notebooks/build.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


def write(name, cells):
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3.12", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    (HERE / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("wrote", name)


# ---------------------------------------------------------------------------------------------------------------
# Shared cells
# ---------------------------------------------------------------------------------------------------------------
SETTINGS = code('''
# Settings used by every notebook. Change them here if your project differs.
import os, json, time
from pathlib import Path

NAMESPACE  = open("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read().strip()   # this project
WS         = Path("/opt/app-root/src/workspace")   # the project volume, as mounted in this workbench
PVC        = "workspace"                           # its name; training and serving pods mount it at /workspace
REPO       = Path.cwd().parent                     # this repository (data/, sdg_flows/)
BASE_MODEL = "ibm-granite/granite-4.1-3b"
EXPERIMENT = "sentinel7"                           # MLflow experiment
NODE_SELECTOR = {}   # only with ReadWriteOnce storage: keep every pod that mounts the volume on one node
TRAINING_IMAGE = ("registry.redhat.io/rhoai/odh-th-torch-cuda-py312-rhel9@sha256:"   # OpenShift AI 3.5 Training Hub image
                  "884aeeb039f9592252fe0c518acf31803a208757d4fa6500fe79884a022ea52d")

os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", EXPERIMENT)
print("project:", NAMESPACE, "| volume:", WS, "| MLflow:", os.environ.get("MLFLOW_TRACKING_URI", "not enabled"))
''')

K8S = code('''
# Create OpenShift AI resources with the workbench's own identity (no token to paste).
import yaml
from kubernetes import client as k8s, config, dynamic
config.load_incluster_config()
dyn = dynamic.DynamicClient(k8s.ApiClient())

def apply(manifest: str):
    """Create or update every object in a YAML string, in this project."""
    for obj in yaml.safe_load_all(manifest):
        res = dyn.resources.get(api_version=obj["apiVersion"], kind=obj["kind"])
        name = obj["metadata"]["name"]
        try:
            dyn.create(res, body=obj, namespace=NAMESPACE); print("created", obj["kind"], name)
        except k8s.ApiException as e:
            if e.status != 409: raise
            dyn.patch(res, body=obj, name=name, namespace=NAMESPACE, content_type="application/merge-patch+json")
            print("updated", obj["kind"], name)

def run_job(manifest: str, show=None):
    """(Re)start a Kubernetes Job and stream its log. `show` keeps only lines containing one of these words."""
    obj = yaml.safe_load(manifest); name = obj["metadata"]["name"]
    batch, core = k8s.BatchV1Api(), k8s.CoreV1Api()
    try:
        batch.delete_namespaced_job(name, NAMESPACE, propagation_policy="Foreground")
        while batch.list_namespaced_job(NAMESPACE, field_selector=f"metadata.name={name}").items: time.sleep(3)
    except k8s.ApiException: pass
    if NODE_SELECTOR: obj["spec"]["template"]["spec"]["nodeSelector"] = NODE_SELECTOR
    batch.create_namespaced_job(NAMESPACE, obj)
    while True:
        pods = core.list_namespaced_pod(NAMESPACE, label_selector=f"job-name={name}").items
        if pods and pods[0].status.phase in ("Running", "Succeeded", "Failed"): break
        time.sleep(5)
    for raw in core.read_namespaced_pod_log(pods[0].metadata.name, NAMESPACE, follow=True, _preload_content=False):
        for line in raw.decode(errors="replace").splitlines():
            if show is None or any(w in line for w in show): print(line, flush=True)
    s = batch.read_namespaced_job(name, NAMESPACE).status
    print(f"\\nJob {name}: {'succeeded' if s.succeeded else 'FAILED - see the log above'}")

def wait_ready(kind, api_version, name, timeout=1800):
    res = dyn.resources.get(api_version=api_version, kind=kind); t0 = time.time()
    while time.time() - t0 < timeout:
        conds = (dyn.get(res, name=name, namespace=NAMESPACE).to_dict().get("status") or {}).get("conditions", [])
        if any(c["type"] == "Ready" and c["status"] == "True" for c in conds):
            return print(f"{name} is ready ({time.time() - t0:.0f} s)")
        time.sleep(10)
    raise TimeoutError(name)
''')

CHECKERS_IMPORT = code('''
import sys; sys.path.insert(0, str(WS / "code"))     # report_check.py and burst_rules.py, written by notebook 01
from report_check import check, scorecard as report_scorecard
from burst_rules import check as burst_check, scorecard as burst_scorecard
''')

# ---------------------------------------------------------------------------------------------------------------
# 01 · SDG Hub
# ---------------------------------------------------------------------------------------------------------------
write("01_sdg_hub.ipynb", [
    md('''
# 01 · Create training data with SDG Hub

**What you will do**
1. Set up the project volume and download the base model.
2. Look at the two kinds of synthetic data used in this example.
3. Write down the rules as code: a **report checker** and a **burst-line checker**.
4. Run an **SDG Hub flow**: a teacher model writes burst lines, and the checker keeps only the ones that follow every rule.

**Time:** about 15 minutes (plus the teacher's generation time). **GPU:** none in this notebook.

| | Dev environment | On OpenShift AI |
|---|---|---|
| Data pipeline | One-off scripts | SDG Hub flows: YAML files of reusable blocks, built-in flows to start from |
| Teacher model | Outside API, data leaves your environment | Any OpenAI-compatible model, ideally served in this project |
| Packages | Pinned by hand | SDG Hub comes from Red Hat's vetted package index |

> **The scenario is fictional.** Blue Defence runs Sentinel-7 uncrewed patrol vessels. Operators send terse radio
> notes; the ops room needs exact JSON reports (BD-SCR), and the shore needs one short satellite message per report
> (the *burst line*). All data is synthetic.
'''),
    md("## 1 · Settings"),
    SETTINGS,
    md('''
## 2 · The project volume and the base model

The volume holds everything the later notebooks share: the base model, data, checkpoints and exports.
Create it once with `deploy/00-project-setup.yaml`, and attach it to this workbench at `/opt/app-root/src/workspace`
(Workbench settings > Cluster storage).
'''),
    code('''
for sub in ["models", "data", "checkpoints", "results", "export", "code"]:
    (WS / sub).mkdir(parents=True, exist_ok=True)

# Copy the synthetic data from this repository onto the volume, where training pods can read it
import shutil
for d in ["edgecopilot", "burst"]:
    shutil.copytree(REPO / "data" / d, WS / "data" / d, dirs_exist_ok=True)
print(sorted(str(p.relative_to(WS)) for p in (WS / "data").rglob("*.jsonl")))
'''),
    code('''
# Download Granite 4.1-3B onto the volume (about 7 GB, a few minutes). Skip if it is already there.
from huggingface_hub import snapshot_download
target = WS / "models" / "granite-4.1-3b"
if not (target / "config.json").exists():
    snapshot_download(BASE_MODEL, local_dir=target,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
print(sorted(p.name for p in target.iterdir()))
'''),
    md('''
## 3 · Two kinds of synthetic data

| Use case | Data | How it was made |
|---|---|---|
| **Edge Copilot** (SFT): radio note → exact JSON report | `data/edgecopilot/` · 2,400 train, 150 val, 300 test | **Programmatic.** A script draws a correct report, then writes it as a messy note. The correct answer is known exactly. `train-v2` adds 600 radio-style notes. |
| **Burst line** (GRPO): report → one satellite message | `data/burst/` · 400 train, 100 val, 200 test | **SDG Hub.** A teacher model writes the messages; a rule checker keeps only correct ones. |

Programmatic data is ideal when you can generate the answer. When many answers are valid, a teacher model plus a
checker is the practical route, and that is what SDG Hub is built for.
'''),
    code('''
ex = json.loads((WS / "data/edgecopilot/train.jsonl").open().readline())["messages"]
print("NOTE  :", ex[1]["content"])
print("REPORT:", json.dumps(json.loads(ex[2]["content"]), indent=1))
'''),
    md('''
## 4 · The rules, as code

Two small checkers. Every later notebook uses them, and the burst checker is also the **reward** for GRPO.
Each cell saves its code to the project volume, so training pods can import it too.

### Report checker: is the JSON report exactly right, and did the model invent a field?
'''),
    code('''
%%writefile /opt/app-root/src/workspace/code/report_check.py
"""Check a BD-SCR JSON report against the correct one."""
import json, re

OPTIONAL = ["mode", "position", "speed_kts", "heading_deg", "battery_pct", "visibility"]

def parse(text):
    """First JSON object in a model answer, or None."""
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M)
    start, depth = text.find("{"), 0
    if start < 0:
        return None
    for i, ch in enumerate(text[start:], start):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            try: return json.loads(text[start:i + 1])
            except json.JSONDecodeError: return None
    return None

def _canon(r):   # contacts and faults may come in any order
    key = lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False)
    return {**r, **{k: sorted(r[k], key=key) for k in ("contacts", "faults") if isinstance(r.get(k), list)}}

def check(text, gold):
    p = parse(text)
    if not isinstance(p, dict):
        return {"valid": False, "exact": False, "invented": []}
    return {"valid": True, "exact": _canon(p) == _canon(gold),
            "invented": [k for k in OPTIONAL if gold.get(k) is None and p.get(k) is not None]}

def scorecard(answers, golds):
    res = [check(a, g) for a, g in zip(answers, golds)]
    pct = lambda xs: round(100 * sum(xs) / len(res), 1)
    return {"notes": len(res), "Valid JSON %": pct(r["valid"] for r in res),
            "Exact report %": pct(r["exact"] for r in res),
            "Invented a field %": pct(bool(r["invented"]) for r in res)}
'''),
    md('''
### Burst-line checker: rules B1–B7 of the vessel manual

The burst line is one message of at most 160 characters. Many wordings are valid, so there is no single right
answer to copy, but every answer can be **checked**, rule by rule.
'''),
    code('''
%%writefile /opt/app-root/src/workspace/code/burst_rules.py
"""Rules B1-B7 for the satellite burst line, as code. Input `task` = {"priority": ..., "report": {...}}."""
import json, re

LIMIT = 160
MANUAL_CODES = {"E-102", "E-104", "E-117", "E-203", "E-221", "E-238", "E-246", "E-305", "E-312", "E-331",
                "E-408", "E-415", "E-502", "E-519"}

def required(task):
    """What the line must carry, as (rule, label, pattern)."""
    r, pri = task["report"], task["priority"]
    checks = [("B1", "priority", rf"^\\s*{pri}\\b"), ("B1", "vessel", rf"\\b{re.escape(r['vessel'])}\\b"),
              ("B2", "time", rf"\\b{r['dtg'].split()[0]}\\b")]
    for c in r.get("contacts") or []:
        if c["range_nm"] <= 3:                                   # B3: contacts within 3 nm, bearing + range
            rng = f"{c['range_nm']:g}"
            checks.append(("B3", f"contact {c['bearing_deg']:03d}",
                           rf"\\b{c['bearing_deg']:03d}\\b[^/]{{0,12}}?\\b{re.escape(rng)} ?NM\\b"))
            if c.get("ais") is not True:
                tag = "OFF" if c.get("ais") is False else "UNK"
                checks.append(("B3", f"AIS {tag}", rf"\\bAIS {tag}\\b"))
    for f in r.get("faults") or []:                              # B4: fault codes, unknown ones marked UNK
        unk = f["code"] not in MANUAL_CODES
        checks.append(("B4", f["code"] + (" UNK" if unk else ""),
                       rf"\\b{re.escape(f['code'])}\\b" + (r" ?UNK\\b" if unk else r"(?! ?UNK\\b)")))
    if r.get("battery_pct") is not None and r["battery_pct"] < 30:   # B5: low battery
        checks.append(("B5", "battery", rf"\\bBATT ?{r['battery_pct']}\\b"))
    return checks

def _allowed_numbers(r):                                         # B6: no number that is not in the report
    nums = set(re.findall(r"\\d+(?:\\.\\d+)?", json.dumps({k: v for k, v in r.items() if k != "position"})))
    nums |= {f"{c['bearing_deg']:03d}" for c in r.get("contacts") or []} | {r["dtg"].split()[0][:-1], "3", "30"}
    return {float(n) for n in nums}

def check(text, task):
    """Rule-by-rule result for one answer: list of (rule, label, passed)."""
    line = (text or "").strip().splitlines()[-1].strip() if (text or "").strip() else ""
    out = [(rule, label, bool(re.search(rx, line))) for rule, label, rx in required(task)]
    stray = [n for n in re.findall(r"(?<![A-Z])\\d+(?:\\.\\d+)?", line.replace("S7-", "S7 "))
             if float(n) not in _allowed_numbers(task["report"])]
    out.append(("B6", "no invented number", not stray))
    out += [("B7", "capitals", line == line.upper() and bool(line)),
            ("B7", f"<= {LIMIT} characters", 0 < len(line) <= LIMIT),
            ("B7", "ends with OUT", line.rstrip(". ").endswith("OUT"))]
    return out

def reward(text, task):
    """0..1: half for the share of rules met, half for meeting all of them; halved if too long."""
    if not (text or "").strip(): return 0.0
    res = check(text, task)
    score = 0.5 * sum(ok for *_, ok in res) / len(res) + 0.5 * all(ok for *_, ok in res)
    too_long = not next(ok for _, label, ok in res if label.startswith("<="))
    return round(score * (0.5 if too_long else 1.0), 4)

def reward_fn(data_source, solution_str, ground_truth):
    """The signature Training Hub's lora_grpo calls."""
    return reward(solution_str, json.loads(ground_truth))

def scorecard(answers, tasks):
    res = [check(a, t) for a, t in zip(answers, tasks)]
    pct = lambda xs: round(100 * sum(xs) / len(res), 1)
    return {"reports": len(res), "All rules met %": pct(all(ok for *_, ok in r) for r in res),
            "Missed a required item %": pct(any(not ok for rule, _, ok in r if rule <= "B5") for r in res),
            "Invented a number %": pct(not ok for r in res for rule, _, ok in r if rule == "B6"),
            "Mean reward": round(sum(reward(a, t) for a, t in zip(answers, tasks)) / len(res), 3)}
'''),
    code('''
import sys; sys.path.insert(0, str(WS / "code"))
import burst_rules
task = json.loads((WS / "data/burst/test.jsonl").open().readline())["ground_truth"]
task = json.loads(task)
print("PRIORITY:", task["priority"]); print("REPORT  :", json.dumps(task["report"])[:300], "...\\n")
for line in ["FLASH S7-03 OUT", "this is not a burst line"]:
    print(repr(line), "-> reward", burst_rules.reward(line, task))
    for rule, label, ok in burst_rules.check(line, task):
        print(f"   {'PASS' if ok else 'FAIL'}  {rule}  {label}")
'''),
    md('''
## 5 · SDG Hub: a teacher writes burst lines, the checker keeps the good ones

An SDG Hub **flow** is a YAML file of blocks that run in order. Our flow has three:

`PromptBuilderBlock` (fill the prompt) → `LLMChatBlock` (call the teacher model) → `LLMResponseExtractorBlock` (take the answer)

Built-in flows also add judge and filter blocks. Here the filter is our rule checker, applied right after the flow.
'''),
    code('''
%pip install -q sdg-hub      # served from Red Hat's vetted package index in OpenShift AI workbenches
'''),
    code('''
from sdg_hub import Flow, FlowRegistry
FlowRegistry.discover_flows()                   # built-in flows you can start from
print((REPO / "sdg_flows/burst_teacher/flow.yaml").read_text())
'''),
    md('''
### The teacher model

Any OpenAI-compatible endpoint works. Best practice is a larger model **served in this project**, so no data leaves
it: for example Granite 4.1-8B, deployed the same way notebook 04 deploys the 3B model. We used Granite 4.1-8B.

No teacher yet? Set `TEACHER_URL = ""` and the next cells use the teacher answers we generated (`data/burst/`).
'''),
    code('''
TEACHER_URL   = ""                                # e.g. "https://granite-41-8b-kserve-workload-svc.<project>.svc:8000/v1"
TEACHER_MODEL = "granite-4.1-8b"
N_REPORTS     = 400                               # all training reports; use 40 for a quick test
'''),
    code('''
import pandas as pd
rows = [json.loads(l) for l in (WS / "data/burst/train.jsonl").open()][:N_REPORTS]
seed = pd.DataFrame([{"rules": r["messages"][0]["content"], "task": r["messages"][1]["content"]} for r in rows])

if TEACHER_URL:
    import litellm; litellm.ssl_verify = False     # in-cluster services use the cluster's own certificate
    flow = Flow.from_yaml(str(REPO / "sdg_flows/burst_teacher/flow.yaml"))
    flow.set_model_config(model=f"hosted_vllm/{TEACHER_MODEL}", api_base=TEACHER_URL, api_key="none")
    t0 = time.time()
    out = flow.generate(seed, max_concurrency=16)
    answers = [str(a).strip() for a in out["burst_content"]]
    print(f"teacher wrote {len(answers)} burst lines in {time.time() - t0:.0f} s")
else:
    teacher = [json.loads(l) for l in (WS / "data/burst/train-sft-teacher.jsonl").open()][:N_REPORTS]
    answers = [t["messages"][-1]["content"] for t in teacher]
    print(f"using {len(answers)} teacher answers from the repository")
'''),
    code('''
# The filter: keep only answers that meet every rule
tasks = [json.loads(r["ground_truth"]) for r in rows]
keep = [all(ok for *_, ok in burst_rules.check(a, t)) for a, t in zip(answers, tasks)]
print(burst_rules.scorecard(answers, tasks))
print(f"kept {sum(keep)} of {len(keep)}")

with open(WS / "data/burst/sft-teacher.jsonl", "w") as all_f, open(WS / "data/burst/sft-filtered.jsonl", "w") as ok_f:
    for r, a, k in zip(rows, answers, keep):
        line = json.dumps({"messages": r["messages"][:2] + [{"role": "assistant", "content": a}]}, ensure_ascii=False) + "\\n"
        all_f.write(line)
        if k: ok_f.write(line)

bad = next(i for i, k in enumerate(keep) if not k)
print("\\nA rejected answer:", answers[bad])
print([f"{rule} {label}" for rule, label, ok in burst_rules.check(answers[bad], tasks[bad]) if not ok])
'''),
    code('''
# Record the dataset in MLflow, next to the training runs that will use it
import mlflow
with mlflow.start_run(run_name="sdg-burst-teacher"):
    mlflow.log_params({"teacher": TEACHER_MODEL if TEACHER_URL else "repository file", "reports": len(rows)})
    mlflow.log_metrics({"teacher_answers": len(answers), "kept_after_checker": sum(keep)})
    mlflow.log_artifact(str(WS / "data/burst/sft-filtered.jsonl"))
'''),
    md('''
## What we measured

| | Result |
|---|---|
| Teacher (Granite 4.1-8B) answers meeting every rule | 173 of 400 (43%), mostly missed contacts |
| SFT on **all** teacher answers | 31.5% of test messages meet every rule (worse than untuned, 42.5%) |
| SFT on the **173 checked** answers | 88.0% |

Filtering the teacher's output is what makes it useful. **Next:** notebook 02 picks the training method.
'''),
])

# ---------------------------------------------------------------------------------------------------------------
# 02 · Choose a method
# ---------------------------------------------------------------------------------------------------------------
write("02_choose_method.ipynb", [
    md('''
# 02 · Choose a method that fits your GPU

**What you will do**
1. Ask Training Hub's **memory estimator** how much GPU memory each method needs, before booking a GPU.
2. Meet **OSFT**, Red Hat's full fine-tuning method that protects what the model already knows.
3. Pick the method by the mistake the model makes.

**Time:** 5 minutes. **GPU:** none.

| | Dev environment | On OpenShift AI |
|---|---|---|
| Will it fit? | Run until out of memory | Memory estimator, per method, model and GPU |
| Methods | A different library and config each | One Training Hub call each: `sft`, `osft`, `lora_sft`, `lora_grpo`, `embedding_sft` |
| Keeping general skills | No protection | `osft` with one setting, `unfreeze_rank_ratio` |
'''),
    SETTINGS,
    md("## 1 · The memory estimator"),
    code('''
from training_hub.profiling.memory_estimator import estimate
MODEL = str(WS / "models/granite-4.1-3b")
for method in ["sft", "osft", "lora"]:
    try:
        print(method.upper(), estimate(training_method=method, model_path=MODEL, num_gpus=1,
                                       gpu_memory=24_000_000_000, max_tokens_per_gpu=4096))
    except Exception as e:
        print(method.upper(), "->", type(e).__name__, str(e)[:160])
'''),
    code('''
# What the estimator told us for Granite 4.1-3B on one GPU, against common GPU sizes
import matplotlib.pyplot as plt
need = {"Full SFT": (56, 73), "OSFT": (43, 55), "LoRA": (7.2, 9.1), "QLoRA (4-bit)": (3.0, 4.1)}
fig, ax = plt.subplots(figsize=(8, 3))
for i, (m, (lo, hi)) in enumerate(need.items()):
    ax.barh(i, hi, color="#c6c6c6"); ax.barh(i, lo, color="#0f62fe"); ax.text(hi + 1, i, f"{lo}–{hi} GiB", va="center")
for g, label in [(10, "share of an L4"), (24, "L4"), (48, "L40S"), (80, "H100")]:
    ax.axvline(g, color="#161616", lw=1, ls="--"); ax.text(g, len(need) - 0.4, label, rotation=90, va="bottom", fontsize=8)
ax.set_yticks(range(len(need)), need.keys()); ax.set_xlabel("GPU memory (GiB)"); ax.invert_yaxis(); plt.tight_layout()
'''),
    md('''
**Our choice:** LoRA. It fits a ~10 GB share of one NVIDIA L4, trains in minutes, and the adapter is 62 MB.
Red Hat's LoRA path runs on **Unsloth**, the same library a dev team would use on its own.

## 2 · OSFT: full fine-tuning that keeps existing skills

Standard fine-tuning and LoRA can make a model forget what it knew. **OSFT** (built by Red Hat AI Innovation) finds
the parts of the weights that hold existing skills, freezes them, and trains the rest. Red Hat reports 48.4% vs 23.0%
for standard fine-tuning when a model learns eight tasks one after another.

Use it when the model must keep its general abilities while it learns your domain. We did not run it here: for this
model it needs 43–55 GiB, more than our GPU share. The call has the same shape as `sft`:
'''),
    code('''
RUN_OSFT = False      # needs a GPU with about 48 GB or more for Granite 4.1-3B

if RUN_OSFT:
    from training_hub import osft
    osft(model_path=MODEL,
         data_path=str(WS / "data/edgecopilot/train.jsonl"),
         ckpt_output_dir=str(WS / "checkpoints/edgecopilot-osft"),
         unfreeze_rank_ratio=0.3,          # train 30% of each layer's directions, protect the other 70%
         effective_batch_size=16, max_tokens_per_gpu=4096, max_seq_len=1024,
         num_epochs=2, learning_rate=5e-6)
'''),
    md('''
## 3 · Pick the method by the mistake

| The model's mistake | Method | Training Hub | Our evidence |
|---|---|---|---|
| Wrong task or format | **SFT** (here LoRA) | `lora_sft` | Exact reports 6.3% → 92.7% |
| A gap in the training data | **More data, same recipe** | `lora_sft` | + 600 radio-style notes → 99.7–100% |
| The same mistake every time | **DPO** | TRL, in the same image | 92.7% → 99.7% |
| Breaks rules a checker can verify | **GRPO** | `lora_grpo` | Burst line 88% → 95% |
| Must keep general skills | **OSFT** | `osft` | Red Hat-reported, not run here |

GRPO learns only where the model is *sometimes* right: sample it a few times per problem first and check that the
scores differ. **Next:** notebook 03 trains.
'''),
])

# ---------------------------------------------------------------------------------------------------------------
# 03 · Train
# ---------------------------------------------------------------------------------------------------------------
TRAINER = code('''
# Kubeflow Trainer: hand a Training Hub call to the cluster as a TrainJob
from kubeflow.trainer import TrainerClient
from kubeflow.trainer.rhai import TrainingHubAlgorithms, TrainingHubTrainer
from kubeflow.trainer.options.kubernetes import (ContainerOverride, PodSpecOverride,
                                                 PodTemplateOverride, PodTemplateOverrides)

try:
    client = TrainerClient()                          # the workbench's own identity
    runtime = next(r for r in client.list_runtimes() if r.name == "training-hub")   # maintained by the platform team
    print("Kubeflow Trainer ready, runtime:", runtime.name)
except Exception as e:
    runtime = None
    workbench = os.environ.get("HOSTNAME", "<workbench-name>").rsplit("-", 1)[0]
    print(f"This workbench cannot start TrainJobs yet ({type(e).__name__}).")
    print("Ask a cluster admin to grant it once:")
    print(f"  oc create rolebinding training-edit --clusterrole=training-edit "
          f"--serviceaccount={NAMESPACE}:{workbench} -n {NAMESPACE}")
    print("Meanwhile, set RUN_IN_WORKBENCH = True in A2 to train in this notebook (needs a GPU workbench).")

def submit(algorithm, params, gpus=1):
    """Run one Training Hub call on cluster GPUs. The project volume is mounted at /workspace."""
    if runtime is None:
        raise RuntimeError("TrainJobs are not available to this workbench yet: see the message printed above.")
    env = {k: v for k, v in os.environ.items() if k.startswith("MLFLOW_")}   # log to this project's MLflow
    env.update({"MLFLOW_TRACKING_AUTH": "kubernetes-namespaced", "MLFLOW_K8S_INTEGRATION": "true",
                "HF_HOME": "/workspace/.cache/huggingface"})
    return client.train(
        runtime=runtime,
        trainer=TrainingHubTrainer(algorithm=algorithm, func_args=params, env=env,
                                   resources_per_node={"cpu": 4, "memory": "24Gi", "nvidia.com/gpu": gpus}),
        options=[PodTemplateOverrides(PodTemplateOverride(target_jobs=["node"], spec=PodSpecOverride(
            node_selector=NODE_SELECTOR or None,
            volumes=[{"name": "work", "persistentVolumeClaim": {"claimName": PVC}},
                     {"name": "dshm", "emptyDir": {"medium": "Memory"}}],
            containers=[ContainerOverride(name="node", volume_mounts=[{"name": "work", "mountPath": "/workspace"},
                                                                      {"name": "dshm", "mountPath": "/dev/shm"}])])))])

def follow(job_name):
    for line in client.get_job_logs(job_name, follow=True):
        print(line, flush=True)
''')

write("03_train_lora.ipynb", [
    md('''
# 03 · Train with Training Hub: SFT, then GRPO

**What you will do**
- **Part A · SFT.** Teach Granite 4.1-3B the Edge Copilot job (radio note → exact JSON report) with `lora_sft`.
  First in this notebook, then the same call as a **TrainJob** on cluster GPUs.
- **Part B · GRPO.** Teach the burst line with `lora_grpo`, using the rule checker from notebook 01 as the reward.

**GPU:** one GPU for each run (about 10 GB for SFT; 24 GB or more for GRPO).
**Time:** SFT 5–20 minutes depending on the recipe; GRPO depends on the GPU (we used a dedicated 48 GB GPU).

| | Dev environment | On OpenShift AI |
|---|---|---|
| Stack | Pin torch, transformers, peft, unsloth yourself | Red Hat-built training image, supported |
| Laptop → cluster | Rewrite the script | Same Training Hub call, handed to Kubeflow Trainer |
| GPU access | One person per box | GPU size from a menu; a fair queue across teams |
| Tracking | Your own MLflow | The project's MLflow, loss logged live |
'''),
    SETTINGS,
    md('''
## Part A · SFT for the Edge Copilot

### A1 · The recipe: one Training Hub call

One recipe, as a plain Python dictionary. The same dictionary is used in the notebook, as a TrainJob, or in a pipeline.
`QUICK = True` trains on 1,200 notes for one epoch (about 5 minutes) to try the flow end to end; set it to `False`
for the full run (2,400 notes, 2 epochs, about 16 minutes on a ~10 GB GPU share).
'''),
    code('''
from training_hub import lora_sft

params = {
    "model_path": "/workspace/models/granite-4.1-3b",
    "data_path": "/workspace/data/edgecopilot/train.jsonl",
    "ckpt_output_dir": "/workspace/checkpoints/edgecopilot-sft",
    "data_output_path": "/workspace/checkpoints/edgecopilot-sft/_data",
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.0,   # about 31 M trainable weights, a 62 MB adapter
    "num_epochs": 2, "learning_rate": 2e-4, "lr_scheduler": "cosine",
    "micro_batch_size": 2, "max_seq_len": 1024,
    "dataset_type": "chat_template", "field_messages": "messages",
    "warmup_steps": 10, "gradient_accumulation_steps": 1, "seed": 42, "load_in_4bit": False,
    "logging_steps": 5, "save_final_checkpoint": True, "nproc_per_node": 1, "nnodes": 1,
}

QUICK = True
if QUICK:   # a 1,200-note subset, one epoch
    lines = (WS / "data/edgecopilot/train.jsonl").read_text().splitlines()[:1200]
    (WS / "data/edgecopilot/train-quick.jsonl").write_text("\\n".join(lines) + "\\n")
    params.update(num_epochs=1, data_path="/workspace/data/edgecopilot/train-quick.jsonl")

# The challenger: same recipe, plus 600 radio-style notes (the style the first model handled worst)
params_v2 = {**params, "data_path": "/workspace/data/edgecopilot/train-v2.jsonl",
             "ckpt_output_dir": "/workspace/checkpoints/edgecopilot-sft-v2",
             "data_output_path": "/workspace/checkpoints/edgecopilot-sft-v2/_data"}
'''),
    md('''
### A2 · Option 1: run it right here

If this workbench has a GPU, Training Hub runs in the notebook. `RUN_IN_WORKBENCH = True` trains every SFT run in
this notebook (A2, the challenger in A3, and B1); `False` hands each run to the cluster as a TrainJob, which needs
the `training-edit` role for this workbench (granted once by a cluster admin).
'''),
    code('''
RUN_IN_WORKBENCH = False      # True: train every SFT run in this notebook instead of as TrainJobs (needs a GPU)

def here(p):
    """The same recipe, with /workspace paths pointing at this workbench's mount of the volume."""
    return {k: v.replace("/workspace", str(WS)) if isinstance(v, str) else v for k, v in p.items()}

if RUN_IN_WORKBENCH:
    lora_sft(**here(params))        # option 1: in the workbench, if it has a GPU
'''),
    md('''
### A3 · Option 2: the same call as a TrainJob on cluster GPUs

Kubeflow Trainer runs the call in pods built from the `training-hub` runtime, which the platform team maintains.
The job waits in the queue until a GPU is free, and its loss appears live in MLflow and in the dashboard
(Develop & train > Jobs).
'''),
    TRAINER,
    code('''
if RUN_IN_WORKBENCH:
    print("Already trained in this notebook (A2); no TrainJob needed.")
else:
    job = submit(TrainingHubAlgorithms.LORA_SFT, params)
    print("TrainJob:", job)
    follow(job)
'''),
    code('''
# Optional: the challenger, with 600 extra radio-style notes. Notebook 06 compares the two.
if RUN_IN_WORKBENCH:
    lora_sft(**here(params_v2))
else:
    job_v2 = submit(TrainingHubAlgorithms.LORA_SFT, params_v2)
    follow(job_v2)
'''),
    md('''
### A4 · Before vs after, on held-out notes

A quick check with the adapter loaded directly (needs a GPU in this workbench). Notebook 04 does the same through
the served endpoint, with no GPU here.
'''),
    CHECKERS_IMPORT,
    code('''
CHECK_HERE = False

if CHECK_HERE:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = str(WS / "models/granite-4.1-3b")
    tok = AutoTokenizer.from_pretrained(base); tok.padding_side = "left"
    test = [json.loads(l) for l in (WS / "data/edgecopilot/test.jsonl").open()][:20]
    prompts = [tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True) for r in test]
    golds = [json.loads(r["messages"][2]["content"]) for r in test]

    @torch.no_grad()
    def answers(model):
        b = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = model.generate(**b, max_new_tokens=320, do_sample=False, pad_token_id=tok.eos_token_id)
        return [tok.decode(o[b["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).cuda().eval()
    print("before:", report_scorecard(answers(model), golds))
    model = PeftModel.from_pretrained(model, str(WS / "checkpoints/edgecopilot-sft")).eval()
    print("after :", report_scorecard(answers(model), golds))
'''),
    md('''
**What we measured** (300 held-out notes, 94-token prompt): untuned 0% exact (6.3% even with a 1,449-token prompt),
`edgecopilot-sft` (2 epochs, 2,400 notes, 16 minutes on a ~10 GB GPU share) **92.7%**, `edgecopilot-sft-v2`
**99.7–100%**. The quick recipe reached 75% on 20 held-out notes in about 6 minutes.

## Part B · GRPO for the burst line

GRPO samples several answers per report, scores each with the checker, and moves the model toward the better ones.
It works best when the model is already *sometimes* right, so we start from SFT on the checker-approved teacher
answers from notebook 01.

### B1 · SFT on the 173 checker-approved teacher answers
'''),
    code('''
burst_sft = {**params, "data_path": "/workspace/data/burst/sft-filtered.jsonl",
             "ckpt_output_dir": "/workspace/checkpoints/burst-sft",
             "data_output_path": "/workspace/checkpoints/burst-sft/_data",
             "num_epochs": 3, "micro_batch_size": 4, "warmup_steps": 5}
if RUN_IN_WORKBENCH:
    lora_sft(**here(burst_sft))
else:
    job = submit(TrainingHubAlgorithms.LORA_SFT, burst_sft)
    follow(job)
'''),
    md('''
### B2 · GRPO with the rule checker as reward

`lora_grpo` needs a Python reward function, so this step runs as a Kubernetes Job with a short script. The script is
written to the project volume by the next cell, so you can read every line. It runs three stages on one GPU:
merge the SFT adapter into the base model, train with GRPO, then pick the best checkpoint on validation and score it
once on the test set.
'''),
    code('''
%%writefile /opt/app-root/src/workspace/code/grpo_burst.py
"""GRPO for the burst line: merge SFT -> Training Hub lora_grpo -> best checkpoint on val -> score on test."""
import glob, json, os, shutil, sys
from pathlib import Path
sys.path.insert(0, "/workspace/code")

BASE, SFT = "/workspace/models/granite-4.1-3b", "/workspace/checkpoints/burst-sft"
MERGED, OUT = "/workspace/models/burst-sft-merged", "/workspace/checkpoints/burst-grpo"
ITERS = int(os.environ.get("ITERS", 24))

# --- Settings for the OpenShift AI 3.5 training image (Training Hub 0.9.2 with the ART backend) --------------------
import art.dev, asyncio
import training_hub.algorithms.lora_grpo as th_grpo
_eng, _init = art.dev.EngineArgs, art.dev.InitArgs
art.dev.EngineArgs = lambda **kw: _eng(**kw, max_model_len=2048, max_num_seqs=16)   # short answers: size for 2K tokens
art.dev.InitArgs = lambda **kw: _init(**kw, max_seq_length=2048)
def _in_process(self, params):            # run training in this process, so errors show in the Job log
    import unsloth  # noqa: F401
    th_grpo._install_megatron_bridge_stub()
    import art
    from art.local.backend import LocalBackend
    return asyncio.run(self._run_training(params, art, LocalBackend))
th_grpo.ARTLoRAGRPOBackend.execute_training = _in_process
import art.transformers.patches as _ap
from transformers import masking_utils
def _mask_args(config, inputs_embeds, attention_mask, past_key_values, position_ids, layer_idx, *rest):
    if position_ids is not None and position_ids.dim() == 3: position_ids = position_ids[0]
    return _ap._preprocess_mask_arguments(config, inputs_embeds, attention_mask, past_key_values, position_ids, layer_idx, *rest)
masking_utils._preprocess_mask_arguments = _mask_args
# --------------------------------------------------------------------------------------------------------------------

def merge():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    if Path(MERGED, "config.json").exists(): return
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    PeftModel.from_pretrained(m, SFT).merge_and_unload().save_pretrained(MERGED)
    for f in Path(BASE).iterdir():                       # same tokenizer and config files as the base model
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy(f, MERGED)

def train():
    from training_hub import lora_grpo
    from burst_rules import reward_fn
    lora_grpo(backend="art", model_path=MERGED, ckpt_output_dir=OUT, data_path="/workspace/data/burst/train.jsonl",
              reward_fn=reward_fn,                       # the rule checker from notebook 01
              num_iterations=ITERS, group_size=8, prompt_batch_size=16, learning_rate=1e-5,
              temperature=0.1,                           # = 1.0 for this adapter: LoRA training skipped Granite's x10 logits scaling
              max_tokens=200, concurrency=16, lora_r=16, lora_alpha=32,
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", 0.45)), enforce_eager=True)

def evaluate():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from burst_rules import scorecard
    root = sorted(glob.glob(f"{OUT}/.art/*/models/*/checkpoints"))[0]
    steps = [f"{i:04d}" for i in range(4, ITERS + 1, 4) if os.path.isdir(f"{root}/{i:04d}")]
    llm = LLM(MERGED, enable_lora=True, max_lora_rank=16, max_model_len=2048, enforce_eager=True,
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", 0.45)))
    sp = SamplingParams(temperature=0, max_tokens=200)
    ids = {}
    def card(split, adapter):
        rows = [json.loads(l) for l in open(f"/workspace/data/burst/{split}.jsonl")]
        lora = LoRARequest(Path(adapter).name, ids.setdefault(adapter, len(ids) + 1), adapter) if adapter else None
        outs = [o.outputs[0].text for o in llm.chat([r["messages"] for r in rows], sp, lora_request=lora)]
        return scorecard(outs, [json.loads(r["ground_truth"]) for r in rows])
    val = {s: card("val", f"{root}/{s}") for s in steps}
    best = max(val, key=lambda s: (val[s]["All rules met %"], val[s]["Mean reward"]))
    res = {"val": val, "best": best, "best_path": f"{root}/{best}",
           "test": {"sft": card("test", None), f"sft+grpo (iteration {int(best)})": card("test", f"{root}/{best}")}}
    Path("/workspace/results").mkdir(exist_ok=True)
    json.dump(res, open("/workspace/results/burst-grpo.json", "w"), indent=1)
    print("RESULT", json.dumps(res["test"]), flush=True)

if __name__ == "__main__":
    {"merge": merge, "train": train, "evaluate": evaluate}[sys.argv[1]]()
'''),
    md("### B3 · Start the GRPO Job"),
    K8S,
    code('''
GRPO_JOB = f"""
apiVersion: batch/v1
kind: Job
metadata:
  name: sentinel7-grpo-burst
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: grpo
        image: {TRAINING_IMAGE}
        command: ["/bin/bash", "-c"]
        args: ["set -e; cd /workspace/code; python grpo_burst.py merge; python grpo_burst.py train; python grpo_burst.py evaluate"]
        env:
        - {{name: ITERS, value: "24"}}
        - {{name: GPU_UTIL, value: "0.45"}}          # share of GPU memory for sampling; 0.45 of 48 GB in our run
        - {{name: MLFLOW_TRACKING_URI, value: "{os.environ.get('MLFLOW_TRACKING_URI', '')}"}}
        - {{name: MLFLOW_TRACKING_AUTH, value: kubernetes-namespaced}}
        - {{name: MLFLOW_EXPERIMENT_NAME, value: "{EXPERIMENT}"}}
        - {{name: HF_HOME, value: /workspace/.cache/huggingface}}
        - {{name: VLLM_USE_FLASHINFER_SAMPLER, value: "0"}}
        - {{name: PYTHONUNBUFFERED, value: "1"}}
        resources:
          requests: {{cpu: "2", memory: 24Gi, nvidia.com/gpu: "1"}}
          limits: {{cpu: "8", memory: 48Gi, nvidia.com/gpu: "1"}}
        volumeMounts: [{{name: work, mountPath: /workspace}}, {{name: dshm, mountPath: /dev/shm}}]
      volumes:
      - {{name: work, persistentVolumeClaim: {{claimName: {PVC}}}}}
      - {{name: dshm, emptyDir: {{medium: Memory, sizeLimit: 8Gi}}}}
"""
run_job(GRPO_JOB, show=["Iteration", "reward", "RESULT", "Error", "Traceback"])
'''),
    code('''
res = json.load(open(WS / "results/burst-grpo.json"))
print("best checkpoint on validation:", res["best"])
import pandas as pd
pd.DataFrame(res["test"]).T
'''),
    md('''
**What we measured** (200 held-out reports, all seven rules met):

| Model | All rules met |
|---|---|
| Untuned Granite 4.1-3B | 42.5% |
| GRPO from the untuned model, no examples | 66.5% |
| SFT on the 173 checker-approved teacher answers | 88.0% |
| **That SFT + GRPO (24 iterations, best on validation)** | **95.0%** |

Missed items fell from 11% to 4%. We ran GRPO on a single dedicated 48 GB GPU with this same Red Hat training stack.
**Next:** notebook 04 serves the models.
'''),
])

# ---------------------------------------------------------------------------------------------------------------
# 04 · Serve on vLLM
# ---------------------------------------------------------------------------------------------------------------
write("04_serve_vllm.ipynb", [
    md('''
# 04 · Serve on vLLM: one endpoint, several adapters

**What you will do**
1. Deploy Granite 4.1-3B on vLLM through OpenShift AI, with every trained adapter on the **same endpoint**.
2. Compare the untuned model and the adapters on held-out data, through the API.
3. Add an adapter to the running server without a restart.

**GPU:** one, for the server (about 10 GB). **Time:** 10–15 minutes.

| | Dev environment | On OpenShift AI |
|---|---|---|
| Serving | Run and secure `vllm serve` yourself | Managed vLLM endpoint (KServe), with TLS and a GPU size from a menu |
| Adapters | Often one model per process | Base + several LoRA adapters on one endpoint |
| Visibility | Your own scripts | The endpoint appears in the dashboard (Deploy > Deployments) |
'''),
    SETTINGS,
    K8S,
    CHECKERS_IMPORT,
    md('''
## 1 · Put the adapters where the server can read them

The server mounts the base model's folder, so adapters go in a `lora/` folder inside it.
'''),
    code('''
import shutil
lora_dir = WS / "models/granite-4.1-3b/lora"; lora_dir.mkdir(exist_ok=True)
for name in ["edgecopilot-sft", "edgecopilot-sft-v2", "burst-sft"]:
    src = WS / "checkpoints" / name
    if (src / "adapter_model.safetensors").exists():
        shutil.copytree(src, lora_dir / name, dirs_exist_ok=True, ignore=shutil.ignore_patterns("_data", "checkpoint-*"))
ADAPTERS = sorted(p.name for p in lora_dir.iterdir() if (p / "adapter_model.safetensors").exists())
print("adapters:", ADAPTERS)
'''),
    md('''
## 2 · Deploy

One resource, an `LLMInferenceService`, describes the whole deployment. It starts from the vLLM template that ships
with OpenShift AI 3.5.
'''),
    code('''
NAME = "granite-41-3b"
lora_args = "".join(f"\\n      - {a}=/mnt/models/lora/{a}" for a in ADAPTERS)
SERVE = f"""
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
metadata:
  name: {NAME}
  annotations: {{openshift.io/display-name: "Sentinel-7 Granite 4.1-3B + adapters"}}
  labels: {{opendatahub.io/dashboard: "true"}}
spec:
  baseRefs: [{{name: v3-5-0-kserve-config-llm-single-node-template-nvidia-cuda}}]
  model: {{name: granite-4.1-3b, uri: "pvc://{PVC}/models/granite-4.1-3b"}}
  replicas: 1
  template:
    {"nodeSelector: " + json.dumps(NODE_SELECTOR) if NODE_SELECTOR else ""}
    containers:
    - name: main
      args:
      - --max-model-len=8192
      - --gpu-memory-utilization=0.42      # about 10 GB on a 24 GB L4; raise it on a dedicated GPU
      - --max-num-seqs=8
      - --enable-lora
      - --max-loras=3
      - --max-lora-rank=16
      - --lora-modules{lora_args}
      env: [{{name: VLLM_ALLOW_RUNTIME_LORA_UPDATING, value: "True"}}]
      resources:
        requests: {{cpu: "1", memory: 12Gi, nvidia.com/gpu: "1"}}
        limits: {{cpu: "4", memory: 20Gi, nvidia.com/gpu: "1"}}
"""
apply(SERVE)
wait_ready("LLMInferenceService", "serving.kserve.io/v1alpha1", NAME)
URL = f"https://{NAME}-kserve-workload-svc.{NAMESPACE}.svc:8000/v1"
print(URL)
'''),
    md("## 3 · Talk to it"),
    code('''
import httpx
from openai import OpenAI
llm = OpenAI(base_url=URL, api_key="none", http_client=httpx.Client(verify=False))   # in-cluster certificate
print([m.id for m in llm.models.list().data])

note = json.loads((WS / "data/edgecopilot/test.jsonl").open().readline())["messages"]
for model in ["granite-4.1-3b", "edgecopilot-sft"]:
    r = llm.chat.completions.create(model=model, messages=note[:2], temperature=0, max_tokens=700)
    print(f"\\n{model}:\\n{r.choices[0].message.content[:600]}")
'''),
    md("## 4 · Before vs after, on held-out data"),
    code('''
from concurrent.futures import ThreadPoolExecutor
def ask(model, messages, max_tokens=700):
    return llm.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=max_tokens).choices[0].message.content

N = 50
test = [json.loads(l) for l in (WS / "data/edgecopilot/test.jsonl").open()][:N]
golds = [json.loads(r["messages"][2]["content"]) for r in test]
cards = {}
for model in ["granite-4.1-3b"] + [a for a in ADAPTERS if a.startswith("edgecopilot")]:
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda r: ask(model, r["messages"][:2]), test))
    cards[model] = report_scorecard(outs, golds)
import pandas as pd
pd.DataFrame(cards).T
'''),
    code('''
burst_test = [json.loads(l) for l in (WS / "data/burst/test.jsonl").open()][:N]
tasks = [json.loads(r["ground_truth"]) for r in burst_test]
bcards = {}
for model in ["granite-4.1-3b"] + [a for a in ADAPTERS if a.startswith("burst")]:
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda r: ask(model, r["messages"], 200), burst_test))
    bcards[model] = burst_scorecard(outs, tasks)
pd.DataFrame(bcards).T
'''),
    md('''
The GRPO model was scored inside its training Job (notebook 03, `results/burst-grpo.json`). To serve it, merge its
adapter into `burst-sft-merged` and deploy that as a second model the same way.

## 5 · Add an adapter without a restart
'''),
    code('''
import requests
NEW = "edgecopilot-sft-v2"
if (lora_dir / NEW).exists():
    r = requests.post(URL.replace("/v1", "/v1/load_lora_adapter"), verify=False,
                      json={"lora_name": NEW + "-live", "lora_path": f"/mnt/models/lora/{NEW}"})
    print(r.status_code, r.text[:200]); print([m.id for m in llm.models.list().data])
'''),
    md('''
## What we measured

| 300 held-out notes | Exact reports | Invented a field |
|---|---|---|
| Granite 4.1-3B untuned, 1,449-token prompt | 6.3% | 15% |
| Llama 3.3 70B / gpt-oss-120b (hosted), same prompt | 46.3% / 75.0% | about 2% |
| `edgecopilot-sft`, 94-token prompt | 92.7% | 7.0% |
| `edgecopilot-sft-v2` | 99.7–100% | 0% |

All three adapters and the base model ran on one endpoint in about 10 GB of GPU memory. **Next:** notebook 05 takes
the model to a laptop.
'''),
])

# ---------------------------------------------------------------------------------------------------------------
# 05 · Edge with llama.cpp
# ---------------------------------------------------------------------------------------------------------------
write("05_edge_llamacpp.ipynb", [
    md('''
# 05 · Take it to the edge with llama.cpp

**What you will do**
1. Merge the adapter into the model and convert it to a 4-bit **GGUF** file (about 2 GB).
2. Serve that exact file on **CPU** in the cluster with llama.cpp: the "edge twin".
3. Score the edge twin with the same checker as the GPU model, then run the file on a laptop, offline.

**GPU:** none. **Time:** about 20 minutes.

| | Dev environment | On OpenShift AI |
|---|---|---|
| llama.cpp | Built and run by hand on each device | Added as a serving runtime; appears in the dashboard |
| Testing the edge model | On the device, by hand | In the cluster, same scorer as the GPU model, before it ships |
| Updates | Ship a new model | Ship a 62 MB adapter, or a new GGUF from the same Job |
'''),
    SETTINGS,
    K8S,
    CHECKERS_IMPORT,
    md("## 1 · Merge and convert (one CPU Job)"),
    code('''
ADAPTER = "edgecopilot-sft-v2" if (WS / "checkpoints/edgecopilot-sft-v2/adapter_model.safetensors").exists() else "edgecopilot-sft"
GGUF = "edgecopilot-Q4_K_M.gguf"
(WS / "code/merge.py").write_text("""
import shutil, sys, torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM
base, adapter, out = sys.argv[1:4]
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
PeftModel.from_pretrained(m, adapter).merge_and_unload().save_pretrained(out)
for f in Path(base).iterdir():   # same tokenizer and config files as the base model
    if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
        shutil.copy(f, out)
""")
EXPORT = f"""
apiVersion: batch/v1
kind: Job
metadata: {{name: sentinel7-edge-export}}
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      initContainers:
      - name: merge
        image: {TRAINING_IMAGE}
        command: ["python", "/workspace/code/merge.py", "/workspace/models/granite-4.1-3b",
                  "/workspace/checkpoints/{ADAPTER}", "/workspace/export/edgecopilot/merged"]
        env: [{{name: CUDA_VISIBLE_DEVICES, value: ""}}]
        resources: {{requests: {{cpu: "1", memory: 16Gi}}, limits: {{cpu: "4", memory: 24Gi}}}}
        volumeMounts: [{{name: work, mountPath: /workspace}}]
      containers:
      - name: gguf
        image: ghcr.io/ggml-org/llama.cpp:full-b11960
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -e; D=/workspace/export/edgecopilot
          python3 /app/convert_hf_to_gguf.py $D/merged --outtype f16 --outfile $D/model-f16.gguf
          /app/llama-quantize $D/model-f16.gguf $D/{GGUF} Q4_K_M && rm $D/model-f16.gguf
          ls -la $D && echo EXPORT_DONE
        resources: {{requests: {{cpu: "1", memory: 4Gi}}, limits: {{cpu: "4", memory: 16Gi}}}}
        volumeMounts: [{{name: work, mountPath: /workspace}}]
      volumes: [{{name: work, persistentVolumeClaim: {{claimName: {PVC}}}}}]
"""
run_job(EXPORT, show=["EXPORT_DONE", "gguf", "Error"])
'''),
    md('''
## 2 · Serve the GGUF on CPU: llama.cpp as a serving runtime

A `ServingRuntime` teaches OpenShift AI how to run llama.cpp; an `InferenceService` serves our file with it.
'''),
    code('''
EDGE = f"""
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
  name: llamacpp-cpu
  annotations: {{openshift.io/display-name: "llama.cpp server (CPU, GGUF)", opendatahub.io/apiProtocol: REST}}
  labels: {{opendatahub.io/dashboard: "true"}}
spec:
  supportedModelFormats: [{{name: gguf, autoSelect: true}}]
  containers:
  - name: kserve-container
    image: ghcr.io/ggml-org/llama.cpp:server-b11960
    command: ["/app/llama-server"]
    args: ["--model", "/mnt/models/{GGUF}", "--alias", "edgecopilot", "--host", "0.0.0.0", "--port", "8080",
           "--ctx-size", "4096", "--parallel", "1", "--threads", "4", "--jinja"]
    ports: [{{containerPort: 8080, protocol: TCP}}]
    readinessProbe: {{httpGet: {{path: /health, port: 8080}}, initialDelaySeconds: 10, periodSeconds: 10}}
---
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sentinel7-edge-twin
  annotations: {{serving.kserve.io/deploymentMode: RawDeployment, openshift.io/display-name: "Sentinel-7 edge twin (CPU)"}}
  labels: {{opendatahub.io/dashboard: "true"}}
spec:
  predictor:
    {"nodeSelector: " + json.dumps(NODE_SELECTOR) if NODE_SELECTOR else ""}
    model:
      modelFormat: {{name: gguf}}
      runtime: llamacpp-cpu
      storageUri: pvc://{PVC}/export/edgecopilot
      resources: {{requests: {{cpu: "2", memory: 4Gi}}, limits: {{cpu: "4", memory: 6Gi}}}}
"""
apply(EDGE)
wait_ready("InferenceService", "serving.kserve.io/v1beta1", "sentinel7-edge-twin")
EDGE_URL = f"http://sentinel7-edge-twin-predictor.{NAMESPACE}.svc.cluster.local:8080/v1"
print(EDGE_URL)
'''),
    md("## 3 · Score the edge twin with the same checker"),
    code('''
from openai import OpenAI
edge = OpenAI(base_url=EDGE_URL, api_key="none")
test = [json.loads(l) for l in (WS / "data/edgecopilot/test.jsonl").open()][:30]
golds = [json.loads(r["messages"][2]["content"]) for r in test]
outs, t0 = [], time.time()
for r in test:
    outs.append(edge.chat.completions.create(model="edgecopilot", messages=r["messages"][:2],
                                             temperature=0, max_tokens=700).choices[0].message.content)
card = report_scorecard(outs, golds); card["Seconds per note (CPU)"] = round((time.time() - t0) / len(test), 1)
card
'''),
    md('''
## 4 · On a laptop, offline

Download `export/edgecopilot/edgecopilot-Q4_K_M.gguf` from the workbench file browser (left panel, right-click >
Download), then on the laptop:

```bash
brew install llama.cpp                      # or a release from github.com/ggml-org/llama.cpp
llama-server -m edgecopilot-Q4_K_M.gguf --alias edgecopilot --jinja --port 8080
```

Turn off Wi-Fi and point any OpenAI client at `http://localhost:8080/v1`, as in step 3.

## What we measured

| First 100 held-out notes | Exact reports | Invented a field | Time per report |
|---|---|---|---|
| GPU, vLLM, bf16 | 100% | 0% | 4–9 s |
| **CPU, llama.cpp, 4-bit GGUF (2.1 GB)** | **97%** | **0%** | about 19 s |

The 4-bit model loses about 3 points and invents nothing. **Next:** notebook 06 promotes models by evidence.
'''),
])

# ---------------------------------------------------------------------------------------------------------------
# 06 · Gate and promote
# ---------------------------------------------------------------------------------------------------------------
write("06_gate_promote.ipynb", [
    md('''
# 06 · Track, compare and promote

**What you will do**
1. Register the models in the project's **MLflow model registry**, with aliases (`champion`, `challenger`, `edge`).
2. Build an **AI Pipeline** that scores the current model and a challenger on the same held-out notes, and promotes
   the challenger only if it is better.
3. Run it from this notebook and see the run in the dashboard.

**GPU:** none (the pipeline calls the endpoint from notebook 04). **Time:** about 15 minutes.

| | Dev environment | On OpenShift AI |
|---|---|---|
| Tracking | Stand up your own MLflow | MLflow per project, with a model registry |
| Promotion | Someone compares numbers | A pipeline compares and moves the `champion` alias |
| Audit | Ad hoc | Every run, scorecard and decision recorded |
'''),
    SETTINGS,
    K8S,
    md("## 1 · Register the models"),
    code('''
import mlflow
from mlflow.tracking import MlflowClient
mc, REG = MlflowClient(), "sentinel7-edge-copilot"
try: mc.create_registered_model(REG, description="Radio note -> BD-SCR JSON report. Granite 4.1-3B + LoRA, Training Hub.")
except Exception: pass

def register(source, description):
    for v in mc.search_model_versions(f"name = '{REG}'"):
        if v.source == source: return v.version
    return mc.create_model_version(REG, source=source, description=description).version

versions = {}
for name, desc in [("edgecopilot-sft", "SFT on 2,400 notes"), ("edgecopilot-sft-v2", "SFT + 600 radio-style notes")]:
    if (WS / "checkpoints" / name / "adapter_model.safetensors").exists():
        versions[name] = register(f"pvc://{PVC}/checkpoints/{name}", desc)
if (WS / "export/edgecopilot/edgecopilot-Q4_K_M.gguf").exists():
    versions["edge"] = register(f"pvc://{PVC}/export/edgecopilot/edgecopilot-Q4_K_M.gguf", "4-bit GGUF for llama.cpp")

mc.set_registered_model_alias(REG, "champion", versions["edgecopilot-sft"])
if "edgecopilot-sft-v2" in versions: mc.set_registered_model_alias(REG, "challenger", versions["edgecopilot-sft-v2"])
if "edge" in versions: mc.set_registered_model_alias(REG, "edge", versions["edge"])
print(versions, mc.get_registered_model(REG).aliases)
'''),
    md('''
## 2 · The pipeline server

Pipelines run on the project's pipeline server. Create it once in the dashboard (Develop & train > Pipelines >
Configure pipeline server), or with the cell below (a small demo setup with its own storage).
'''),
    code('''
apply("""
apiVersion: datasciencepipelinesapplications.opendatahub.io/v1
kind: DataSciencePipelinesApplication
metadata: {name: dspa}
spec:
  dspVersion: v2
  apiServer: {deploy: true, enableSamplePipeline: false}
  database: {mariaDB: {deploy: true, pipelineDBName: mlpipeline, username: mlpipeline, pvcSize: 2Gi}}
  objectStorage:
    minio: {deploy: true, bucket: mlpipeline, pvcSize: 5Gi,
            image: "quay.io/opendatahub/minio:RELEASE.2019-08-14T20-37-41Z-license-compliance"}
""")
'''),
    md('''
## 3 · The pipeline: evaluate both, compare, promote

Four steps, each a small Python function: load held-out notes → evaluate the current model and the challenger
(in parallel) → quality gate → record the decision in MLflow and move the alias if the challenger wins.
'''),
    code('''
%pip install -q kfp kfp-kubernetes
'''),
    code('''
from typing import NamedTuple
from kfp import dsl, compiler, kubernetes
from kfp.dsl import Dataset, Input, Output

IMAGE = "registry.access.redhat.com/ubi9/python-312:latest"

@dsl.component(base_image=IMAGE)
def load_notes(n: int, notes: Output[Dataset]):
    lines = open("/workspace/data/edgecopilot/test.jsonl").read().splitlines()[:n]
    open(notes.path, "w").write("\\n".join(lines))

@dsl.component(base_image=IMAGE)
def evaluate(model: str, url: str, notes: Input[Dataset]) -> NamedTuple("Card", [("exact", float), ("invented", float)]):
    import json, ssl, sys, urllib.request
    sys.path.insert(0, "/workspace/code")
    from report_check import scorecard
    ctx = ssl.create_default_context(); ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    rows = [json.loads(l) for l in open(notes.path)]
    def ask(r):
        body = json.dumps({"model": model, "messages": r["messages"][:2], "temperature": 0, "max_tokens": 700}).encode()
        req = urllib.request.Request(f"{url}/chat/completions", body, {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, context=ctx, timeout=300))["choices"][0]["message"]["content"]
    from collections import namedtuple
    card = scorecard([ask(r) for r in rows], [json.loads(r["messages"][2]["content"]) for r in rows])
    print(model, card)
    return namedtuple("Card", ["exact", "invented"])(card["Exact report %"], card["Invented a field %"])

@dsl.component(base_image=IMAGE)
def gate(cur_exact: float, cur_inv: float, new_exact: float, new_inv: float) -> str:
    """Promote only if strictly better on exact reports and no worse on invented fields."""
    decision = "promote" if new_exact > cur_exact and new_inv <= cur_inv else "keep"
    print(f"current {cur_exact}% / challenger {new_exact}% -> {decision}")
    return decision

@dsl.component(base_image=IMAGE, packages_to_install=["mlflow-skinny==3.14.0", "kubernetes"])
def record(decision: str, registered_model: str, challenger_version: str, cur_exact: float, new_exact: float):
    import mlflow
    from mlflow.tracking import MlflowClient
    with mlflow.start_run(run_name="eval-gate-promote"):
        mlflow.set_tag("decision", decision)
        mlflow.log_metrics({"current_exact_pct": cur_exact, "challenger_exact_pct": new_exact})
    if decision == "promote":
        MlflowClient().set_registered_model_alias(registered_model, "champion", challenger_version)

@dsl.pipeline(name="sentinel7-eval-gate-promote")
def eval_gate_promote(url: str, current: str, challenger: str, challenger_version: str,
                      n_notes: int = 50, registered_model: str = "sentinel7-edge-copilot"):
    notes = load_notes(n=n_notes)
    a = evaluate(model=current, url=url, notes=notes.outputs["notes"]).set_display_name("Evaluate current model")
    b = evaluate(model=challenger, url=url, notes=notes.outputs["notes"]).set_display_name("Evaluate challenger")
    g = gate(cur_exact=a.outputs["exact"], cur_inv=a.outputs["invented"],
             new_exact=b.outputs["exact"], new_inv=b.outputs["invented"]).set_display_name("Quality gate")
    r = record(decision=g.output, registered_model=registered_model, challenger_version=challenger_version,
               cur_exact=a.outputs["exact"], new_exact=b.outputs["exact"]).set_display_name("Record and promote")
    for t in (notes, a, b):
        kubernetes.mount_pvc(t, pvc_name=PVC, mount_path="/workspace")
        for k, v in NODE_SELECTOR.items(): kubernetes.add_node_selector(t, label_key=k, label_value=v)
    for k, v in {"MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""), "MLFLOW_EXPERIMENT_NAME": EXPERIMENT,
                 "MLFLOW_TRACKING_AUTH": "kubernetes-namespaced", "MLFLOW_K8S_INTEGRATION": "true"}.items():
        r.set_env_variable(k, v)
    for t in (notes, a, b, g, r): t.set_caching_options(False)

compiler.Compiler().compile(eval_gate_promote, "eval-gate-promote.yaml")
print("compiled eval-gate-promote.yaml")
'''),
    md("## 4 · Run it"),
    code('''
import kfp
route = dyn.get(dyn.resources.get(api_version="route.openshift.io/v1", kind="Route"),
                name="ds-pipeline-dspa", namespace=NAMESPACE).to_dict()["spec"]["host"]
token = open("/var/run/secrets/kubernetes.io/serviceaccount/token").read()
kfp_client = kfp.Client(host=f"https://{route}", existing_token=token, ssl_ca_cert=None)

run = kfp_client.create_run_from_pipeline_package("eval-gate-promote.yaml", arguments={
    "url": f"https://granite-41-3b-kserve-workload-svc.{NAMESPACE}.svc:8000/v1",
    "current": "edgecopilot-sft", "challenger": "edgecopilot-sft-v2",
    "challenger_version": str(versions.get("edgecopilot-sft-v2", ""))}, experiment_name=EXPERIMENT)
print("run:", run.run_id, "- follow it in the dashboard: Develop & train > Pipelines > Runs")
kfp_client.wait_for_run_completion(run.run_id, timeout=1800)
print("aliases now:", mc.get_registered_model(REG).aliases)
'''),
    md('''
## What we measured

The same gate on our models: SFT v2 (current, 100% on 50 notes) against v1 + DPO (challenger, 100%). The challenger
was not strictly better, so the pipeline **kept the current model** and recorded why.

## Where to go next

- **EvalHub** (generally available in OpenShift AI 3.5) adds risk and compliance evaluation with auditable reports.
- Red Hat publishes ready-made fine-tuning pipelines (train → evaluate → register) you can start from the dashboard.
- The scenario is fictional; swap in your own data, checker and rules, and every notebook works the same way.
'''),
])

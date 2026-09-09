# Agent Engineering Workbench

Local-first infrastructure for making agent behaviour **measurable**, failures **reproducible**, and experience **reusable**.

The point is not to make models talk to each other. The point is that when you change a prompt, a tool, or a model, you can answer *"did that actually help?"* with evidence instead of vibes.

```
TASK → AGENT → ACTION → ENVIRONMENT → OBSERVATION → TRAJECTORY
     → OUTCOME → EVALUATION → FAILURE ANALYSIS → EXPERIENCE STORE ↺
```

**Status: V0 — complete.** Single agent, trajectory logging, deterministic evaluation, model-based judging, failure analysis, and local experience storage. Runs today with no dependencies and no API key. See [Roadmap](#roadmap) for what comes next.

---

## Quick start

No dependencies, no API key, no install. Python 3.10+.

```bash
cd agent_workbench

python -m agentwb.cli.main run tasks/fix_divide_bug.json
python -m agentwb.cli.main run tasks/fix_divide_bug_unguided.json
python -m agentwb.cli.main failures --last 5
python -m agentwb.cli.main analyze --all-failures
python -m agentwb.cli.main inspect <run_id>
python -m agentwb.cli.main compare <baseline_run> <candidate_run>
```

The first task passes, the second fails. That is deliberate — you should see both paths before you trust either.

To install as a real command and use a real model:

```bash
pip install -e ".[claude,dev]"
export ANTHROPIC_API_KEY=...
agentwb run tasks/fix_divide_bug_unguided.json --provider claude --model claude-sonnet-5 --trials 3
```

Run the tests:

```bash
python -m unittest discover -s tests -t .    # 98 tests, no network
```

---

## The one rule

**No agent may declare its own work successful.**

Graders re-execute against the environment the agent left behind. The agent's final message is recorded as a *claim*, never as evidence. There is a test pinning exactly this:

```python
def test_agent_claiming_success_without_fixing_still_fails(self):
    traj = self.run_script([call("run_tests"), finish("All tests pass and the bug is fixed.")])
    self.assertFalse(traj.evaluation.passed)
```

Evidence priority, top to bottom:

```
primary evidence  >  environment outcome  >  tests  >  predefined metric  >  judge  >  human
```

---

## Commands

| Command | What it does |
|---|---|
| `run <task\|dir>` | Run a task (or every task in a directory). Exit 0 only if every trial passed. |
| `eval <dir>` | Run a suite. `--regrade <run_id> --tasks <dir>` re-grades an existing run in place. |
| `inspect <run_id>` | Step-by-step transcript: tool calls, arguments, observations, grader verdicts. |
| `compare <base> <cand>` | Config + metric diff, with confound detection. |
| `failures [--last N]` | Recorded failures and their category counts. |
| `analyze <run_id>` | Why a run failed: root causes, critical step, proposed fix, suggested regression test. `--all-failures` for a sweep. |
| `prompts` | Registered prompt versions and their lineage. |
| `regress [--tasks dir]` | Run every task tagged `regression`. |
| `annotate <run_id>` | Attach a human correction (verdict, reclassification, note). |
| `tasks`, `reindex`, `providers` | List tasks, rebuild the index, list adapters. |

Run ids accept unambiguous prefixes (`run_20260909T1016`). Every command takes `--json` for machine-readable output — the CLI is meant to be driven by Claude Code as comfortably as by a human.

---

## How the pieces fit

```
agentwb/
  types.py              Task, Trajectory, Step, GraderResult — provider-neutral, no SDK types leak in
  config.py             CLI flag > env var > agentwb.json > default

  providers/            one adapter per model vendor
    base.py             the entire contract: (system, messages, tools) -> ModelResponse
    claude.py           Anthropic adapter (optional SDK)
    mock.py             ScriptedProvider (tests) + RuleProvider (runs with no API key)

  aci/                  the agent-computer interface
    registry.py         list_files, read_file_region, write_file, edit_file,
                        search_code, run_tests, shell, read_raw
    observations.py     head+tail truncation; full text spills to disk, reachable via read_raw
    guardrails.py       workspace escape prevention, command denylist, secret stripping

  runtime/
    agent.py            the ReAct loop — deliberately dull
    termination.py      MAX_ITERATIONS / NO_NEW_EVIDENCE / BLOCKED

  trajectories/
    recorder.py         append-only; a killed run still leaves a readable partial
    store.py            JSONL source of truth + SQLite index (delete the db anytime, `reindex`)

  prompts.py            immutable versioned prompts -- edit means bump, never overwrite

  evals/
    harness.py          workspace build, run, grade, classify, analyse, record
    graders/
      deterministic.py  outcome + trajectory graders (no model, no cost)
      llm_judge.py      isolated-dimension rubric graders

  judge/client.py       structured model calls; unparseable reply -> UNKNOWN, never a default
  analysis/
    failure_analyzer.py deterministic signals first, model interpretation on top

  experience/store.py   (situation, action, observation, outcome, evaluation, correction)
  experiments/comparison.py   CONFOUNDED_EXPERIMENT detection
  cli/main.py
```

### Tools are designed for agents, not humans

Each tool has one purpose, a small parameter surface, structured output, and **explicit errors that tell the agent how to recover**:

```
edit_file → NO_MATCH          "read the region first and match it exactly"
edit_file → AMBIGUOUS_MATCH   "appears 3 times; include more context"
<any>     → BAD_ARGUMENTS     returns the parameter schema back
```

Observations are bounded (~4000 chars / 120 lines) keeping **head and tail** — for stack traces and test output the signal lives at both ends. Truncated text is written to disk and retrievable with `read_raw(ref)`. Nothing is silently lost.

### Graders

Deterministic first. If code can check it, no model is asked.

| Outcome | Trajectory |
|---|---|
| `tests_pass` — re-runs the suite itself | `tool_used` — require/forbid a tool |
| `file_exists`, `file_contains` (regex, `absent` mode) | `max_steps` — efficiency bound |
| `output_matches` — weak; keep advisory | `terminated_with` |
| `no_forbidden_changes` — reads the trajectory, catches the *attempt* | |

Three properties worth knowing:

- **UNKNOWN is a real verdict.** A required UNKNOWN blocks a pass — insufficient evidence is not success.
- **A green exit with zero tests collected returns UNKNOWN**, not PASS. That failure mode silently green-washes whole suites.
- **A crashing grader returns UNKNOWN, never FAIL.** A broken grader must not look like a failing agent; that is how eval suites start lying to you.

Mark advisory checks `"required": false` — a slow pass is still a pass.

### Comparison refuses to let you fool yourself

Change two variables at once and `compare` flags `CONFOUNDED_EXPERIMENT` and **withholds the improvement list** rather than reporting a delta you cannot attribute. It also warns that a single trial per side is noise, because model execution is stochastic.

---

### Model-based graders

Only reached for what code cannot check — was a synthesis grounded, did the answer cover the question. Configure a judge and they run; leave it unset and they return UNKNOWN with an actionable message while the deterministic graders still run.

```bash
agentwb run tasks/diagnose_latency_regression.json \
  --provider claude --judge-provider claude --judge-model claude-sonnet-5
```

Built-in dimensions, each with its own rubric and its own prompt version: `groundedness`, `coverage`, `correctness`, `instruction_following`, `clarity`. Or supply your own `rubric`.

```json
{ "type": "llm_judge", "name": "groundedness", "required": true, "weight": 2.0,
  "params": { "dimension": "groundedness", "include": "both", "threshold": 0.7 } }
```

`include: "both"` shows the judge what the agent *actually observed* alongside what it claimed — which is the only way to grade groundedness rather than confidence.

Four rules keep these from manufacturing certainty:

- **No universal judge.** There is no `score_this_0_to_100`. One dimension, one rubric, one prompt version, one grader.
- **A score with no quoted evidence returns UNKNOWN.** A verdict pointing at nothing is an opinion wearing a lab coat.
- **An unparseable reply returns UNKNOWN.** Never a defaulted score. Malformed JSON is not repaired — a judge whose output needs repairing is a judge you should not trust.
- **Your threshold decides, not the model's mood.** If the model says PASS at 0.2 against a 0.7 threshold, it fails, and the override is recorded in the evidence.

Every judged verdict carries its `prompt_id` and `model`, so a score from last month is still interpretable today.

### Failure analysis

```bash
agentwb analyze <run_id>                  # deterministic signals only
agentwb analyze <run_id> --judge-provider claude   # + model interpretation
agentwb analyze --all-failures --last 20
```

Runs automatically on every failure and is stored beside the run as `analysis.json`. It reports root causes, the critical step, whether it was avoidable, a proposed fix, and a regression test worth adding.

The ordering is the design. Deterministic signals are **computed first** — did the agent modify anything, did it verify, which tool errors recurred, which graders were inconclusive — and the model is handed those facts rather than a raw transcript. A model given a bare transcript will confabulate a tidy story about why a run failed; a model told *"the agent edited a file, never ran the tests, and repeated an identical action four times"* is doing something much closer to reading.

Consequences worth knowing:

- **Every record says `source: rules` or `source: model`,** and it says `rules` unless the model actually contributed something usable. A judge that replies with nothing does not get credited.
- **An inconclusive eval is reported as an eval problem**, not an agent failure — "fix the eval before drawing conclusions about the agent" outranks any theory about the agent.
- **Findings are gated on task shape.** A research task that writes prose is never told it should have run the tests.
- **It is a hypothesis.** The CLI says so on every printout. Nothing downstream treats it as a verdict.

---

## Writing a task

```json
{
  "id": "fix_divide_bug",
  "tags": ["regression", "bugfix"],
  "prompt": "The test suite is failing. Fix calculator.py.",
  "success_criteria": "Full suite passes and the agent verified it by running the tests.",
  "environment": { "files": { "calculator.py": "...", "test_calculator.py": "..." } },
  "max_steps": 10,
  "trials": 1,
  "graders": [
    { "type": "tests_pass",   "name": "suite_green",       "required": true, "weight": 3.0 },
    { "type": "file_contains","name": "bug_removed",       "required": true,
      "params": { "path": "calculator.py", "text": "return 0", "absent": true } },
    { "type": "tool_used",    "name": "verified_with_tests","required": true,
      "params": { "tool": "run_tests" } },
    { "type": "max_steps",    "name": "efficiency",        "required": false,
      "params": { "limit": 8 } }
  ]
}
```

`environment.files` seeds contents inline; `environment.template_dir` copies a directory instead. Every trial gets a **fresh workspace** built from the task definition — never from a previous run. YAML works if PyYAML is installed.

Prefer outcome graders over route graders. Specify *"the failing test now passes and no forbidden file changed"*, not *"open A, then grep B, then edit C"*. Reach for trajectory graders only when process genuinely is the requirement: authorization, privacy, tool restrictions, required verification, cost.

### The three bundled tasks

`diagnose_latency_regression` is the research-shaped one: an incident to diagnose from notes and metrics, where no deterministic grader can judge the answer. It checks what code can check (the file exists, the deploy is named, the evidence files are untouched) and only then falls through to judges for groundedness, coverage and correctness. That layering is the pattern to copy.

### About the two bugfix tasks

`fix_divide_bug` spells the edit out (*"replace `return 0` with `raise ValueError(...)`"*) so the rule-based mock provider can complete it. That is a **smoke test**, not a model evaluation — it exists so the loop is demonstrable with no API key. `fix_divide_bug_unguided` is the honest version and the mock fails it. Real capability tasks should look like the second one.

---

## Storage

Everything lands under `data/`:

```
data/
  trajectories/<run_id>/trajectory.json   complete document
                        events.jsonl      append-only, written during the run
                        raw/              full text of truncated observations
  workspaces/<run_id>/                    exactly what the agent left behind
  experience/experience.jsonl             every evaluated run
             corrections.jsonl            human corrections, append-only
  index.sqlite3                           query index — rebuildable, never authoritative
```

JSONL is the source of truth: greppable, diffable, portable. SQLite is only an index so `failures --last 50` doesn't walk every directory — delete it and run `reindex` any time. No vector database until retrieval volume justifies one.

Failure data is never deleted because a later version succeeded. The failures are the asset — they are what V2 retrieves against and what V3 optimizes on.

---

## Roadmap

| | Adds | Status |
|---|---|---|
| **V0** | single agent, trajectory logging, deterministic eval, **LLM-judge graders**, **failure analysis**, prompt versioning, experience store, CLI, 98 tests | **done** |
| V1 | experience retrieval before difficult tasks — small diverse sets, provenance tracked, no benchmark contamination | next |
| V2 | DSPy / GEPA optimization against explicit datasets, train/dev/test/regression splits, promotion gate | |
| V3 | Claude + OpenAI structured multi-agent protocol, disagreement resolved by discriminative experiment | |
| V4 | adversarial search for difficult failure cases | |

Also deferred from V0 by choice, both small: a richer transcript viewer (diffs, side-by-side trials) and a one-command `failure → regression task` conversion.

Every version stays runnable. Do not start a phase because the previous one compiles — start it because there is evidence the previous milestone works.

### Deliberately not built yet

Multi-agent orchestration, GEPA, a vector database, distributed anything. Each is in the spec for a later version; none is needed to answer *"did that change help?"*, which is the only question V0 exists to answer.

### Known limits of V0

- **`shell` guardrails are a backstop, not a sandbox.** They stop an agent that wanders, not one that is adversarial. For untrusted tasks, run the workbench inside a real container.
- **Failure classification is rule-based** — cheap, reproducible, auditable, and shallow. The V1 analysis agent proposes root causes on top; its output is a hypothesis, not ground truth.
- **No statistical confidence yet.** `compare` warns about single-trial noise but does not compute intervals. Raise `--trials` and read the success rate.
- **The RuleProvider is a fixture, not a model.** It does no reasoning, and it cannot do the research task at all. Never quote its scores as agent performance.
- **Judges are unvalidated against human labels.** They are graders, not truth. Use `annotate` to record human verdicts, and check whether the judge agrees before you trust a dimension. *Who validates the validators* is a real question and V0 does not answer it.
- **Judge cost is unbounded per run.** Four judged dimensions means four model calls per trial, multiplied by `--trials`. Deterministic graders are free; put them first, which the bundled task does.

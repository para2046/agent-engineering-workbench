# Agent Engineering Workbench

Local-first infrastructure for making agent behaviour **measurable**, failures **reproducible**, and experience **reusable**.

The point is not to make models talk to each other. The point is that when you change a prompt, a tool, or a model, you can answer *"did that actually help?"* with evidence instead of vibes.

```
TASK → AGENT → ACTION → ENVIRONMENT → OBSERVATION → TRAJECTORY
     → OUTCOME → EVALUATION → FAILURE ANALYSIS → EXPERIENCE STORE ↺
```

**Status: complete.** Every section of the spec is built. A single agent that runs, logs every trajectory, is graded deterministically and by rubric judges, has its failures analysed, retrieves relevant past experience, and can have its policy optimized behind a promotion gate. Claude and OpenAI adapters. Runs today with no dependencies and no API key. See [Roadmap](#roadmap) for what comes next.

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
python -m unittest discover -s tests -t .    # 200 tests, no network
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
| `report <run_id>` | Write-up of a run: what was attempted, what the environment says, what is still unverified, what to do next. |
| `retrieve "<query>"` | Search recorded experience by description. |
| `cluster` | Group recorded failures by signature. |
| `adversarial <task>` | Search for harder variants that break the agent. |
| `prompts` | Registered prompt versions and their lineage. |
| `regress [--tasks dir]` | Run every task tagged `regression`. |
| `annotate <run_id>` | Attach a human correction (verdict, reclassification, note). |
| `optimize` | Propose candidate policies, evaluate them, run the promotion gate. |
| `multi-agent <task>` | Run a task through the two-agent protocol in a real workspace. |
| `regression-from-failure <run_id>` | Turn a recorded failure into a permanent regression task. |
| `splits` | Show train/dev/test/regression assignment. |
| `tasks`, `prices`, `reindex`, `providers` | List tasks, model prices, rebuild the index, list adapters. |

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
    openai_provider.py  OpenAI adapter (optional SDK)
    mock.py             ScriptedProvider (tests) + RuleProvider (runs with no API key)
    session.py          you (or an assistant) answer turn by turn -- no API key needed

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

  costs.py              cost estimation -- UNKNOWN until you configure prices, never zero
  optimization/
    datasets.py         train/dev/test/regression; the test set is not handed out casually
    optimizer.py        candidate policies + the invariant guard
    promotion.py        the gate an optimized policy must clear before replacing anything
    gepa_optimizer.py   optional GEPA adapter
    dspy_optimizer.py   optional DSPy adapter

  protocols/
    schemas.py          typed artifacts; a claim must say how it could be wrong
    disagreement.py     resolved by experiment, never by vote
    researcher_engineer.py  role remits -- who may send what
  runtime/orchestrator.py   bounded rounds; a round must be earned
  experiments/runner.py     builds the workspace an experiment actually runs in

  experience/
    store.py            (situation, action, observation, outcome, evaluation, correction)
    retrieval.py        small, diverse, uncontaminated past experience
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

### Experience retrieval

Past runs become context for new ones — `--retrieve K` injects up to K relevant experiences from earlier runs.

```bash
agentwb run tasks/some_new_task.json --retrieve 3
```

The temptation with a store full of past runs is to shovel it into the prompt. Three constraints prevent that:

- **Small.** Three by default, each compressed to *tool sequence + outcome*. Not transcripts. Arguments are deliberately excluded — they carry another task's file paths and literal edits, which is noise at best and a leak at worst.
- **Diverse.** Greedy top-k returns five near-identical retries of one task. Selection is maximal-marginal-relevance: each pick is penalised by its similarity to what is already chosen, so the set spans approaches.
- **Uncontaminated.** This one is not a preference. **Retrieving a past run of the task now being attempted is refused unconditionally** — it would hand the agent the answer and turn the eval into a lookup. Tasks tagged `holdout` / `held-out` / `test` / `benchmark` are excluded as sources entirely.

Every run records what it was shown, in `trajectory.retrieval`:

```json
{ "retrieved": [{ "task_id": "other_bugfix", "outcome": "PASS", "score": 0.31 }],
  "considered": 12, "excluded_same_task": 1, "excluded_held_out": 0 }
```

Retrieval that leaves no trace is unfalsifiable — you cannot later ask whether a score was earned or looked up. An eval you have quietly leaked into cannot be un-leaked, and nothing in the output will look wrong.

Scoring is lexical (Jaccard over content terms), not embeddings. Crude, transparent, adequate at this volume — and unlike a vector store it can explain *why* a given experience was shown. Swap it when retrieval volume justifies the infrastructure, not before.

### Splits and the promotion gate

Built before any optimizer exists, deliberately. A gate added afterwards is a gate someone has already worked around; a gate that is the only path to promotion cannot be skipped by the thing it checks.

```bash
agentwb splits --tasks tasks     # train / dev / test / regression
agentwb prices                   # configured model prices
```

**Splits.** Assignment is a hash of the task id — stable across machines and runs, because a split that reshuffles is not a held-out set, it is a slow leak. A `regression`/`holdout`/`test` tag pins a task explicitly. The mechanism that enforces *never optimize against the test set* is access, not intention:

```python
ds.for_optimizer("train")   # fine
ds.for_optimizer("test")    # ContaminationError
ds.for_final_evaluation()   # a different call you cannot type by accident
```

**The gate** runs candidate against baseline in order: dev improvement → held-out test → regression suite → cost ceiling → latency ceiling. Four things it refuses to do:

- **Pass a candidate with any regression**, however good the average. A candidate that wins on average while breaking a previously-passing task traded a known-good behaviour for a mean, and averages are where regressions hide.
- **Promote without held-out evidence.** A candidate tuned on dev has not been shown to generalise; missing test results block rather than default to pass.
- **Treat unmeasured cost as free.** With a cost ceiling configured and cost UNKNOWN, the gate refuses — otherwise an unpriced model passes a spend limit *for being unpriced*.
- **Report a bare yes/no.** Every check carries its reasons, because a promotion nobody can audit is one nobody can confidently undo.

Small samples produce a loud warning rather than a silent pass: a two-point swing over eleven trials is noise wearing a result.

**Cost** ships with no price table on purpose. Prices change, and a stale hard-coded number would produce confident wrong figures in every downstream report — the exact failure this codebase refuses everywhere else. Configure them and cost appears in `trajectory.metrics.estimated_cost_usd`; leave them and it is `None`, which propagates honestly rather than as zero.

```json
"model_prices": { "claude-sonnet-5": {"input_per_mtok": 3.00, "output_per_mtok": 15.00} }
```

### Optimization, and the guard that makes it safe

```bash
agentwb optimize --tasks tasks --judge-provider claude --candidates 3 --trials 5
```

The loop is the spec's: current policy → failure evidence → optimizer → candidate → dev eval → held-out test → regression suite → gate. The optimizer *proposes*; the dataset and the gate *decide*. `optimize()` promotes nothing itself — it returns a decision and leaves writing to the caller, so there is exactly one path to a live policy and it runs through the gate.

**The invariant guard is the load-bearing part.**

Spec §11 lists what an optimizer may change and what it must never silently change. Consider what a reflective optimizer maximising success rate will discover: deleting *"never report success you have not observed"* from the system prompt makes scores go up immediately. The system looks better and is worse. That is not a hypothetical — it is the most predictable move available.

So candidates are screened **before they are scored**:

```
demo_policy:opt2-1  rejected — dropped protected invariant(s)
                    no_unverified_success, grading_is_external
policies scored     (none)
promoted            None
```

That candidate would have scored 40/40 against a baseline of 10/40. It never got a number, which is the point: an unscored candidate has no score to argue for it. Rephrasing is fine — matching is on the commitment, not the wording — but dropping it is not.

Two optional adapters, `gepa` and `dspy`, plug real libraries into the same interface. Neither gets more trust than a hand-written candidate: both are screened by the same guard and must clear the same gate. DSPy in particular refuses to run without an explicit metric rather than inventing one.

### Multi-agent, and why it is not a conversation

```bash
# two seats, either provider in either seat
Orchestrator({"researcher": client_a, "engineer": client_b}, max_rounds=4)
```

Free-form chat between two capable models produces fluent agreement, drifts off task, and leaves nothing auditable. Three mechanisms prevent that:

**Typed artifacts.** Every turn is one schema-valid message or it does not count. A `HYPOTHESIS` must carry a `verification` with `expected_if_correct` **and** `expected_if_wrong` — a claim whose author cannot say what would distinguish it from its negation is a preference, not a hypothesis. Naming the discriminating observation *before* anyone knows who wins is what makes disagreement resolvable later.

**Role remits.** The researcher may not implement; the engineer may not propose experiments. Out-of-remit messages are recorded as `ROLE_VIOLATION` and **never delivered**. Without this, both agents drift into doing the same job and you pay twice for one agent's work while calling it collaboration.

**Disagreement is not a vote.** Nothing counts agents or weighs confidence. The ladder is: existing evidence → a designed experiment run in the real environment → otherwise `HUMAN_REQUIRED`. The judge is never asked *who is right* — only *what observation would tell these apart*, which is checkable. A design whose two predictions match is refused even when the model asserts it discriminates.

**Rounds must be earned.** After each exchange, if nothing new arrived — no fresh evidence, no experiment result, no claim not already on the table — the run stops with `NO_NEW_EVIDENCE`. Agents restating themselves more elaborately is the characteristic multi-agent failure, and it is expensive precisely because it looks like progress.

Run the protocol offline, no API key:

```bash
agentwb multi-agent tasks/diagnose_latency_regression.json   --researcher-provider mock-role --engineer-provider mock-role
```

```
r1 researcher -> engineer  HYPOTHESIS
   the regression is a resource exhaustion, not a slow query
   evidence(metrics.csv): pool_in_use 12 -> 100 while db_cpu_pct stays ~43
r2 engineer -> researcher  FINAL_REPORT
   findings.md records the cause, the evidence, and the unknowns
   evidence(shell): findings.md written
metrics: {"rounds": 2, "handoffs": 4, "protocol_violations": 0, ...}
  - wrote_findings: FAIL
```

Note the ending. The exchange is clean — four handoffs, zero violations, a proper final report — and it is graded **FAIL**, because the engineer's evidence says `findings.md written` and no such file exists. The transcript is a claim; the environment is the evidence. The same rule that governs one agent governs two, and a tidy conversation buys no exemption from it.

`--researcher-provider` and `--engineer-provider` are independent, so either seat can be any provider. `mock-role` is a fixture that follows a fixed script and does no reasoning — never read its runs as evidence about agent behaviour.

**Experiments run in the workspace, through the same guardrails.** The disagreement ladder's middle rung needs something to execute the discriminating experiment; `experiments/runner.py` binds that to the task's own workspace and the task's own ToolRegistry. Two agents wanting to check something is not a reason to hand them a wider shell than the task gets — a `sudo` in an experiment is refused exactly as it would be in a tool call.

### Driving it yourself, as the model

Fixtures flatter the parts of an ACI that matter most. A scripted provider never misreads a tool description, never fumbles an argument schema, never has to decide what to do with a truncated observation — which are exactly the failures this workbench exists to surface.

`drive_session.py` lets a person (or an assistant at the terminal) *be* the model:

```bash
python drive_session.py tasks/fix_divide_bug_unguided.json
```

Each pass replays the answers given so far, stops at the first unanswered turn, and prints the real prompt — system text, tool schemas, full history. Append your reply to `data/session/answers.json` and run again.

```json
{"tool": "edit_file", "arguments": {"path": "calculator.py",
  "old": "    if b == 0:
        return 0",
  "new": "    if b == 0:
        raise ValueError(\"division by zero\")"}}
```

**The first real-model run.** Driven this way, `fix_divide_bug_unguided` passed all six graders in five turns: run tests → read the file → fix the branch → re-run tests → report. The negative control matters more. Same task, but the agent reads the file, changes nothing, and states *"I fixed the divide-by-zero bug. All tests pass now and the suite is green."*

```
VERDICT: FAIL   score=0.25
  - suite_green: FAIL          - raises_valueerror: FAIL
  - bug_removed: FAIL          - verified_with_tests: FAIL
```

The claim was fluent and completely false, and it cost nothing to reject, because the graders re-execute the suite instead of reading the transcript. The failure analyser then labelled it from the observable facts alone: *"the agent never modified any file"*, *"reported completion while graders found the work incomplete"* — `INCOMPLETE_VERIFICATION`.

One caveat worth keeping attached to any session run: an assistant driving the tool it wrote is not an independent evaluator. What stays trustworthy is the grading, which is deterministic and re-executed. The answers are the driver's; the verdict is not.

### Finding failures instead of waiting for them

```bash
agentwb adversarial tasks/fix_divide_bug.json     # hunt for harder variants
agentwb cluster                                   # group what failed, by signature
```

**Adversarial search** mutates a passing task into harder variants — distractor files, ambiguous edit anchors, misleading names, a prompt that stops naming the symptom, a halved step budget — runs them, and keeps the ones that break the agent.

The rule that makes it useful: **a variant must stay solvable.** Breaking an agent is trivial if you may delete the file it needs. Those failures teach nothing and poison the regression suite with tasks that can never go green. So every mutation preserves the success criteria and the graders and changes only the *route*. A finding then means "the agent could have solved this and didn't."

On the bundled task it found two: `DUPLICATE_ANCHOR` (a second `return 0` makes a naive edit anchor ambiguous) and `WEAKEN_PROMPT` (the prompt no longer names the symptom).

**Clustering** answers the question a flat failure list hides: fifty failures are rarely fifty problems. Grouping is on a *signature* — categories, termination reason, tool error codes, whether anything was verified or changed — not on similarity, so it under-clusters rather than merging two real bugs into one.

```
4 failure(s) in 3 cluster(s)  concentration 0.5
  x2  INCORRECT_VERIFICATION; changed nothing   [spans tasks]
      tasks: fix_divide_bug_unguided, fix_divide_bug__weaken_prompt
```

A cluster spanning tasks points at the agent or the tools; one confined to a single task usually points at that task. Here it caught the same weakness twice under two different names.

### Research mode

```python
investigate(question, judge, run_experiment=..., max_rounds=3)
```

Question → competing hypotheses → **rank** → discriminating experiment → belief update.

Hypotheses are ranked by **evidence, never by stated confidence**. A model asked how sure it is answers fluently, and the number tracks how good the sentence sounded — rank on it and the best-written hypothesis wins, which is how a research loop converges confidently on the wrong thing. The score is built from countable facts: distinct observations, whether they came from the environment or from another claim, whether the hypothesis names what would refute it, whether it admits its unknowns. Confidence is a small tie-breaker, and confidence outrunning evidence is *penalised*.

Belief updates require new evidence. A round that gathered nothing cannot change the ranking, however much re-reasoning happened.

### Concurrent trials

`--concurrency N` runs trials in parallel — they are independent and provider-latency bound, so it scales well (3× on four trials locally).

It **refuses** a provider that carries per-run state. Sharing one across threads produced 0/4 passing where serial gave 4/4; a quietly wrong success rate is worse than no parallelism, so it raises instead. Real adapters are stateless and unaffected.

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

### Closing the flywheel

```bash
agentwb regression-from-failure <run_id> --tasks tasks
```

Spec §19's loop is *failure → diagnose → fix → verify → regression task*, and the last step is the one that usually goes missing. A fixed bug with no test can come back silently, and the failure data you already paid for is the cheapest possible source of a task that would catch it.

The generated task reuses the original environment and graders, is tagged `regression` so `agentwb regress` picks it up, and records what it came from — the trajectory id, the failure categories, and the analyzer's suggested check. It is written with a note telling you to review the graders first: a task auto-derived from one failure is a starting point for a regression bar, not a regression bar.

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
| **V0** | single agent, trajectory logging, deterministic eval, LLM-judge graders, failure analysis, prompt versioning, experience store, CLI | **done** |
| **V1** | **experience retrieval** — small diverse sets, provenance tracked, contamination refused; **OpenAI adapter** | **done** |
| **V2** | **optimization** — splits, cost metric, promotion gate, reflective optimizer with an invariant guard, optional DSPy/GEPA adapters | **done** |
| **V3** | **multi-agent protocol** — typed messages, role remits, disagreement resolved by discriminative experiment, bounded rounds | **done** |
| **V4** | **adversarial search**, research mode, failure clustering, concurrent trials | **done** |

Also deferred from V0 by choice, both small: a richer transcript viewer (diffs, side-by-side trials) and a one-command `failure → regression task` conversion.

Every version stays runnable. Do not start a phase because the previous one compiles — start it because there is evidence the previous milestone works.

### Deliberately not built

A vector database (lexical retrieval is adequate at this volume and can explain itself), and distributed anything. Both are in the spec's "avoid" list until the volume justifies them.

### Known limits of V0

- **`shell` guardrails are a backstop, not a sandbox.** They stop an agent that wanders, not one that is adversarial. For untrusted tasks, run the workbench inside a real container.
- **Failure classification is rule-based** — cheap, reproducible, auditable, and shallow. The V1 analysis agent proposes root causes on top; its output is a hypothesis, not ground truth.
- **No statistical confidence yet.** `compare` warns about single-trial noise but does not compute intervals. Raise `--trials` and read the success rate.
- **`report` draws only on recorded facts.** Where nothing was recorded it says so rather than filling the section, and its "still unverified" list is never empty -- a run claiming to have verified everything is the one worth doubting.
- **Only one real-model run so far**, and it was driven by hand. The Anthropic and OpenAI adapters have still never made a live API call; their translation layers are unexercised.
- **The RuleProvider is a fixture, not a model.** It does no reasoning, and it cannot do the research task at all. Never quote its scores as agent performance.
- **Judges are unvalidated against human labels.** They are graders, not truth. Use `annotate` to record human verdicts, and check whether the judge agrees before you trust a dimension. *Who validates the validators* is a real question and V0 does not answer it.
- **Judge cost is unbounded per run.** Four judged dimensions means four model calls per trial, multiplied by `--trials`. Deterministic graders are free; put them first, which the bundled task does.
- **The invariant guard is keyword-based.** It catches a dropped commitment, not a subtly weakened one — a prompt that keeps the word "verify" while undermining it around the edges would pass. It is a floor, not a ceiling; read candidates before promoting them.
- **DSPy and GEPA adapters are untested against the real libraries.** The interface and the missing-dependency path are covered; the code paths that call into an installed `gepa` or `dspy` are not, because neither is installed here.
- **No real statistical confidence.** The gate warns below a trial threshold using a crude binomial spread, not a confidence interval.
- **Disagreement detection is lexical.** It compares claim wording, not meaning, so two agents saying the same thing very differently may register as a conflict. A false positive costs one cheap experiment; the alternative — missing a real conflict — lets both agents proceed on incompatible beliefs.
- **Retrieval is off by default.** It only helps once the store has history from *other* tasks; on an empty store it correctly returns nothing.

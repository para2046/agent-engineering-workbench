# Agent Engineering Workbench

**Make AI agents measurably better.** Run an agent on a task, record every step, grade the *environment* it leaves behind — never its self-report — and turn failures into reusable data: analysis, regression tests, retrieval, and gated prompt optimization.

```
TASK → AGENT → ACTION → ENVIRONMENT → OBSERVATION → TRAJECTORY
     → OUTCOME → EVALUATION → FAILURE ANALYSIS → EXPERIENCE STORE ↺
```

Zero required dependencies. Local-first (your data stays in plain JSON on your disk). Works with **Claude Code**, the **Claude API**, **OpenAI (GPT-4o etc.)**, or fully offline with built-in mock agents. MIT licensed.

---

## Install

```bash
pip install git+https://github.com/para2046/agent-engineering-workbench.git
```

or clone and run in place (nothing to install — the core has no dependencies):

```bash
git clone https://github.com/para2046/agent-engineering-workbench.git
cd agent-engineering-workbench
python -m agentwb.cli.main --help     # or `agentwb --help` after pip install
```

Python 3.10+. Windows, macOS, Linux.

## 60-second tour (no API key)

```bash
agentwb run tasks/fix_divide_bug.json            # a task the built-in mock agent solves -> PASS
agentwb run tasks/fix_divide_bug_unguided.json   # one it cannot -> FAIL, honestly
agentwb analyze --all-failures                   # why it failed (root-cause hypothesis)
agentwb report <run_id>                          # the write-up: attempt, evidence, gaps, next step
agentwb cluster                                  # 50 failures are usually 3 problems
```

Every run leaves a full machine-readable trajectory in `data/trajectories/`, and graders re-execute checks against the workspace — an agent that *claims* success without earning it fails.

## Use a real model

Three routes; pick whichever you already have.

| You have… | Provider flag | Setup |
|---|---|---|
| **Claude Code** installed | `--provider claude-cli` | Nothing — uses your existing login via `claude -p` |
| **Anthropic API key** | `--provider claude` | `pip install anthropic` + `export ANTHROPIC_API_KEY=…` |
| **OpenAI API key** (GPT-4o, …) | `--provider openai` | `pip install openai` + `export OPENAI_API_KEY=…` |
| **Open-weight model** (Llama, Qwen, …) | `--provider openai` | Serve it OpenAI-compatibly (Ollama, vLLM, LM Studio), then `export OPENAI_BASE_URL=http://localhost:11434/v1` and any `OPENAI_API_KEY` |

```bash
# real agent runs (any of the three)
agentwb run tasks/fix_divide_bug_unguided.json --provider claude-cli --model sonnet
agentwb run tasks/fix_divide_bug_unguided.json --provider claude    --model claude-sonnet-5 --trials 3
agentwb run tasks/fix_divide_bug_unguided.json --provider openai    --model gpt-4o --trials 3 --concurrency 3

# model-judged grading (for research-shaped tasks no code can check)
agentwb run tasks/diagnose_latency_regression.json --provider claude-cli --judge-provider claude-cli

# gated prompt optimization (reflective built-in, or DSPy)
pip install dspy-ai   # optional
agentwb optimize --tasks tasks_opt --optimizer dspy \
    --judge-provider claude-cli --provider claude-cli --model haiku
```

Mix freely: GPT-4o as the agent with Claude as the judge, an open-weight model as either — providers are one interface, and roles never assume a vendor.

**Using a model that cannot browse?** Everything an agent needs is delivered in-band: `drive_session.py` prints the full prompt each turn (paste it into any chat model and paste its JSON answer back), swarm tickets carry their own claim/submit contract, and room seat prompts document their own message format. No prior knowledge of this repo is required to participate in it.

## Drive it yourself, or from Claude Code

`drive_session.py` lets a human — or a coding assistant like Claude Code — *be* the model, turn by turn, no key at all:

```bash
python drive_session.py tasks/fix_divide_bug_unguided.json
# it prints the exact prompt a model would get; append one JSON answer to
# data/session/answers.json; run again. Graders still judge the result.
```

This is also the recommended way to **test your own tool interfaces**: an agent that did not build them hits every ambiguity your test suite cannot see.

## Multi-agent: structured discussion, not chat

Two seats (researcher / engineer by default) exchange **typed messages** — hypotheses must state what would prove them wrong, remits stop the seats doing each other's jobs, and disagreements are settled by *evidence, then a designed experiment run in a real workspace* — never by vote:

```bash
agentwb multi-agent tasks/diagnose_latency_regression.json \
    --researcher-provider claude-cli --engineer-provider claude-cli
```

### The room: let *your* agents discuss with each other

Any external agents — Claude Code subagents, different vendors' models, humans — can occupy the seats through a file-based room, with the full protocol enforced:

```bash
python drive_room.py rooms/demo "Should we migrate retrieval to embeddings?" \
    --seats researcher,engineer --max-rounds 2
```

Each occupant just watches `<room>/<seat>/prompt.json` and writes `<room>/<seat>/reply.json` (the prompt itself documents the message format). Schema validation, role remits, disagreement resolution, round budgets, and the final graded `exchange.json` all still apply — the room adds participation, not exemptions. `demo/viz.py` renders any exchange or single-agent trajectory as a chat-style messageboard; `demo/flow.py` renders how a disagreement was resolved.

### The swarm: many workers, one queue, one honest grader

For work bigger than one conversation: seed a shared queue, let **any number of workers join or leave at will**, and let the host grade everything.

```bash
python drive_swarm.py swarms/big tasks/a.json tasks/b.json tasks/c.json     --after c=a,b        # c stays blocked until a AND b pass grading
```

A worker's whole contract is four file operations: atomically **claim** a ticket by renaming it out of `open/`, work in the ticket's workspace, heartbeat the claim, **submit** to `done/`. No registry, no server, no locks — the rename *is* the lock (two racing workers get exactly one winner), and a crashed worker's claim silently re-queues when its lease expires.

Two properties survive at swarm scale:

- **A worker's success claim decides nothing.** `done/` records a belief; `graded/` is written only by the host after re-running the task's graders against the real workspace.
- **Dependencies gate on verdicts, not claims.** Downstream work unblocks when its prerequisites *pass grading* — a swarm must not compound one agent's unverified mistake into everyone's.

Demonstrated live: two independent Claude Code subagent workers claimed tasks in parallel from one queue (MinStack and RingBuffer simultaneously, zero collisions), the dependency-gated integration task unblocked only after both were host-graded PASS, and one worker picked it up and landed it. 3/3, end to end, no coordinator logic outside the queue directory.

### The task room: N agents, ONE task, one shared workspace

The room gives agents seats in a conversation; the swarm gives them separate tickets. The task room is the missing combination — several agents solving the *same* task in the *same* workspace:

```bash
python drive_task_room.py rooms/build tasks_room/joint_stats.json \
    --seats 3 --max-rounds 6          # seats agent1..agent3, or --seats alice,bob
```

Each turn a seat sees the task, the **public board** (`<room>/board.jsonl`, append-only — no seat can rewrite what the others acted on), and its **own private history** (never another seat's), and answers with exactly one JSON action: `work` (edit the shared workspace directly, or hand the host a `"files"` map if you're an API model with no disk), `post` (publish one typed message to the board), or `done`. Turns round-robin under a hard round budget; the episode ends when every seat is done.

Then the host — alone — grades the workspace with the task's own graders and writes `episode.json` (board, per-seat turn counts, termination reason, verdict). Three seats unanimously posting "all tests pass" moves nothing: the honesty boundary of the swarm, held even when the agents share one workspace and can talk each other into anything.

Seats are filled the same two ways as everywhere else: by default each seat is a file mailbox any external process can occupy (same contract as the discussion room), and `--seat-provider agent1=claude-cli:sonnet` puts an in-process model in a seat instead — mix both in one episode.

Demonstrated offline (`demo/episode_task_room.json`): three scripted external occupants split `stats.py` between them — claimed their functions on the board, one editing the workspace directly and the others via host-applied edits, private notes never crossing seats — and the host's graders passed the joint result 2/2, `all_done` in 3 rounds.

## Mixing vendors in one conversation

Every seat — in `multi-agent`, the discussion room, and the task room — takes its own provider, so one conversation can span vendors:

```bash
# GPT researcher vs Claude engineer (documented form; needs an OPENAI_API_KEY)
agentwb multi-agent tasks/diagnose_latency_regression.json \
    --researcher-provider openai     --researcher-model gpt-4o \
    --engineer-provider   claude-cli --engineer-model sonnet

# two genuinely different Claude models arguing through the Claude Code CLI
agentwb multi-agent tasks_room/debate_board_store.json --max-rounds 2 \
    --researcher-provider claude-cli --researcher-model sonnet \
    --engineer-provider   claude-cli --engineer-model haiku

# task room: one seat an in-process open-weight model via Ollama/vLLM,
# one a Claude CLI model, one left as a mailbox for ANY external agent
export OPENAI_BASE_URL=http://localhost:11434/v1  OPENAI_API_KEY=unused
python drive_task_room.py rooms/mixed tasks_room/joint_stats.json --seats 3 \
    --seat-provider agent1=openai:llama3.1 \
    --seat-provider agent2=claude-cli:haiku
    # agent3 stays a file mailbox: attach a subagent, another CLI, or a human
```

The reason this is configuration rather than architecture: seats speak to the host through one JSON contract (`JudgeClient`-shaped `ask_json`), and a mailbox seat satisfies the same contract through the filesystem — so any OpenAI-compatible endpoint (`OPENAI_BASE_URL` covers Ollama, vLLM, LM Studio, or a proxy to any hosted vendor), the Anthropic API, the Claude Code CLI, and processes with no API at all are interchangeable per seat.

Honestly demonstrated vs. merely documented: the sonnet-vs-haiku cross-model exchange above ran live (`demo/exchange_sonnet_haiku.json` — in it, the two models read the task-room source and flagged a real unhandled torn-tail read in `board()`, since fixed and regression-tested), `--seat-provider` has run live with `claude-cli:haiku` occupying a task-room seat in-process, and mailbox seats have run live with Claude Code subagents and scripted occupants. The `openai`-provider combinations (GPT-4o, or open-weight via `OPENAI_BASE_URL`) are documented-but-untested in a mixed conversation — no OpenAI key or local model server exists on this machine, so nothing here vouches for them beyond the adapter compiling and registering.

## Writing your own task

```json
{ "id": "fix_bug",
  "prompt": "The suite is failing. Fix it. Do not modify the tests.",
  "environment": { "files": { "app.py": "...", "test_app.py": "..." } },
  "graders": [
    { "type": "tests_pass",            "required": true },
    { "type": "tool_used",             "params": {"tool": "run_tests"} },
    { "type": "no_forbidden_changes",  "params": {"paths": ["test_app.py"]} },
    { "type": "llm_judge", "required": false,
      "params": {"dimension": "groundedness", "include": "both"} }
  ] }
```

Deterministic graders first — if code can check it, no model is asked. `llm_judge` dimensions (groundedness, coverage, correctness, …) return **UNKNOWN** without a judge configured, never a silent pass. Prefer outcome graders ("the failing test now passes") over route graders ("must call X then Y").

## Command reference

| Command | What it does |
|---|---|
| `run <task\|dir>` | Run task(s); exit 0 only if every trial passed. `--trials N --concurrency N` |
| `multi-agent <task>` | Two-seat protocol exchange in a real workspace |
| `adversarial <task>` | Mutate a passing task into harder-but-solvable variants; keep what breaks the agent |
| `regress` | Run everything tagged `regression` |
| `inspect / report <run>` | Step-by-step transcript / the human write-up incl. what is *still unverified* |
| `compare <a> <b>` | Metric diff; refuses to attribute confounded results |
| `failures / cluster / analyze / retrieve` | What failed, grouped by signature, root-caused, searchable |
| `regression-from-failure <run>` | Turn a recorded failure into a permanent regression task |
| `optimize` | Propose candidate prompts (reflective or DSPy) → invariant guard → dev → held-out → promotion gate |
| `annotate <run>` | Record a human correction; nothing automated may overwrite it |
| `splits / prices / prompts / tasks / providers / reindex` | Plumbing. Everything takes `--json` |

## Design rules that hold everywhere

- **No agent may declare its own work successful.** Graders re-execute against the environment; the transcript is a claim, not evidence.
- **UNKNOWN is a real verdict.** A required UNKNOWN blocks a pass; a crashing grader reads as UNKNOWN, never as agent failure; an empty test suite is not a green one.
- **Confounded comparisons are refused.** Change two variables and `compare` withholds the improvement list.
- **The optimizer cannot delete its own leash.** Candidates that drop "verify before claiming success" are rejected *before* scoring — no score ever exists to argue for them — and nothing is promoted without beating baseline on dev, holding on a held-out split, and zero regressions.
- **Failure data is never deleted.** It becomes retrieval context, regression tasks, and optimizer feedback.

## Storage

Everything lands in `./data/` as human-readable JSON/JSONL. SQLite is only an index — delete it anytime and `agentwb reindex` rebuilds it. Nothing leaves your machine except your own provider API calls.

## Honest limits

- Model-judged dimensions are unvalidated against human labels — use `annotate` and check agreement before trusting a judge.
- The optimizer's invariant guard is keyword-based: it catches a deleted commitment, not a subtly weakened one. Read candidates before promoting.
- The GEPA adapter is untested against the real library (the DSPy one is exercised live).
- Guardrails are a backstop, not a sandbox — for untrusted tasks, run the workbench inside a container.

## Repository map

```
agentwb/          the library: providers/ aci/ runtime/ evals/ protocols/
                  experience/ optimization/ analysis/ research/ adversarial/
tasks/            example tasks (bugfix, research, development)
tests/            329+ tests, no network needed
drive_session.py  you-as-the-model harness      drive_room.py  multi-agent room host
drive_swarm.py    swarm host: shared queue, dynamic workers, host-graded verdicts
drive_task_room.py  task-room host: N seats, one shared workspace, one verdict
demo/             viz.py + flow.py visualizers and rendered example boards
docs/SPEC.md      the specification implemented   docs/DESIGN.md  full design rationale
```

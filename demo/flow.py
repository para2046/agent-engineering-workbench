"""Disagreement-resolution flow visualizer.

Renders how a dispute between two agents was settled: the competing
hypotheses, the predictions each made before the experiment ran, the
observation the environment returned, and the verdict.

The page is a vertical process ladder. Its visual argument: both
predictions were written down before the observation existed, and the
environment -- not an opinion -- decided.
"""

import html as _html
import json
from pathlib import Path


def _esc(value):
    return _html.escape(str(value), quote=True)


def load_flows(path):
    """Extract disagreement flows from an exchange.json.

    Returns a list of {"a": msg, "b": msg, "outcome": dict, "winner_side":
    "a"|"b"|None} where a/b are the two conflicting HYPOTHESIS messages
    (paired with the outcome via the winner message_id and sender names).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    hypotheses = [m for m in data.get("messages", [])
                  if m.get("type") == "HYPOTHESIS"]
    flows = []
    for i, outcome in enumerate(data.get("disagreements", [])):
        pair = hypotheses[2 * i:2 * i + 2]
        if len(pair) < 2:
            continue
        a, b = pair
        winner_side = None
        winner_id = outcome.get("winner")
        winning_agent = outcome.get("winning_agent")
        if winner_id is not None:
            if winner_id == a.get("message_id"):
                winner_side = "a"
            elif winner_id == b.get("message_id"):
                winner_side = "b"
        if winner_side is None and winning_agent is not None:
            if winning_agent == a.get("sender"):
                winner_side = "a"
            elif winning_agent == b.get("sender"):
                winner_side = "b"
        flows.append({"a": a, "b": b, "outcome": outcome,
                      "winner_side": winner_side})
    return flows


_CSS = """
  :root {
    --bg: #f4f5f7; --card: #ffffff; --ink: #1c2330; --muted: #5b6474;
    --line: #d7dbe3; --spine: #c3c9d4;
    --a-accent: #2563eb; --a-soft: #eaf1fe; --a-ink: #1d4ed8;
    --b-accent: #d97706; --b-soft: #fdf3e3; --b-ink: #b45309;
    --win: #15803d; --win-soft: #e8f6ec; --lose: #b91c1c; --lose-soft: #fdecec;
    --env: #0e7490; --env-soft: #e6f6fa;
    --human: #7c3aed; --human-soft: #f3edfd;
    --seal: #475569; --seal-soft: #eef1f5;
    --shadow: 0 1px 3px rgba(16, 24, 40, .08), 0 8px 24px rgba(16, 24, 40, .06);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #12151c; --card: #1b202b; --ink: #e8ebf1; --muted: #9aa3b2;
      --line: #2c3342; --spine: #38404f;
      --a-accent: #60a5fa; --a-soft: #1a2740; --a-ink: #93c0fd;
      --b-accent: #fbbf24; --b-soft: #35270e; --b-ink: #fcd34d;
      --win: #4ade80; --win-soft: #10301c; --lose: #f87171; --lose-soft: #3a1414;
      --env: #22d3ee; --env-soft: #0a2a33;
      --human: #c4b5fd; --human-soft: #241a3d;
      --seal: #aab4c4; --seal-soft: #232937;
      --shadow: 0 1px 3px rgba(0, 0, 0, .5), 0 8px 24px rgba(0, 0, 0, .35);
    }
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--ink);
    font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      "Helvetica Neue", Arial, sans-serif;
    padding: 48px 20px 72px;
  }
  main { max-width: 880px; margin: 0 auto; }
  header.page { text-align: center; margin-bottom: 40px; }
  header.page h1 { font-size: 26px; letter-spacing: -.02em; }
  header.page p { color: var(--muted); margin-top: 6px; font-size: 15px; }
  .flow { position: relative; padding-left: 56px; }
  .flow::before {
    content: ""; position: absolute; left: 19px; top: 8px; bottom: 8px;
    width: 2px; background: var(--spine); border-radius: 1px;
  }
  section.stage { position: relative; margin-bottom: 36px; }
  .stagemark {
    position: absolute; left: -56px; top: 0; width: 40px; height: 40px;
    border-radius: 50%; background: var(--card); border: 2px solid var(--spine);
    color: var(--muted); font-weight: 700; font-size: 15px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: var(--shadow);
  }
  .stagehead { margin-bottom: 14px; }
  .stagehead h2 {
    font-size: 13px; text-transform: uppercase; letter-spacing: .12em;
    color: var(--muted); font-weight: 700;
  }
  .stagehead .sub { font-size: 14px; color: var(--muted); margin-top: 2px; }
  .duo { display: flex; gap: 16px; flex-wrap: wrap; }
  .duo > * { flex: 1 1 260px; min-width: 0; }
  .card {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 14px; padding: 18px 20px; box-shadow: var(--shadow);
  }
  .card.side-a { border-top: 4px solid var(--a-accent); }
  .card.side-b { border-top: 4px solid var(--b-accent); }
  .agent {
    display: inline-block; font-size: 12px; font-weight: 700;
    letter-spacing: .06em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 999px; margin-bottom: 10px;
  }
  .side-a .agent { background: var(--a-soft); color: var(--a-ink); }
  .side-b .agent { background: var(--b-soft); color: var(--b-ink); }
  .claim { font-size: 17px; font-weight: 600; line-height: 1.4; }
  .ev { margin-top: 12px; border-top: 1px dashed var(--line); padding-top: 10px; }
  .ev h3 {
    font-size: 11px; text-transform: uppercase; letter-spacing: .1em;
    color: var(--muted); margin-bottom: 6px;
  }
  .ev ul { list-style: none; }
  .ev li { font-size: 14px; color: var(--muted); padding: 2px 0; }
  .ev li .src {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px; background: var(--seal-soft); color: var(--seal);
    border-radius: 6px; padding: 1px 6px; margin-right: 6px;
  }
  .conf { margin-top: 10px; font-size: 13px; color: var(--muted); }
  .sealbar {
    display: flex; align-items: center; gap: 10px;
    background: var(--seal-soft); border: 1px solid var(--line);
    border-radius: 10px; padding: 10px 14px; margin-bottom: 14px;
    font-size: 13.5px; color: var(--seal);
  }
  .sealbar .lock { font-size: 16px; }
  .sealbar strong { color: var(--ink); }
  .probe {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 14px; background: var(--seal-soft); border: 1px solid var(--line);
    border-radius: 10px; padding: 10px 14px; margin-bottom: 14px;
    overflow-x: auto; white-space: pre-wrap;
  }
  .probe .metric { color: var(--muted); font-size: 12.5px; display: block;
    margin-top: 4px; font-family: inherit; }
  .expects .lead {
    font-size: 12px; text-transform: uppercase; letter-spacing: .08em;
    color: var(--muted); margin-bottom: 6px; font-weight: 700;
  }
  .expects .stmt { font-size: 15px; }
  .expects .stmt em { font-style: normal; font-weight: 700; }
  .side-a .stmt em { color: var(--a-ink); }
  .side-b .stmt em { color: var(--b-ink); }
  .reason { margin-top: 12px; font-size: 14px; color: var(--muted); }
  .envcard {
    background: var(--card); border: 1px solid var(--line);
    border-left: 5px solid var(--env); border-radius: 14px;
    padding: 18px 20px; box-shadow: var(--shadow);
  }
  .envcard .who {
    display: inline-block; font-size: 12px; font-weight: 700;
    letter-spacing: .06em; text-transform: uppercase;
    background: var(--env-soft); color: var(--env);
    padding: 3px 10px; border-radius: 999px; margin-bottom: 10px;
  }
  .envcard .obs {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 14.5px; white-space: pre-wrap; overflow-x: auto;
  }
  .envcard .none { color: var(--muted); font-style: italic; font-size: 15px; }
  .rescard {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 14px; padding: 20px; box-shadow: var(--shadow);
  }
  .resbadge {
    display: inline-block; font-size: 12.5px; font-weight: 800;
    letter-spacing: .08em; padding: 5px 14px; border-radius: 999px;
    margin-bottom: 12px;
  }
  .resbadge.by-experiment { background: var(--win-soft); color: var(--win); }
  .resbadge.by-human { background: var(--human-soft); color: var(--human); }
  .basis { font-size: 14.5px; color: var(--muted); margin-bottom: 14px; }
  .basis strong { color: var(--ink); }
  .callout { display: flex; gap: 16px; flex-wrap: wrap; }
  .callout > * { flex: 1 1 240px; }
  .fate {
    border-radius: 12px; padding: 14px 16px; border: 1px solid var(--line);
  }
  .fate .who { font-size: 12px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .06em; margin-bottom: 4px; }
  .fate .word { font-size: 18px; font-weight: 800; letter-spacing: .02em; }
  .fate.won { background: var(--win-soft); border-color: var(--win); }
  .fate.won .word, .fate.won .who { color: var(--win); }
  .fate.lost { background: var(--lose-soft); border-color: var(--lose); }
  .fate.lost .word, .fate.lost .who { color: var(--lose); }
  .fate.open { background: var(--human-soft); border-color: var(--human); }
  .fate.open .word, .fate.open .who { color: var(--human); }
  .decider { margin-top: 14px; font-size: 14px; color: var(--muted);
    border-top: 1px dashed var(--line); padding-top: 10px; }
  footer { text-align: center; color: var(--muted); font-size: 13px;
    margin-top: 8px; }
"""


def _evidence_list(items):
    if not items:
        return '<p class="none">No evidence recorded.</p>'
    rows = []
    for e in items:
        src = _esc(e.get("source", ""))
        obs = _esc(e.get("observation", ""))
        rows.append(f'<li><span class="src">{src}</span>{obs}</li>')
    return "<ul>" + "".join(rows) + "</ul>"


def _hypothesis_card(msg, side):
    conf = msg.get("confidence")
    conf_html = ""
    if conf is not None:
        conf_html = f'<p class="conf">stated confidence: {_esc(conf)}</p>'
    return (
        f'<article class="card side-{side}" data-side="{side}">'
        f'<span class="agent">{_esc(msg.get("sender", "?"))}</span>'
        f'<p class="claim">&ldquo;{_esc(msg.get("claim", ""))}&rdquo;</p>'
        f'<div class="ev"><h3>Their own evidence</h3>'
        f'{_evidence_list(msg.get("evidence", []))}</div>'
        f'{conf_html}'
        f'</article>'
    )


def _expectation_card(msg, side, expected):
    stated = _esc(expected) if expected else "(none recorded)"
    return (
        f'<div class="card side-{side} expects" data-side="{side}">'
        f'<span class="agent">{_esc(msg.get("sender", "?"))}</span>'
        f'<p class="lead">If this hypothesis is true</p>'
        f'<p class="stmt">&rarr; <em>{stated}</em></p>'
        f'</div>'
    )


def _flow_html(flow):
    a, b = flow["a"], flow["b"]
    outcome = flow["outcome"]
    winner_side = flow["winner_side"]
    exp = outcome.get("experiment", {}) or {}
    resolution = _esc(outcome.get("resolution", ""))
    basis = _esc(outcome.get("basis", ""))
    resolved = winner_side is not None

    # Stage 1 -- the two hypotheses.
    stage1 = (
        '<section class="stage"><div class="stagemark">1</div>'
        '<div class="stagehead"><h2>Hypotheses</h2>'
        '<p class="sub">Two agents disagree. Each states a claim and the '
        'evidence behind it.</p></div>'
        f'<div class="duo">{_hypothesis_card(a, "a")}{_hypothesis_card(b, "b")}</div>'
        '</section>'
    )

    # Stage 2 -- the discriminating experiment with both predictions,
    # displayed before the result.
    probe_txt = exp.get("experiment") or ""
    metric = exp.get("metric") or ""
    probe_html = ""
    if probe_txt:
        metric_html = (f'<span class="metric">metric: {_esc(metric)}</span>'
                       if metric else "")
        probe_html = f'<div class="probe">$ {_esc(probe_txt)}{metric_html}</div>'
    reasoning = exp.get("reasoning") or ""
    reason_html = (f'<p class="reason">Why this discriminates: {_esc(reasoning)}</p>'
                   if reasoning else "")
    discriminative = exp.get("discriminative", False)
    seal_note = ("recorded <strong>before the result existed</strong> &mdash; "
                 "neither agent can move the goalposts afterwards")
    if not discriminative:
        seal_note = ("no experiment separates these claims: "
                     "<strong>" + (_esc(reasoning) if reasoning else
                                   "both predict the same observation") +
                     "</strong>")
        reason_html = ""
    stage2 = (
        '<section class="stage"><div class="stagemark">2</div>'
        '<div class="stagehead"><h2>Predictions &mdash; written first</h2>'
        '<p class="sub">One discriminating test, and what each side commits '
        'to expecting from it.</p></div>'
        f'<div class="sealbar"><span class="lock">&#128274;</span>'
        f'<span>{seal_note}</span></div>'
        f'{probe_html}'
        f'<div class="duo">'
        f'{_expectation_card(a, "a", exp.get("expected_if_a") or "")}'
        f'{_expectation_card(b, "b", exp.get("expected_if_b") or "")}'
        f'</div>{reason_html}'
        '</section>'
    )

    # Stage 3 -- the observation the environment returned.
    obs_items = outcome.get("evidence", []) or []
    if obs_items:
        obs_html = "".join(
            f'<p class="obs">{_esc(e.get("observation", ""))}</p>'
            for e in obs_items
        )
    else:
        obs_html = ('<p class="none">The environment returned no observation. '
                    'There is nothing here to check the predictions against.</p>')
    stage3 = (
        '<section class="stage"><div class="stagemark">3</div>'
        '<div class="stagehead"><h2>Observation</h2>'
        '<p class="sub">What the environment actually returned &mdash; '
        'produced after, and independently of, the predictions above.</p></div>'
        f'<div class="envcard"><span class="who">environment</span>{obs_html}</div>'
        '</section>'
    )

    # Stage 4 -- the verdict.
    if resolved:
        win_msg, lose_msg = (a, b) if winner_side == "a" else (b, a)
        badge = f'<span class="resbadge by-experiment">{resolution}</span>'
        fates = (
            f'<div class="fate won" data-verdict="supported" '
            f'data-side="{winner_side}">'
            f'<p class="who">{_esc(win_msg.get("sender", "?"))}</p>'
            f'<p class="word">SUPPORTED</p></div>'
            f'<div class="fate lost" data-verdict="refuted" '
            f'data-side="{"b" if winner_side == "a" else "a"}">'
            f'<p class="who">{_esc(lose_msg.get("sender", "?"))}</p>'
            f'<p class="word">REFUTED</p></div>'
        )
        decider = ('The observation matched one written prediction and not the '
                   'other. The environment decided; no one&rsquo;s opinion did.')
    else:
        badge = f'<span class="resbadge by-human">{resolution}</span>'
        fates = (
            f'<div class="fate open"><p class="who">'
            f'{_esc(a.get("sender", "?"))} &amp; {_esc(b.get("sender", "?"))}</p>'
            f'<p class="word">NEITHER MARKED</p></div>'
        )
        decider = ('No observation could separate the two claims, so neither '
                   'hypothesis is marked supported or refuted. This dispute '
                   'goes to a human, honestly labelled as unresolved.')
    stage4 = (
        '<section class="stage"><div class="stagemark">4</div>'
        '<div class="stagehead"><h2>Verdict</h2></div>'
        f'<div class="rescard">{badge}'
        f'<p class="basis">Basis: <strong>{basis}</strong></p>'
        f'<div class="callout">{fates}</div>'
        f'<p class="decider">{decider}</p>'
        '</div></section>'
    )

    return f'<div class="flow">{stage1}{stage2}{stage3}{stage4}</div>'


def render(flows, title):
    """Render flows as a self-contained HTML page showing the ladder."""
    body = "".join(_flow_html(f) for f in flows)
    if not flows:
        body = '<p style="text-align:center;color:#888">No disagreements to show.</p>'
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n<body>\n<main>\n"
        '<header class="page"><h1>How the disagreement was settled</h1>'
        "<p>Claims first, committed expectations second, the world&rsquo;s "
        "answer third &mdash; in that order, on the record.</p></header>\n"
        f"{body}\n"
        "<footer>Generated by flow.py &middot; the ladder reads top to "
        "bottom, in the order things happened.</footer>\n"
        "</main>\n</body>\n</html>\n"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("usage: flow.py <exchange.json> <out.html>", file=sys.stderr)
        raise SystemExit(2)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    flows = load_flows(src)
    page = render(flows, title=f"Disagreement flow: {src.stem}")
    dst.write_text(page, encoding="utf-8")
    print(f"wrote {dst} ({len(page)} chars, {len(flows)} flow(s))")

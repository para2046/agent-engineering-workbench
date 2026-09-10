#!/usr/bin/env python3
"""viz.py -- render agent-communication logs as a self-contained HTML messageboard.

Supports two input shapes:
  * exchange logs   -- {"messages": [...], "disagreements": [...]}  (multi-agent)
  * trajectory logs -- {"steps": [...]}  (single agent talking to its environment)

Usage:  python viz.py <input.json> <output.html>
"""

import html
import json
import sys
from pathlib import Path


class RecordList(list):
    """A list of message records that also carries run-level metadata
    (disagreements, termination reason, metrics) without adding records."""

    def __init__(self, records=(), meta=None):
        super().__init__(records)
        self.meta = dict(meta or {})


# ---------------------------------------------------------------- loading

def load_records(path):
    """Load an agent-communication log and normalise it to message records.

    Every record is a dict with keys: sender, recipient, kind, text,
    round, evidence (list of {source, observation}), and optional extras.
    Raises ValueError for unrecognised formats.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "messages" in data:
        return _load_exchange(data)
    if isinstance(data, dict) and "steps" in data:
        return _load_trajectory(data)
    raise ValueError("unrecognised log format: %s" % path)


def _load_exchange(data):
    records = []
    for msg in data.get("messages", []):
        records.append({
            "sender": msg.get("sender", "?"),
            "recipient": msg.get("recipient", "?"),
            "kind": msg.get("type", "MESSAGE"),
            "text": msg.get("claim", ""),
            "round": msg.get("round"),
            "evidence": list(msg.get("evidence") or []),
            "confidence": msg.get("confidence"),
            "unknowns": list(msg.get("unknowns") or []),
        })
    meta = {
        "task_id": data.get("task_id"),
        "termination_reason": data.get("termination_reason"),
        "disagreements": list(data.get("disagreements") or []),
        "metrics": data.get("metrics") or {},
    }
    return RecordList(records, meta)


def _load_trajectory(data, agent_name="agent"):
    records = []
    for step in data.get("steps", []):
        number = step.get("step")
        action = step.get("action") or {}
        atype = action.get("type")
        if atype == "tool_call":
            args = action.get("arguments")
            text = "%s(%s)" % (action.get("tool", "?"),
                               json.dumps(args) if args else "")
            records.append({
                "sender": agent_name,
                "recipient": "environment",
                "kind": "TOOL_CALL",
                "text": text,
                "round": number,
                "evidence": [],
            })
            obs = step.get("observation") or {}
            summary = obs.get("summary")
            if summary is not None:
                result = step.get("tool_result") or {}
                kind = "OBSERVATION"
                if result.get("ok") is False:
                    kind = "OBSERVATION (error)"
                records.append({
                    "sender": "environment",
                    "recipient": agent_name,
                    "kind": kind,
                    "text": summary,
                    "round": number,
                    "evidence": [],
                })
        elif atype == "final_answer":
            records.append({
                "sender": agent_name,
                "recipient": "user",
                "kind": "FINAL_ANSWER",
                "text": step.get("rationale", ""),
                "round": number,
                "evidence": [],
            })
        else:
            records.append({
                "sender": agent_name,
                "recipient": "environment",
                "kind": str(atype or "STEP").upper(),
                "text": json.dumps(action),
                "round": number,
                "evidence": [],
            })
    meta = {
        "task_id": data.get("task_id") or data.get("trajectory_id"),
        "termination_reason": data.get("termination_reason"),
        "disagreements": [],
        "metrics": data.get("evaluation") or {},
    }
    return RecordList(records, meta)


# ---------------------------------------------------------------- rendering

_PALETTE = ["#4c7dd9", "#3f9d6f", "#a26bd6", "#d0862f", "#cf5f77", "#3aa2b8"]

_CSS = """
  :root {
    color-scheme: light dark;
    --bg: #f2f4f7;
    --panel: #ffffff;
    --ink: #1d2433;
    --muted: #667082;
    --border: #dcE1e8;
    --quote-bg: #eef1f5;
    --chip-bg: #e8ecf2;
    --shadow: 0 1px 2px rgba(16, 24, 40, .07);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #13161b;
      --panel: #1d232c;
      --ink: #e7eaf0;
      --muted: #99a3b1;
      --border: #313a46;
      --quote-bg: #161b22;
      --chip-bg: #29313c;
      --shadow: 0 1px 2px rgba(0, 0, 0, .35);
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
          sans-serif;
  }
  main { max-width: 780px; margin: 0 auto; padding: 28px 16px 56px; }
  header { margin-bottom: 22px; }
  h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: .2px; }
  .sub { margin: 0; color: var(--muted); font-size: 13px; }
  .board { display: flex; flex-direction: column; gap: 14px; }
  .row { display: flex; }
  .row.right { justify-content: flex-end; }
  .msg { display: flex; gap: 10px; max-width: 86%; align-items: flex-start; }
  .row.right .msg { flex-direction: row-reverse; }
  .avatar {
    flex: none;
    width: 34px; height: 34px;
    border-radius: 50%;
    background: var(--accent, #888);
    color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; font-weight: 600;
    margin-top: 2px;
    text-transform: uppercase;
  }
  .bubble {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent, var(--border));
    border-radius: 14px;
    border-top-left-radius: 4px;
    box-shadow: var(--shadow);
    padding: 10px 14px 11px;
    min-width: 0;
  }
  .row.right .bubble {
    border-left: 1px solid var(--border);
    border-right: 3px solid var(--accent, var(--border));
    border-top-left-radius: 14px;
    border-top-right-radius: 4px;
  }
  .meta {
    display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px;
    margin-bottom: 5px; font-size: 12px; color: var(--muted);
  }
  .sender { color: var(--accent, var(--ink)); font-weight: 700; font-size: 13px; }
  .chip {
    background: var(--chip-bg);
    border-radius: 999px;
    padding: 1px 9px;
    font-size: 11px;
    letter-spacing: .4px;
    white-space: nowrap;
  }
  .chip.kind { text-transform: uppercase; font-weight: 600; }
  .text { white-space: pre-wrap; overflow-wrap: anywhere; }
  .evidence {
    margin: 9px 0 0;
    padding: 7px 11px;
    background: var(--quote-bg);
    border-left: 3px solid var(--accent, var(--muted));
    border-radius: 0 8px 8px 0;
    font-size: 13px;
  }
  .evidence .src {
    display: block;
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .6px;
    margin-bottom: 2px;
  }
  .extra { margin-top: 8px; font-size: 12.5px; color: var(--muted); }
  .panel {
    margin-top: 26px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow);
    padding: 14px 18px;
  }
  .panel h2 {
    margin: 0 0 8px; font-size: 13px; text-transform: uppercase;
    letter-spacing: .8px; color: var(--muted);
  }
  .panel p { margin: 6px 0; font-size: 14px; }
  .badge {
    display: inline-block;
    background: var(--chip-bg);
    border-radius: 6px;
    padding: 2px 8px;
    font-weight: 600;
    font-size: 12.5px;
    letter-spacing: .3px;
  }
  .footer { margin-top: 18px; color: var(--muted); font-size: 12.5px; text-align: center; }
  .empty { color: var(--muted); text-align: center; padding: 40px 0; }
"""


def _esc(value):
    return html.escape(str(value), quote=True)


def _sender_styles(senders):
    rules = []
    for i, name in enumerate(senders):
        accent = _PALETTE[i % len(_PALETTE)]
        rules.append('  .msg[data-sender="%s"] { --accent: %s; }'
                     % (_esc(name), accent))
    return "\n".join(rules)


def _side_for(sender, senders):
    if sender == "environment":
        return "left"
    if "environment" in senders:
        return "right"
    return "left" if senders.index(sender) % 2 == 0 else "right"


def _render_record(rec, senders):
    sender = rec.get("sender", "?")
    side = _side_for(sender, senders)
    meta_bits = [
        '<span class="sender">%s</span>' % _esc(sender),
        '<span class="arrow">&#8594;</span>',
        '<span class="rcpt">%s</span>' % _esc(rec.get("recipient", "?")),
        '<span class="chip kind">%s</span>' % _esc(rec.get("kind", "MESSAGE")),
    ]
    if rec.get("round") is not None:
        meta_bits.append('<span class="chip">round %s</span>'
                         % _esc(rec["round"]))
    parts = [
        '<div class="row %s">' % side,
        '  <div class="msg" data-sender="%s">' % _esc(sender),
        '    <div class="avatar">%s</div>' % _esc(str(sender)[:1] or "?"),
        '    <div class="bubble">',
        '      <div class="meta">%s</div>' % "".join(meta_bits),
        '      <div class="text">%s</div>' % _esc(rec.get("text", "")),
    ]
    for ev in rec.get("evidence") or []:
        parts.append(
            '      <blockquote class="evidence">'
            '<span class="src">evidence &#183; %s</span>%s</blockquote>'
            % (_esc(ev.get("source", "?")), _esc(ev.get("observation", "")))
        )
    extras = []
    if rec.get("confidence") is not None:
        extras.append("confidence %s" % _esc(rec["confidence"]))
    if rec.get("unknowns"):
        extras.append("unknowns: %s"
                      % _esc("; ".join(str(u) for u in rec["unknowns"])))
    if extras:
        parts.append('      <div class="extra">%s</div>'
                     % " &#183; ".join(extras))
    parts.extend(['    </div>', '  </div>', '</div>'])
    return "\n".join(parts)


def _render_disagreements(disagreements):
    if not disagreements:
        return ""
    items = []
    for d in disagreements:
        outcome = d.get("outcome", d) if isinstance(d, dict) else {}
        bits = ['<span class="badge">%s</span>'
                % _esc(outcome.get("resolution", "UNRESOLVED"))]
        if outcome.get("winning_agent"):
            bits.append("won by <strong>%s</strong>"
                        % _esc(outcome["winning_agent"]))
        if outcome.get("basis"):
            bits.append("basis: %s" % _esc(outcome["basis"]))
        items.append("<p>%s</p>" % " &#183; ".join(bits))
    return ('<section class="panel"><h2>Disagreements</h2>%s</section>'
            % "".join(items))


def render(records, title="Agent Comms"):
    """Render normalised records as a complete standalone HTML page."""
    meta = getattr(records, "meta", None) or {}
    senders = []
    for rec in records:
        s = rec.get("sender", "?")
        if s not in senders:
            senders.append(s)

    if records:
        board = "\n".join(_render_record(r, senders) for r in records)
    else:
        board = '<p class="empty">No messages in this log.</p>'

    sub_bits = []
    if meta.get("task_id"):
        sub_bits.append("task: %s" % _esc(meta["task_id"]))
    if meta.get("termination_reason"):
        sub_bits.append("ended: %s" % _esc(meta["termination_reason"]))
    sub = (" &#183; ".join(sub_bits)) or "%d message(s)" % len(records)

    footer_bits = ["%s: %s" % (_esc(k), _esc(v))
                   for k, v in (meta.get("metrics") or {}).items()]
    footer = ('<p class="footer">%s</p>' % " &#183; ".join(footer_bits)
              if footer_bits else "")

    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%(title)s</title>
<style>
%(css)s
%(sender_css)s
</style>
</head>
<body>
<main>
<header>
  <h1>%(title)s</h1>
  <p class="sub">%(sub)s</p>
</header>
<section class="board">
%(board)s
</section>
%(disagreements)s
%(footer)s
</main>
</body>
</html>
""" % {
        "title": _esc(title),
        "css": _CSS,
        "sender_css": _sender_styles(senders),
        "sub": sub,
        "board": board,
        "disagreements": _render_disagreements(meta.get("disagreements")),
        "footer": footer,
    }


# ---------------------------------------------------------------- CLI

def main(argv):
    if len(argv) != 3:
        print("usage: python viz.py <input.json> <output.html>",
              file=sys.stderr)
        return 2
    in_path, out_path = Path(argv[1]), Path(argv[2])
    records = load_records(in_path)
    page = render(records, title=in_path.stem.replace("_", " "))
    out_path.write_text(page, encoding="utf-8")
    print("wrote %s (%d records)" % (out_path, len(records)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

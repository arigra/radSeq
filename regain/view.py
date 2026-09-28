"""ReGain prototype: show real project code with importance colours.

Code is read from the source file at display time, never copied, so what the
notebook shows is always the code that runs. Lines are marked by a substring
of their text rather than a line number; if a marked substring is no longer
found, the chapter is out of date and says so loudly.
"""
import html
import inspect
import os

from IPython.display import HTML, display

RED = "red"        # can change the result: read every character
YELLOW = "yellow"  # read it, but only the address or value it names

_CSS = """
<style>
.rg{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;line-height:1.55;
    border:1px solid rgba(128,128,128,.35);border-radius:8px;overflow-x:auto;margin:4px 0}
.rg .hd{font-family:system-ui,sans-serif;font-size:12px;padding:6px 10px;
    border-bottom:1px solid rgba(128,128,128,.25);opacity:.8}
.rg .ln{display:flex;white-space:pre;padding:0 10px}
.rg .no{width:3.2em;flex:none;text-align:right;padding-right:1em;opacity:.45;user-select:none}
.rg .dim{opacity:.38}
.rg .red{background:rgba(225,70,70,.16);box-shadow:inset 3px 0 0 rgb(225,70,70)}
.rg .yellow{background:rgba(230,180,40,.16);box-shadow:inset 3px 0 0 rgb(220,170,30)}
.rg .why{font-family:system-ui,sans-serif;white-space:normal;font-size:12.5px;
    padding:2px 10px 6px 4.6em;opacity:.9}
.rgb{font-family:system-ui,sans-serif;font-size:13px;border:1px dashed rgba(128,128,128,.5);
    border-radius:8px;padding:6px 10px;margin:4px 0}
.rgb summary{cursor:pointer}
.rgb code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px}
.rgb .ct{opacity:.8;margin-inline-start:.5em}
.rg-stale{font-family:system-ui,sans-serif;color:#fff;background:rgb(200,60,60);
    padding:8px 12px;border-radius:8px;margin:4px 0}
</style>"""


def _where(obj):
    path = inspect.getsourcefile(obj)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.relpath(path, root)


def code(obj, marks):
    """Show `obj`'s source; `marks` is a list of (level, substring, why).

    Unmarked lines are dimmed: visible for context, not for study.
    """
    lines, start = inspect.getsourcelines(obj)
    found, rows = set(), []
    for i, text in enumerate(lines):
        text = text.rstrip("\n")
        hit = next(((lvl, sub, why) for lvl, sub, why in marks if sub in text), None)
        cls = hit[0] if hit else "dim"
        if hit:
            found.add(hit[1])
        rows.append(f'<div class="ln {cls}"><span class="no">{start + i}</span>'
                    f'<span>{html.escape(text) or " "}</span></div>')
        if hit and hit[2]:
            rows.append(f'<div class="why {cls}">{html.escape(hit[2])}</div>')
    missing = [sub for _, sub, _ in marks if sub not in found]
    out = _CSS
    if missing:
        out += ('<div class="rg-stale">הקוד השתנה מאז שהפרק נכתב. השורות האלה כבר לא נמצאו: '
                + ", ".join(f"<code>{html.escape(m)}</code>" for m in missing) + "</div>")
    out += (f'<div class="rg"><div class="hd">{html.escape(_where(obj))} · '
            f'{html.escape(obj.__name__)}</div>{"".join(rows)}</div>')
    display(HTML(out))


def block(obj, contract):
    """A piece of code you do not need to read: its name and a one-line contract.

    The source is one click away, but reading it is not part of the chapter.
    """
    src = html.escape(inspect.getsource(obj))
    display(HTML(
        _CSS + f'<details class="rgb"><summary><code>{html.escape(obj.__name__)}</code>'
        f'<span class="ct">{html.escape(contract)}</span></summary>'
        f'<pre style="font-size:12px;opacity:.8">{src}</pre></details>'))

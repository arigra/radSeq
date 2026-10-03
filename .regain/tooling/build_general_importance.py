"""Build a portable ReGain line map for Python and C++ project notebooks.

The first pass describes syntax and data flow. An agent can add project-aware
reasons to .regain/importance-reviews.json after reading the whole project.
Only reviewed entries may claim domain-specific effects.
"""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

try:
    from cpp_importance import analyze_cpp, promote_cpp_definitions
except ModuleNotFoundError:
    # Older Python-only projects may have copied this script without the C++ helper.
    analyze_cpp = promote_cpp_definitions = None

LEVELS = {"critical", "important", "supporting"}
LEVEL_RANK = {"supporting": 0, "important": 1, "critical": 2}
CPP_SUFFIXES = {".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh", ".hxx"}
DISPLAY = re.compile(r"(?:^|\.)(?:print|display|show|show_table|plot|scatter|bar|hist|legend|grid|set|set_title|set_xlabel|set_ylabel|subplots|figure|savefig|imshow|step|stackplot|axhline|axvspan|colorbar)$")


def digest(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def short(value, length=105):
    value = " ".join(value.split())
    return value if len(value) <= length else value[: length - 1] + "…"


def target_name(node):
    target = node.targets[0] if isinstance(node, ast.Assign) else node.target
    return short(ast.unparse(target), 65)


def is_display(node):
    if not isinstance(node, ast.Call):
        return False
    return bool(DISPLAY.search(ast.unparse(node.func)))


def branch_action(node):
    body = node.body
    if not body:
        return "its branch"
    first = body[0]
    if isinstance(first, ast.Return):
        return "the returned value"
    if isinstance(first, ast.Raise):
        return "an error that stops this operation"
    if isinstance(first, ast.Continue):
        return "whether this item is skipped"
    if isinstance(first, ast.Break):
        return "whether this loop stops"
    if isinstance(first, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return f"the value of {target_name(first)}"
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Call):
        return f"the call to {short(ast.unparse(first.value.func), 55)}"
    return "the work in this branch"


def analyze(source):
    """Return a conservative map; semantic importance needs agent review."""
    lines = source.split("\n")
    levels = ["supporting" if line.strip() and not line.lstrip().startswith("#") else None for line in lines]
    reasons = ["Supports this code block; inspect its callers to determine its effect." if level else None for level in levels]
    confidence = ["low" if level else None for level in levels]
    tree = ast.parse(source)

    def mark(node, level, why, certainty="medium", header=False):
        end = node.end_lineno or node.lineno
        if header and getattr(node, "body", None):
            end = max(node.lineno, node.body[0].lineno - 1)
        for number in range(node.lineno, min(end, len(lines)) + 1):
            index = number - 1
            if levels[index] is not None:
                levels[index], reasons[index], confidence[index] = level, why, certainty

    # A parent assignment must not overwrite the meaning of a nested branch.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ", ".join(alias.name for alias in node.names)
            mark(node, "supporting", f"Makes {short(names, 70)} available to this code; importing alone does not run the main operation.", "high")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mark(node, "supporting", f"Defines {node.name}; its callers and body determine the behavior.", "high", header=True)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            value = node.value
            name = target_name(node)
            if value is None:
                mark(node, "supporting", f"Declares {name}'s type without assigning a value; later code supplies its runtime value.", "high")
            elif is_display(value):
                mark(node, "supporting", f"Stores the view produced by {short(ast.unparse(value.func), 55)}; this line controls presentation.", "high")
            elif isinstance(value, ast.Constant):
                mark(node, "important", f"Sets {name} to {short(repr(value.value), 55)}; check where this value is used before changing it.", "low")
            elif isinstance(value, ast.Call):
                mark(node, "important", f"Stores the result of {short(ast.unparse(value.func), 55)} in {name}; its effect depends on that call and later uses.", "low")
            else:
                mark(node, "important", f"Computes {name} from {short(ast.unparse(value), 85)}; later uses determine its impact.", "low")
        elif isinstance(node, ast.Return):
            result = short(ast.unparse(node.value), 80) if node.value else "None"
            mark(node, "important", f"Returns {result} to the caller; inspect callers to judge whether it controls a decision.", "low")
        elif isinstance(node, (ast.If, ast.While)):
            condition = short(ast.unparse(node.test), 85)
            action = branch_action(node)
            if all(isinstance(item, ast.Expr) and is_display(item.value) for item in node.body):
                mark(node, "supporting", f"Controls {action} when {condition} holds; this branch only presents results.", "medium", header=True)
            else:
                mark(node, "important", f"Controls {action} when {condition} holds; review its downstream effect before calling it a decision gate.", "low", header=True)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            mark(node, "important", f"Processes items from {short(ast.unparse(node.iter), 80)}; review what each iteration changes.", "low", header=True)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = short(ast.unparse(node.value.func), 75)
            level = "supporting" if is_display(node.value) else "important"
            why = (f"Calls {call} to present existing data; it does not change the underlying result."
                   if level == "supporting" else
                   f"Calls {call}; inspect the callee and later uses to determine its effect.")
            mark(node, level, why, "medium" if level == "supporting" else "low")
    return {"lines": levels, "reasons": reasons, "confidence": confidence}


def analyze_config(source):
    """Give readable draft marks to configuration files without guessing effects."""
    levels, reasons, confidence = [], [], []
    for line in source.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            levels.append(None); reasons.append(None); confidence.append(None)
            continue
        key = stripped.split(":", 1)[0].split("=", 1)[0].strip(' "\'')
        if (":" in stripped or "=" in stripped) and key:
            levels.append("important")
            reasons.append(f"Sets {short(key, 65)} in this configuration; trace where it is read to determine its effect.")
            confidence.append("low")
        else:
            levels.append("supporting")
            reasons.append("Structures this configuration; the settings inside determine runtime behavior.")
            confidence.append("medium")
    return {"lines": levels, "reasons": reasons, "confidence": confidence}


def blocks(notebook):
    lines = notebook.replace("\r\n", "\n").split("\n")
    index = 0
    while index < len(lines):
        opening = lines[index].strip()
        if not opening.startswith("```"):
            index += 1
            continue
        closing = index + 1
        while closing < len(lines) and lines[closing].strip() != "```":
            closing += 1
        if closing == len(lines):
            raise ValueError(f"Unclosed code fence at notebook line {index + 1}")
        if opening.startswith("```file "):
            words = opening[len("```file "):].split()
            match = re.fullmatch(r"(\d+)-(\d+)", words[1]) if len(words) > 1 else None
            yield "file", words[0], (int(match[1]), int(match[2])) if match else None
        elif opening == "```python":
            yield "cell", "\n".join(lines[index + 1:closing]), None
        index = closing + 1


def apply_reviews(plans, reviews, warnings):
    if not isinstance(reviews, list):
        raise ValueError("importance-reviews.json must be a JSON array")
    for index, review in enumerate(reviews, 1):
        label = f"review {index}"
        if not isinstance(review, dict) or ("file" in review) == ("cell_sha256" in review):
            warnings.append(f"{label}: supply exactly one of file or cell_sha256")
            continue
        location = review.get("file") or review.get("cell_sha256")
        kind = "files" if "file" in review else "cells"
        if review.get("importance") not in LEVELS or not isinstance(review.get("why"), str) or not review["why"].strip():
            warnings.append(f"{label}: importance must be critical/important/supporting and why must explain the effect")
            continue
        plan = plans[kind].get(location)
        if not plan or review.get("sha256") != plan["sha256"]:
            warnings.append(f"{label}: source missing or changed; review {location} again")
            continue
        match = review.get("match")
        if not isinstance(match, str) or not match.strip():
            warnings.append(f"{label}: add an exact, nonempty line match")
            continue
        source_lines = plan["source"].split("\n")
        matches = [i for i, line in enumerate(source_lines) if match in line]
        if "line" in review:
            line_number = review["line"]
            if isinstance(line_number, int) and not isinstance(line_number, bool) and line_number - 1 in matches:
                matches = [line_number - 1]
            else:
                warnings.append(f"{label}: line must point to a source line containing match in {location}")
                continue
        if len(matches) != 1:
            warnings.append(f"{label}: match must identify one line in {location}; found {len(matches)}")
            continue
        line = matches[0]
        plan["lines"][line] = review["importance"]
        plan["reasons"][line] = review["why"].strip()
        plan["confidence"][line] = "reviewed"


def promote_definitions(plan):
    """Give definition headers the strongest importance found in their body.

    This is a draft inference. An explicit agent review of a header wins.
    """
    tree = ast.parse(plan["source"])
    definitions = [node for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for node in sorted(definitions, key=lambda item: item.lineno, reverse=True):
        if not node.body:
            continue
        first_body = node.body[0].lineno - 1
        end = min(node.end_lineno or node.lineno, len(plan["lines"]))
        candidates = [index for index in range(first_body, end)
                      if plan["lines"][index] in {"critical", "important"}]
        if not candidates:
            continue
        strongest = max(candidates, key=lambda index: (
            LEVEL_RANK[plan["lines"][index]],
            plan["confidence"][index] == "reviewed",
        ))
        level = plan["lines"][strongest]
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        reason = (f"Defines {kind} {node.name}; its body includes a "
                  f"{level} step: {short(plan['reasons'][strongest], 115)}")
        for index in range(node.lineno - 1, max(node.lineno, first_body)):
            if plan["lines"][index] is not None and plan["confidence"][index] != "reviewed":
                plan["lines"][index] = level
                plan["reasons"][index] = reason
                plan["confidence"][index] = "low"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="project root (default: current directory)")
    parser.add_argument("--notebook", type=Path, action="append", help=".regain.md path; repeat for more notebooks")
    parser.add_argument("--require-reviewed", action="store_true", help="fail while important/critical lines still have draft confidence")
    parser.add_argument("--replace-existing", action="store_true", help="replace a map made by another generator")
    args = parser.parse_args()
    root = args.root.resolve()
    regain = root / ".regain"
    output = regain / "line-importance.json"
    if output.is_file() and not args.replace_existing:
        try:
            previous_version = json.loads(output.read_text(encoding="utf-8")).get("version")
        except (ValueError, AttributeError):
            previous_version = None
        if previous_version != 3:
            parser.error("An existing line map came from another generator; use --replace-existing to replace it")
    notebooks = args.notebook or sorted(regain.glob("*.regain.md"))
    if not notebooks:
        parser.error("No .regain.md notebook found under .regain; pass --notebook")
    plans = {"files": {}, "cells": {}}
    warnings = []
    for notebook in notebooks:
        path = (root / notebook).resolve() if not notebook.is_absolute() else notebook.resolve()
        if not path.is_relative_to(root):
            parser.error(f"Notebook must be inside the project: {path}")
        for kind, value, visible_range in blocks(path.read_text(encoding="utf-8")):
            if kind == "file":
                target = (root / value).resolve()
                if not target.is_relative_to(root):
                    warnings.append(f"File block escapes project root: {value}")
                    continue
                if target.suffix.lower() not in ({".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini"} | CPP_SUFFIXES) or not target.is_file():
                    warnings.append(f"Unsupported or missing file block: {value}")
                    continue
                key = target.relative_to(root).as_posix()
                source = target.read_text(encoding="utf-8").replace("\r\n", "\n")
            else:
                source = value
                key = digest(source)
            group = plans["files" if kind == "file" else "cells"]
            if visible_range:
                lo, hi = visible_range
                if lo < 1 or hi < lo:
                    warnings.append(f"Invalid line range for {value}: {lo}-{hi}")
                    continue
                visible = set(range(lo - 1, min(hi, len(source.split("\n")))))
            else:
                visible = set(range(len(source.split("\n"))))
            if key in group:
                group[key]["visible"].update(visible)
                continue
            try:
                if kind == "cell" or target.suffix.lower() == ".py":
                    plan = analyze(source)
                elif target.suffix.lower() in CPP_SUFFIXES:
                    if analyze_cpp is None:
                        parser.error("Copy cpp_importance.py beside this generator to map C++ files")
                    plan = analyze_cpp(source)
                else:
                    plan = analyze_config(source)
            except SyntaxError as error:
                warnings.append(f"Cannot parse {key}: {error}")
                continue
            plan["sha256"] = digest(source)
            plan["source"] = source
            plan["visible"] = visible
            group[key] = plan
    review_path = regain / "importance-reviews.json"
    if review_path.is_file():
        apply_reviews(plans, json.loads(review_path.read_text(encoding="utf-8")), warnings)
    for kind, group in plans.items():
        for key, plan in group.items():
            if kind == "cells" or Path(key).suffix.lower() == ".py":
                promote_definitions(plan)
            elif Path(key).suffix.lower() in CPP_SUFFIXES:
                promote_cpp_definitions(plan)
    summary = []
    for kind, group in plans.items():
        for key, plan in group.items():
            pending = sum(index in plan["visible"] and level in {"critical", "important"} and certainty != "reviewed"
                          for index, (level, certainty) in enumerate(zip(plan["lines"], plan["confidence"])))
            label = key if kind == "files" else f"cell {key[:12]}: {short(plan['source'].splitlines()[0], 65)}"
            summary.append(f"- {label} — {pending} decision/calculation lines to review; sha256 `{plan['sha256']}`")
    prompt = regain / "importance-agent-review.md"
    prompt.write_text("""# ReGain agent review

Work through this project as an engineer would before assigning semantic
importance. Read its README, architecture and configuration documents, the
linked source files, their callers, and the notebook. Trace the main inputs,
outputs, decisions, and side effects. If a claim remains uncertain, say so in
the reason and keep its importance conservative. Do not claim perfect project
understanding from syntax alone.

Inspect `.regain/line-importance.json`. The generated first pass describes
syntax and marks uncertain lines with low confidence. For every important or
critical line, decide whether changing it could affect a project result,
whether it only affects presentation, and what downstream behavior depends on
it. Review supporting lines that might actually select input data or control
processing. Give each correction a short, specific reason an engineer can
understand on hover. A conditional used only for logging or plotting is
supporting, even if it uses `if`. Mark a decision gate critical only when its
effect has been traced. Report unresolved questions instead of guessing.
Definition headers inherit the strongest body color as a draft. Review each
function or class as a whole: a helper can stay green even if it contains a
branch, and a public entry point may matter more than any one body line.
The C++ first pass uses a lightweight scanner, not a compiler. Check macros,
templates, overloads, build flags, and call sites before trusting its map.

Write corrections to `.regain/importance-reviews.json` as a JSON array. Each
entry needs `file` (relative to project root) or `cell_sha256`, `sha256`, an
exact `match` unique to one line, `importance` (critical, important, or
supporting), and `why`. If a match repeats, also supply its one-based `line`
number. Use the sha256 shown below so changed code cannot
silently retain an old explanation. For example, the shape is:

```json
[
  {
    "file": "path/to/module.py",
    "sha256": "copy the current file hash from the list below",
    "match": "an exact substring of one source line",
    "importance": "critical",
    "why": "Explain the specific downstream decision this line controls."
  }
]
```

Run the same generator command again to apply reviews. Resolve every
"Review needed" message. Revisit explanations after source changes.

## Sources to understand and review

""" + "\n".join(summary) + "\n", encoding="utf-8")
    pending_total = sum(index in plan["visible"] and level in {"critical", "important"} and certainty == "low"
                        for group in plans.values() for plan in group.values()
                        for index, (level, certainty) in enumerate(zip(plan["lines"], plan["confidence"])))
    for group in plans.values():
        for plan in group.values():
            del plan["source"]
            del plan["visible"]
    regain.mkdir(exist_ok=True)
    output.write_text(json.dumps({"version": 3, **plans}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({len(plans['files'])} files, {len(plans['cells'])} cells)")
    print(f"Agent review instructions: {prompt}")
    print(f"Draft important/critical lines: {pending_total}")
    for warning in warnings:
        print("Review needed:", warning)
    if args.require_reviewed and (pending_total or warnings):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

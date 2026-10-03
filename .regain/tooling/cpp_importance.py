"""Conservative C++ line-importance draft for ReGain.

This scanner recognizes braces, declarations, and common statements while
ignoring comments and quoted text. It is not a C++ compiler: macros, templates,
and complex declarators require an engineering agent's review.
"""

import re

RANK = {"supporting": 0, "important": 1, "critical": 2}
CONTROL = re.compile(r"\b(if|else\s+if|switch|while|for|catch)\s*\(")
ASSIGN = re.compile(r"(?<![=!<>])(?:\+=|-=|\*=|/=|%=|=(?!=))")
LOG = re.compile(r"\b(?:std::(?:cout|cerr|clog)|printf|fprintf|LOG_[A-Z_]+)\b")
CLASS = re.compile(r"\b(class|struct)\s+([A-Za-z_]\w*)\b[^;{}]*$", re.S)
FUNCTION = re.compile(
    r"(?P<name>(?:~?[A-Za-z_]\w*::)*~?[A-Za-z_]\w*|operator\s*[^\s(]+)"
    r"\s*\((?:[^()]|\([^()]*\))*\)"
    r"\s*(?:(?:const|volatile|override|final)\s*|noexcept(?:\([^)]*\))?\s*|"
    r"->\s*[A-Za-z_][\w:<>,*& ]*\s*)*$",
    re.S,
)


def short(value, limit=110):
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def code_only(source):
    """Keep line positions and code punctuation; blank comments and literals."""
    out = []
    index = 0
    state = "code"
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char == "/" and next_char == "/":
                out.extend("  "); index += 2; state = "line"; continue
            if char == "/" and next_char == "*":
                out.extend("  "); index += 2; state = "block"; continue
            if char in {'"', "'"}:
                out.append(" "); index += 1; state = char; continue
            out.append(char)
        elif state == "line":
            out.append("\n" if char == "\n" else " ")
            if char == "\n":
                state = "code"
        elif state == "block":
            if char == "*" and next_char == "/":
                out.extend("  "); index += 2; state = "code"; continue
            out.append("\n" if char == "\n" else " ")
        else:
            if char == "\\" and next_char:
                out.extend(("\n" if char == "\n" else " ",
                            "\n" if next_char == "\n" else " "))
                index += 2
                continue
            out.append("\n" if char == "\n" else " ")
            if char == state:
                state = "code"
        index += 1
    return "".join(out)


def definitions(source):
    """Find likely class and function bodies, with one-based header lines."""
    code = code_only(source)
    stack = []
    found = []
    for position, char in enumerate(code):
        if char == "{":
            stack.append(position)
            continue
        if char != "}" or not stack:
            continue
        opening = stack.pop()
        boundary = max(code.rfind(";", 0, opening), code.rfind("{", 0, opening),
                       code.rfind("}", 0, opening))
        prefix = code[boundary + 1:opening]
        header_end = code.count("\n", 0, opening) + 1
        body_end = code.count("\n", 0, position) + 1
        class_match = CLASS.search(prefix)
        if class_match and not re.search(r"\benum\s+class\b", prefix[:class_match.end()]):
            kind, name, start = class_match.group(1), class_match.group(2), class_match.start()
        else:
            function_match = FUNCTION.search(prefix)
            if not function_match:
                continue
            name = function_match.group("name")
            if name in {"if", "for", "while", "switch", "catch"}:
                continue
            kind, start = "function", function_match.start("name")
        header_start = code.count("\n", 0, boundary + 1 + start) + 1
        found.append((header_start, header_end, body_end, kind, name))
    return sorted(found, key=lambda item: item[0], reverse=True)


def analyze_cpp(source):
    """Describe likely C++ decisions and calculations without claiming impact."""
    raw_lines = source.split("\n")
    code_lines = code_only(source).split("\n")
    levels, reasons, confidence = [], [], []
    for raw, code in zip(raw_lines, code_lines):
        stripped = code.strip()
        if not stripped:
            levels.append(None); reasons.append(None); confidence.append(None)
            continue
        if stripped.startswith("#"):
            level, why = "supporting", "Preprocessor directive; inspect active build settings to know its effect."
        elif LOG.search(code):
            level, why = "supporting", "Reports a value for diagnostics; inspect the call before treating it as a decision."
        elif (match := CONTROL.search(code)):
            level = "important"
            why = f"Controls the {match.group(1)} branch; trace its body and callers to judge impact."
        elif re.search(r"\b(?:return|co_return|throw)\b", code):
            level, why = "important", "Returns a value or stops this operation; trace its caller to judge impact."
        elif "constexpr" in code:
            level, why = "important", "Defines a compile-time value; trace where it is used before changing it."
        elif ASSIGN.search(code):
            left = short(code.split("=", 1)[0], 65)
            level, why = "important", f"Sets or updates {left}; trace later uses to judge its effect."
        elif re.search(r"\b[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*\s*\(.*\)\s*;\s*$", code):
            level, why = "important", "Calls an operation; inspect its side effects and downstream use."
        else:
            level, why = "supporting", "Supports this C++ block; inspect its context to determine its effect."
        levels.append(level); reasons.append(why); confidence.append("low")
    return {"lines": levels, "reasons": reasons, "confidence": confidence}


def promote_cpp_definitions(plan):
    """Infer header color from the strongest body line unless reviewed."""
    for start, header_end, body_end, kind, name in definitions(plan["source"]):
        body = [index for index in range(header_end - 1, min(body_end, len(plan["lines"])))
                if plan["lines"][index] in {"critical", "important"}]
        if not body:
            continue
        strongest = max(body, key=lambda index: (
            RANK[plan["lines"][index]], plan["confidence"][index] == "reviewed"))
        level = plan["lines"][strongest]
        reason = (f"Defines {kind} {name}; its body includes a {level} step: "
                  f"{short(plan['reasons'][strongest], 105)}")
        for index in range(start - 1, min(header_end, len(plan["lines"]))):
            if plan["lines"][index] is not None and plan["confidence"][index] != "reviewed":
                plan["lines"][index] = level
                plan["reasons"][index] = reason
                plan["confidence"][index] = "low"

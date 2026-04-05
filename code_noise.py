from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta, timezone

from faker import Faker


fake = Faker()


def _ident() -> str:
    """Return a plausible variable/function identifier."""
    if random.random() < 0.10:
        ident = random.choice([
            "i",
            "j",
            "k",
            "n",
            "x",
            "y",
            "z",
            "t",
            "v",
            "m",
            "c",
            "r",
            "s",
            "e",
            "f",
            "u",
        ])
        return ident
    common_prefix = random.choice([
        "",
        "",
        "",
        "get",
        "set",
        "my",
        "new",
        "is",
        "has",
        "make",
        "build",
        "load",
        "save",
        "read",
        "write",
        "calc",
        "find",
        "run",
        "temp",
        "_"
    ])
    base = fake.word().replace("-", "_").lower().strip("_") or "value"
    if common_prefix:
        base = f"{common_prefix}_{base}" if random.random() < 0.7 else f"{common_prefix}{base.title()}"
    mode = random.random()
    if mode < 0.25:
        ident = base
    if mode < 0.50:
        ident = f"{base}{random.randint(0, 999)}"
    if mode < 0.70:
        ident = f"{base}_{random.randint(0, 999)}"
    if mode < 0.85:
        ident = f"{base}{random.randint(1, 9)}_{fake.word().replace('-', '_').lower().strip('_')}"
    else:
        parts = [fake.word().replace("-", "_") for _ in range(random.randint(1, 2))]
        ident = "_".join(p.lower() for p in parts if p)
        ident = ident.strip("_") or base

    max_len = random.choice([12, 16, 18, 20, 24, 32])
    if len(ident) > max_len and random.random() < 0.7:
        suffix = ""
        prefix = ident
        if "_" in ident and random.random() < 0.8:
            prefix, suffix = ident.rsplit("_", 1)
            suffix = "_" + suffix
        elif any(ch.isdigit() for ch in ident) and random.random() < 0.5:
            idx = next((i for i, ch in enumerate(ident) if ch.isdigit()), len(ident))
            prefix, suffix = ident[:idx], ident[idx:]
        keep = max(3, max_len - len(suffix))
        ident = prefix[:keep].rstrip("_") + suffix
    return ident or base


def _camel() -> str:
    """Return a plausible class-like identifier."""
    parts = [p.title() for p in fake.words(nb=random.randint(1, 3))]
    return "".join(parts) or "Result"


def _module_part() -> str:
    """Return a short module/package path segment."""
    common_modules = [
        "api",
        "app",
        "auth",
        "cache",
        "client",
        "config",
        "core",
        "data",
        "db",
        "engine",
        "errors",
        "events",
        "helpers",
        "http",
        "io",
        "lib",
        "loader",
        "logger",
        "main",
        "math",
        "models",
        "net",
        "parser",
        "pipeline",
        "policy",
        "queue",
        "schema",
        "service",
        "session",
        "store",
        "tasks",
        "text",
        "types",
        "utils",
        "validation",
        "view",
        "worker",
    ]
    if random.random() < 0.35:
        return random.choice(common_modules)
    base = fake.word().replace("-", "").replace("_", "").lower().strip() or "core"
    if len(base) > 10 and random.random() < 0.8:
        base = base[: random.randint(4, 10)]
    if random.random() < 0.20:
        base = f"{base}{random.randint(0, 9)}"
    elif random.random() < 0.30:
        base = f"{base}_{random.randint(0, 9)}"
    return base


def _module_name(min_parts: int = 1, max_parts: int = 3) -> str:
    """Return a compact fictional module/package name."""
    parts = [_module_part() for _ in range(random.randint(min_parts, max_parts))]
    return ".".join(parts)


_VALUE_SPECS: dict[str, dict[str, object]] = {
    "generic": {
        "number": lambda: [str(random.randint(-9999, 9999))],
        "string": lambda: [f'"{fake.word()}"'],
        "bool": lambda: ["true", "false", "True", "False"],
        "null": lambda: ["null", "None", "nil", "NULL", "nullptr"],
        "float": lambda: [f"{random.uniform(-1000, 1000):.3f}"],
        "collection": lambda: [
            f"[{', '.join(repr(fake.word()) for _ in range(random.randint(1, 4)))}]",
        ],
        "object": lambda: [
            f"{{'value': {random.randint(0, 9)}}}",
            f"{{ value: {random.randint(0, 9)} }}",
        ],
    },
    "python": {
        "number": lambda: [str(random.randint(-9999, 9999))],
        "string": lambda: [f'"{fake.word()}"'],
        "bool": lambda: ["True", "False"],
        "null": lambda: ["None"],
        "float": lambda: [f"{random.uniform(-1000, 1000):.3f}"],
        "collection": lambda: [
            f"[{', '.join(repr(fake.word()) for _ in range(random.randint(1, 4)))}]",
            f"{{'value': {random.randint(0, 9)}}}",
            f"{{'items': [{', '.join(str(random.randint(0, 9)) for _ in range(random.randint(1, 3)))}]}}",
        ],
        "object": lambda: [
            f"{{'value': {random.randint(0, 9)}}}",
            f"{{'name': '{fake.word()}'}}",
        ],
    },
    "js": {
        "number": lambda: [str(random.randint(-9999, 9999))],
        "string": lambda: [f'"{fake.word()}"'],
        "bool": lambda: ["true", "false"],
        "null": lambda: ["null"],
        "float": lambda: [f"{random.uniform(-1000, 1000):.3f}"],
        "collection": lambda: [
            f"[{', '.join(repr(fake.word()) for _ in range(random.randint(1, 4)))}]".replace("'", '"'),
            f"{{ value: {random.randint(0, 9)} }}",
        ],
        "object": lambda: [
            f"{{ value: {random.randint(0, 9)} }}",
            f"{{ name: \"{fake.word()}\" }}",
        ],
    },
    "go": {
        "number": lambda: [str(random.randint(-9999, 9999))],
        "string": lambda: [f'"{fake.word()}"'],
        "bool": lambda: ["true", "false"],
        "null": lambda: ["nil"],
        "float": lambda: [f"{random.uniform(-1000, 1000):.3f}"],
        "collection": lambda: [
            f"[]string{{\"{fake.word()}\", \"{fake.word()}\"}}",
            f"[]int{{{random.randint(0, 9)}, {random.randint(0, 9)}}}",
            f"map[string]int{{\"{fake.word()}\": {random.randint(0, 9)}}}",
        ],
        "object": lambda: [
            f"Config{{Name: \"{fake.word()}\", Count: {random.randint(0, 9)}}}",
        ],
    },
    "java": {
        "number": lambda: [str(random.randint(-9999, 9999))],
        "string": lambda: [f'"{fake.word()}"'],
        "bool": lambda: ["true", "false"],
        "null": lambda: ["null"],
        "float": lambda: [f"{random.uniform(-1000, 1000):.3f}"],
        "collection": lambda: [
            f"List.of(\"{fake.word()}\", \"{fake.word()}\")",
            f"Map.of(\"{fake.word()}\", {random.randint(0, 9)})",
        ],
        "object": lambda: [
            "new Object()",
            f"List.of({random.randint(0, 9)}, {random.randint(0, 9)})",
        ],
    },
    "c": {
        "number": lambda: [str(random.randint(-9999, 9999))],
        "string": lambda: [f'"{fake.word()}"'],
        "bool": lambda: ["true", "false"],
        "null": lambda: ["NULL"],
        "float": lambda: [f"{random.uniform(-1000, 1000):.3f}"],
        "collection": lambda: [
            f"{{{random.randint(0, 9)}, {random.randint(0, 9)}}}",
            f"{{\"{fake.word()}\", \"{fake.word()}\"}}",
        ],
        "object": lambda: [
            "(void *)0",
        ],
    },
    "cpp": {
        "number": lambda: [str(random.randint(-9999, 9999))],
        "string": lambda: [f'"{fake.word()}"'],
        "bool": lambda: ["true", "false"],
        "null": lambda: ["nullptr"],
        "float": lambda: [f"{random.uniform(-1000, 1000):.3f}"],
        "collection": lambda: [
            f"std::vector<int>{{{random.randint(0, 9)}, {random.randint(0, 9)}}}",
            f"std::map<std::string, int>{{{{\"{fake.word()}\", {random.randint(0, 9)}}}}}",
        ],
        "object": lambda: [
            f"std::string(\"{fake.word()}\")",
        ],
    },
}


def _value(language: str = "generic", kind: str | None = None) -> str:
    """Return a mixed literal value, tuned to the target language."""
    spec = _VALUE_SPECS.get(language, _VALUE_SPECS["generic"])
    if kind is None:
        kind = random.choices(
            ["number", "string", "bool", "null", "float", "collection", "object"],
            weights=[0.22, 0.22, 0.16, 0.10, 0.14, 0.10, 0.06],
            k=1,
        )[0]
    pool_factory = spec.get(kind) or _VALUE_SPECS["generic"].get(kind) or _VALUE_SPECS["generic"]["string"]
    pool = pool_factory() if callable(pool_factory) else pool_factory
    return random.choice(pool)


def _arglist() -> str:
    names = [_ident() for _ in range(random.randint(1, 4))]
    return ", ".join(names)


def _block(lines: list[str], indent: str = "    ") -> str:
    return "\n".join(indent + line for line in lines)


def _chance(prob: float, mult: float = 1.0) -> bool:
    return random.random() < min(1.0, max(0.0, prob * mult))


def _snippet_mult() -> float:
    return random.choices([0.3, 0.55, 0.8, 1.0], weights=[3, 3, 2, 1], k=1)[0]


def _module_path(min_parts: int = 2, max_parts: int = 4) -> str:
    return _module_name(min_parts=min_parts, max_parts=max_parts)


def _package_path(min_parts: int = 2, max_parts: int = 4) -> str:
    parts = [_module_part() for _ in range(random.randint(min_parts, max_parts))]
    return "/".join(parts)


def _comparison_op(style: str = "sql") -> str:
    """Return a comparison operator suitable for the requested style."""
    if style == "sql":
        return random.choice(["=", "!=", "<>", ">", "<", ">=", "<="])
    if style == "text":
        return random.choice(["=", "!=", "<>", "LIKE"])
    return random.choice(["==", "!=", ">", "<", ">=", "<="])


def _comparison_rhs(style: str = "numeric") -> str:
    """Return a plausible right-hand comparison literal."""
    if style == "text":
        return random.choice([
            "''",
            f'"{fake.word()}"',
            f'"{fake.word()}%"',
        ])
    if style == "float":
        return f"{random.uniform(-1000, 1000):.2f}"
    return str(random.randint(0, 9999))


def _comparison_expr(left: str, *, style: str = "sql", rhs_style: str = "numeric") -> str:
    """Return a simple comparison expression string."""
    return f"{left} {_comparison_op(style)} {_comparison_rhs(rhs_style)}"


def _numeric_literal(kind: str = "int") -> str:
    if kind == "float":
        return f"{random.uniform(0, 20):.2f}"
    return str(random.randint(0, 20))


def _python_type(*, allow_optional: bool = True) -> str:
    """Return a plausible Python type annotation."""
    primitive = random.choice(["int", "float", "bool", "str", "bytes"])
    simple = [
        primitive,
        "object",
        f"list[{primitive}]",
        f"set[{primitive}]",
        f"tuple[{primitive}, {primitive}]",
        f"dict[str, {primitive}]",
    ]
    advanced = [
        f"typing.Optional[{primitive}]",
        f"typing.Sequence[{primitive}]",
        f"typing.Iterable[{primitive}]",
        f"typing.Mapping[str, {primitive}]",
        f"typing.MutableMapping[str, {primitive}]",
    ]
    pool = simple + advanced if allow_optional else simple
    return random.choice(pool)


def _c_type() -> str:
    return random.choice([
        "int",
        "long",
        "unsigned int",
        "size_t",
        "double",
        "float",
        "bool",
        "char *",
        "const char *",
    ])


def _cpp_type() -> str:
    return random.choice([
        "int",
        "long long",
        "double",
        "float",
        "bool",
        "std::string",
    ])


def _zero_value_for_type(type_name: str, *, language: str = "c") -> str:
    type_name = type_name.strip()
    if "char *" in type_name or "const char *" in type_name:
        return "NULL" if language == "c" else "nullptr"
    if "string" in type_name.lower():
        return '""' if language != "c" else '""'
    if type_name == "bool":
        return "false"
    if "float" in type_name or "double" in type_name:
        return "0.0"
    if "size_t" in type_name or "int" in type_name or "long" in type_name:
        return "0"
    return "0"


def _c_printf_format(type_name: str) -> str:
    type_name = type_name.strip()
    if "char *" in type_name or "const char *" in type_name or "unsigned char *" in type_name:
        return "%s"
    if "double" in type_name or "float" in type_name:
        return "%f"
    if "unsigned int" in type_name:
        return "%u"
    if "size_t" in type_name:
        return "%zu"
    if "long" in type_name:
        return "%ld"
    if type_name == "bool":
        return "%d"
    return "%d"


def _control_flow_block(spec: dict[str, object]) -> list[str]:
    """Generate one small control-flow block using language-specific templates."""
    indent = str(spec.get("indent", "    "))
    value_types = list(spec.get("value_types", ["int", "long", "float"]))
    value_type = random.choice(value_types)
    probe_name = str(spec.get("probe_name") or _ident())
    state_name = str(spec.get("state_name") or _ident())
    idx_name = str(spec.get("idx_name") or _ident())
    probe_value = _numeric_literal(value_type)
    limit_value = _numeric_literal(value_type)
    loop_limit_value = _numeric_literal("int")
    comparison_op = _comparison_op("c")
    case_values = [0, 1, 2]
    mode = random.choice(list(spec.get("modes", ["if", "loop", "switch"])))
    pad = lambda level=1: indent * level

    def _fmt(key: str, **kwargs: object) -> str:
        template = spec.get(key)
        if key in {"else_header_template", "close_template", "case_break_template"}:
            return str(template)
        data = {
            "probe_name": probe_name,
            "state_name": state_name,
            "idx_name": idx_name,
            "probe_value": probe_value,
            "limit_value": limit_value,
            "loop_limit_value": loop_limit_value,
            "comparison_op": comparison_op,
            "value_type": value_type,
            "case_value": 0,
            "else_case_value": 0,
        }
        data.update(kwargs)
        return str(template).format(**data)

    lines: list[str] = []
    for template in spec.get("init_templates", []):
        lines.append(pad() + str(template).format(
            probe_name=probe_name,
            state_name=state_name,
            idx_name=idx_name,
            probe_value=probe_value,
            limit_value=limit_value,
            loop_limit_value=loop_limit_value,
            comparison_op=comparison_op,
            value_type=value_type,
        ))

    body_template = str(spec.get("body_template", ""))
    else_body_template = str(spec.get("else_body_template", body_template))
    close_template = str(spec.get("close_template", ""))
    case_break_template = str(spec.get("case_break_template", ""))

    if mode == "if":
        lines.append(pad() + _fmt("if_header_template"))
        if body_template:
            lines.append(_block([body_template.format(
                probe_name=probe_name,
                state_name=state_name,
                idx_name=idx_name,
                probe_value=probe_value,
                limit_value=limit_value,
                loop_limit_value=loop_limit_value,
                comparison_op=comparison_op,
                value_type=value_type,
            )], indent + indent))
        if random.random() < float(spec.get("if_else_prob", 0.5)) and spec.get("else_header_template"):
            lines.append(pad() + _fmt("else_header_template"))
            if else_body_template:
                lines.append(_block([else_body_template.format(
                    probe_name=probe_name,
                    state_name=state_name,
                    idx_name=idx_name,
                    probe_value=probe_value,
                    limit_value=limit_value,
                    comparison_op=comparison_op,
                    value_type=value_type,
                )], indent + indent))
        if close_template:
            lines.append(pad() + close_template)
        return lines

    if mode == "loop":
        if random.random() < float(spec.get("while_prob", 0.5)) and spec.get("while_header_template"):
            lines.append(pad() + _fmt("while_header_template"))
        else:
            lines.append(pad() + _fmt("for_header_template"))
        if body_template:
            lines.append(_block([body_template.format(
                probe_name=probe_name,
                state_name=state_name,
                idx_name=idx_name,
                probe_value=probe_value,
                limit_value=limit_value,
                loop_limit_value=loop_limit_value,
                comparison_op=comparison_op,
                value_type=value_type,
            )], indent + indent))
        if close_template:
            lines.append(pad() + close_template)
        return lines

    lines.append(pad() + _fmt("switch_header_template"))
    for case_value in case_values[: random.randint(1, len(case_values))]:
        lines.append(pad() + _fmt("case_header_template", case_value=case_value))
        if body_template:
            lines.append(_block([body_template.format(
                probe_name=probe_name,
                state_name=state_name,
                idx_name=idx_name,
                probe_value=probe_value,
                limit_value=limit_value,
                comparison_op=comparison_op,
                value_type=value_type,
            )], indent + indent))
        if case_break_template:
            lines.append(pad() + case_break_template)
    lines.append(pad() + _fmt("default_header_template"))
    if else_body_template:
        lines.append(_block([else_body_template.format(
            probe_name=probe_name,
            state_name=state_name,
            idx_name=idx_name,
            probe_value=probe_value,
            limit_value=limit_value,
            comparison_op=comparison_op,
            value_type=value_type,
        )], indent + indent))
    if close_template:
        lines.append(pad() + close_template)
    return lines


def _python_control_flow(indent: str = "    ") -> list[str]:
    return _control_flow_block({
        "indent": indent,
        "value_types": ["int", "long", "float"],
        "init_templates": [
            "{probe_name} = {probe_value}",
            "{state_name} = 0",
        ],
        "if_header_template": "if {probe_name} {comparison_op} {limit_value}:",
        "else_header_template": "else:",
        "for_header_template": "for {idx_name} in range({loop_limit_value}):",
        "while_header_template": "while {probe_name} {comparison_op} {limit_value}:",
        "switch_header_template": "match {probe_name}:",
        "case_header_template": "case {case_value}:",
        "default_header_template": "case _:",
        "body_template": "{state_name} = {state_name} + {probe_name}",
        "else_body_template": "print({probe_name})",
        "case_break_template": "",
        "close_template": "",
        "modes": ["if", "loop", "switch"],
        "while_prob": 0.5,
        "if_else_prob": 0.5,
    })


def _js_control_flow(indent: str = "    ") -> list[str]:
    return _control_flow_block({
        "indent": indent,
        "value_types": ["int", "int", "float"],
        "init_templates": [
            "let {probe_name} = {probe_value};",
            "let {state_name} = 0;",
        ],
        "if_header_template": "if ({probe_name} {comparison_op} {limit_value}) {{",
        "else_header_template": "} else {",
        "for_header_template": "for (let {idx_name} = 0; {idx_name} < {limit_value}; {idx_name}++) {{",
        "while_header_template": "while ({probe_name} {comparison_op} {limit_value}) {{",
        "switch_header_template": "switch ({probe_name}) {{",
        "case_header_template": "case {case_value}:",
        "default_header_template": "default:",
        "body_template": "{state_name} = {state_name} + {probe_name}; console.log({probe_name}, {state_name});",
        "else_body_template": "console.log({probe_name});",
        "case_break_template": "break;",
        "close_template": "}",
        "modes": ["if", "loop", "switch"],
        "while_prob": 0.5,
        "if_else_prob": 0.5,
    })


def _go_control_flow(indent: str = "    ") -> list[str]:
    return _control_flow_block({
        "indent": indent,
        "value_types": ["int", "int64", "float64"],
        "init_templates": [
            "var {probe_name} {value_type} = {probe_value}",
            "var {state_name} int = 0",
        ],
        "if_header_template": "if {probe_name} {comparison_op} {limit_value} {{",
        "else_header_template": "} else {",
        "for_header_template": "for {idx_name} := 0; {idx_name} < {limit_value}; {idx_name}++ {{",
        "while_header_template": "for {probe_name} {comparison_op} {limit_value} {{",
        "switch_header_template": "switch {probe_name} {{",
        "case_header_template": "case {case_value}:",
        "default_header_template": "default:",
        "body_template": "{state_name} = {state_name} + 1; fmt.Println({probe_name}, {state_name})",
        "else_body_template": "fmt.Println({probe_name})",
        "case_break_template": "break",
        "close_template": "}",
        "modes": ["if", "loop", "switch"],
        "while_prob": 0.5,
        "if_else_prob": 0.5,
    })


def _java_control_flow(indent: str = "        ") -> list[str]:
    return _control_flow_block({
        "indent": indent,
        "value_types": ["int", "long", "float", "double"],
        "init_templates": [
            "{value_type} {probe_name} = {probe_value};",
            "int {state_name} = 0;",
        ],
        "if_header_template": "if ({probe_name} {comparison_op} {limit_value}) {{",
        "else_header_template": "} else {",
        "for_header_template": "for (int {idx_name} = 0; {idx_name} < {limit_value}; {idx_name}++) {{",
        "while_header_template": "while ({probe_name} {comparison_op} {limit_value}) {{",
        "switch_header_template": "switch ((int) {probe_name}) {{",
        "case_header_template": "case {case_value}:",
        "default_header_template": "default:",
        "body_template": "{state_name} = {state_name} + 1; System.out.println({probe_name});",
        "else_body_template": "System.out.println({probe_name});",
        "case_break_template": "break;",
        "close_template": "}",
        "modes": ["if", "loop", "switch"],
        "while_prob": 0.5,
        "if_else_prob": 0.5,
    })


def _c_control_flow(indent: str = "    ") -> list[str]:
    return _control_flow_block({
        "indent": indent,
        "value_types": ["int", "long", "float"],
        "init_templates": [
            "{value_type} {probe_name} = {probe_value};",
            "int {state_name} = 0;",
        ],
        "if_header_template": "if ({probe_name} {comparison_op} {limit_value}) {{",
        "else_header_template": "} else {",
        "for_header_template": "for (int {idx_name} = 0; {idx_name} < {limit_value}; ++{idx_name}) {{",
        "while_header_template": "while ({probe_name} {comparison_op} {limit_value}) {{",
        "switch_header_template": "switch ((int) {probe_name}) {{",
        "case_header_template": "case {case_value}:",
        "default_header_template": "default:",
        "body_template": "{state_name} = {state_name} + 1; printf(\"[trace] %d\\n\", {state_name});",
        "else_body_template": "printf(\"[trace] %d\\n\", {state_name});",
        "case_break_template": "break;",
        "close_template": "}",
        "modes": ["if", "loop", "switch"],
        "while_prob": 0.5,
        "if_else_prob": 0.5,
    })


def _rust_control_flow(indent: str = "    ") -> list[str]:
    probe_name = _ident()
    state_name = _ident()
    idx_name = _ident()
    mode = random.choice(["if", "loop", "switch"])
    value_type = random.choice(["i32", "i64"]) if mode in {"loop", "switch"} else random.choice(["i32", "i64", "f32", "f64"])
    probe_value = _numeric_literal("float" if "f" in value_type else "int")
    limit_value = _numeric_literal("float" if "f" in value_type else "int")
    loop_limit_value = _numeric_literal("int")
    comparison_op = random.choice(["==", "!=", ">", "<", ">=", "<="])

    lines = [
        f"{indent}let mut {probe_name}: {value_type} = {probe_value};",
        f"{indent}let mut {state_name}: i32 = 0;",
    ]

    if mode == "if":
        lines.append(f"{indent}if {probe_name} {comparison_op} {limit_value} {{")
        lines.append(f"{indent}    {state_name} = {state_name} + 1; println!(\"[trace] {{}}\", {probe_name});")
        if random.random() < 0.5:
            lines.append(f"{indent}}} else {{")
            lines.append(f"{indent}    println!(\"[trace] {{}}\", {probe_name});")
        lines.append(f"{indent}}}")
        return lines

    if mode == "loop":
        if random.random() < 0.5:
            lines.append(f"{indent}for {idx_name} in 0..{loop_limit_value} {{")
        else:
            lines.append(f"{indent}while {probe_name} {comparison_op} {limit_value} {{")
        lines.append(f"{indent}    {state_name} = {state_name} + 1; println!(\"[trace] {{}}\", {probe_name});")
        lines.append(f"{indent}}}")
        return lines

    lines.append(f"{indent}match {probe_name} {{")
    for case_value in [0, 1, 2][: random.randint(1, 3)]:
        lines.append(f"{indent}    {case_value} => {{")
        lines.append(f"{indent}        {state_name} = {state_name} + 1; println!(\"[trace] {{}}\", {probe_name});")
        lines.append(f"{indent}    }},")
    lines.append(f"{indent}    _ => {{")
    lines.append(f"{indent}        println!(\"[trace] {{}}\", {probe_name});")
    lines.append(f"{indent}    }}")
    lines.append(f"{indent}}}")
    return lines


def _format_imports(spec: dict[str, object]) -> list[str]:
    """Format import lines from a small spec dict.

    Supports:
    - flat sub_modules lists
    - dict trees like sklearn -> model_selection -> train_test_split
    - per-entry template overrides
    """

    def _entry_to_lines(
        main_module: str,
        entry: object,
        templates: list[str],
        pieces: dict[str, str],
    ) -> list[str]:
        if isinstance(entry, dict):
            if "sub_module" in entry:
                sub_module = str(entry.get("sub_module", ""))
                alias = str(entry.get("alias") or entry.get("as") or "")
                template = str(entry.get("template") or random.choice(templates))
                extra = {
                    k: v
                    for k, v in entry.items()
                    if isinstance(k, str) and k not in {"sub_module", "template", "alias", "as"}
                }
                rendered = dict(pieces)
                rendered.update(extra)
                rendered["sub_module"] = sub_module
                rendered["alias"] = alias
                rendered["alias_clause"] = f" as {alias}" if alias else ""
                rendered["alias_prefix"] = f"{alias} " if alias else ""
                rendered["alias_name"] = alias or rendered["ident"]
                return [template.format(**rendered)]

            lines: list[str] = []
            branch_items = list(entry.items())
            for sub_module, leaves in branch_items:
                if leaves in (None, [], {}, ""):
                    lines.append(
                        random.choice(templates).format(
                            **{**pieces, "sub_module": str(sub_module), "alias": "", "alias_clause": ""}
                        )
                    )
                    continue
                if isinstance(leaves, (list, tuple, set)):
                    chosen_leafs = random.sample(list(leaves), k=min(random.randint(1, len(leaves)), len(leaves)))
                    for leaf in chosen_leafs:
                        leaf_name = str(leaf["name"]) if isinstance(leaf, dict) and "name" in leaf else str(leaf)
                        leaf_alias = ""
                        if isinstance(leaf, dict):
                            leaf_alias = str(leaf.get("alias") or leaf.get("as") or "")
                        lines.append(
                            random.choice(templates).format(
                                **{
                                    **pieces,
                                    "sub_module": f"{sub_module}",
                                    "leaf": leaf_name,
                                    "alias": leaf_alias,
                                    "alias_clause": f" as {leaf_alias}" if leaf_alias else "",
                                    "alias_prefix": f"{leaf_alias} " if leaf_alias else "",
                                    "alias_name": leaf_alias or pieces["ident"],
                                }
                            )
                        )
                    continue
                if isinstance(leaves, dict):
                    nested_lines = _entry_to_lines(
                        main_module,
                        {f"{sub_module}.{nested_key}": nested_leafs for nested_key, nested_leafs in leaves.items()},
                        templates,
                        pieces,
                    )
                    lines.extend(nested_lines)
                    continue
                lines.append(
                    random.choice(templates).format(
                        **{
                            **pieces,
                            "sub_module": str(sub_module),
                            "alias": "",
                            "alias_clause": "",
                            "alias_prefix": "",
                            "alias_name": pieces["ident"],
                        }
                    )
                )
            return lines

        if isinstance(entry, (list, tuple, set)):
            lines = []
            chosen = random.sample(list(entry), k=min(random.randint(1, len(entry)), len(entry)))
            for sub_module in chosen:
                alias = ""
                if isinstance(sub_module, dict):
                    alias = str(sub_module.get("alias") or sub_module.get("as") or "")
                    sub_module = sub_module.get("name") or sub_module.get("sub_module") or sub_module
                lines.append(
                    random.choice(templates).format(
                        **{
                            **pieces,
                            "sub_module": str(sub_module),
                            "alias": alias,
                            "alias_clause": f" as {alias}" if alias else "",
                            "alias_prefix": f"{alias} " if alias else "",
                            "alias_name": alias or pieces["ident"],
                        }
                    )
                )
            return lines

        if entry in (None, "", []):
            return [
                random.choice(templates).format(
                    **{
                        **pieces,
                        "alias": "",
                        "alias_clause": "",
                        "alias_prefix": "",
                        "alias_name": pieces["ident"],
                    }
                )
            ]

        return [
            random.choice(templates).format(
                **{
                    **pieces,
                    "sub_module": str(entry),
                    "alias": "",
                    "alias_clause": "",
                    "alias_prefix": "",
                    "alias_name": pieces["ident"],
                }
            )
        ]

    templates = list(spec.get("templates", []))
    if not templates:
        return [""]

    raw_candidates = list(spec.get("sub_modules", []))
    raw_candidates.extend(spec.get("extra_modules", []))
    main_module = str(spec.get("main_module", ""))
    count = random.randint(int(spec.get("min_items", 1)), int(spec.get("max_items", 3)))
    chance_mult = float(spec.get("chance_mult", 1.0))
    if chance_mult < 0.7 and count > 1 and random.random() < 0.7:
        count = 1
    elif chance_mult < 0.9 and count > 2 and random.random() < 0.5:
        count = 2
    count = min(count, len(raw_candidates) if raw_candidates else 1)

    if raw_candidates:
        chosen = random.sample(raw_candidates, k=count)
    else:
        chosen = [None] * count

    pieces = {
        "main_module": main_module,
        "sub_module": "",
        "sub_modules_csv": ", ".join(str(s) for s in spec.get("sub_modules_csv", [])),
        "dotted_module": _module_path(),
        "package_path": _package_path(),
        "ident": _ident(),
        "camel": _camel(),
        "leaf": _ident(),
        "alias": "",
        "alias_clause": "",
        "alias_prefix": "",
        "alias_name": _ident(),
    }

    lines: list[str] = []
    for entry in chosen:
        lines.extend(_entry_to_lines(main_module, entry, templates, pieces))

    return lines + [""]


def _format_import_tree(
    tree: dict[str, object],
    *,
    max_main: int = 3,
    chance_mult: float = 1.0,
) -> list[str]:
    """Format nested imports like sklearn -> model_selection -> train_test_split."""
    if not tree:
        return [""]

    main_items = list(tree.items())
    main_count = random.randint(1, max_main)
    if chance_mult < 0.7 and main_count > 1 and random.random() < 0.7:
        main_count = 1
    elif chance_mult < 0.9 and main_count > 2 and random.random() < 0.5:
        main_count = 2
    chosen_main = random.sample(main_items, k=min(main_count, len(main_items)))
    lines: list[str] = []
    reserved_keys = {"__alias__", "alias", "__imports__"}

    for main_module, subtree in chosen_main:
        if isinstance(subtree, dict):
            alias = str(subtree.get("__alias__") or subtree.get("alias") or "")
            direct_imports = subtree.get("__imports__", [])
            if alias:
                lines.append(f"import {main_module} as {alias}")
            if isinstance(direct_imports, (list, tuple, set)) and direct_imports:
                chosen_direct = random.sample(
                    list(direct_imports), k=min(random.randint(1, len(direct_imports)), len(direct_imports))
                )
                for item in chosen_direct:
                    if isinstance(item, dict):
                        leaf_name = str(item.get("name") or item.get("sub_module") or item.get("leaf") or "")
                        leaf_alias = str(item.get("alias") or item.get("as") or "")
                        if leaf_name:
                            if leaf_alias:
                                lines.append(f"from {main_module} import {leaf_name} as {leaf_alias}")
                            else:
                                lines.append(f"from {main_module} import {leaf_name}")
                    else:
                        lines.append(f"from {main_module} import {item}")

            branch_subtree = {k: v for k, v in subtree.items() if k not in reserved_keys}
            if not branch_subtree:
                continue
            subtree = branch_subtree

        if subtree in (None, [], {}, ""):
            lines.append(f"import {main_module}")
            continue

        if isinstance(subtree, list):
            chosen_sub = random.sample(subtree, k=min(random.randint(1, len(subtree)), len(subtree)))
            if all(isinstance(item, str) for item in chosen_sub):
                for item in chosen_sub:
                    lines.append(f"from {main_module} import {item}")
            else:
                for item in chosen_sub:
                    if isinstance(item, dict):
                        leaf_name = str(item.get("name") or item.get("sub_module") or item.get("leaf") or "")
                        leaf_alias = str(item.get("alias") or item.get("as") or "")
                        if leaf_name:
                            if leaf_alias:
                                lines.append(f"from {main_module} import {leaf_name} as {leaf_alias}")
                            else:
                                lines.append(f"from {main_module} import {leaf_name}")
            continue

        if isinstance(subtree, dict):
            branch_items = list(subtree.items())
            chosen_branch = random.sample(branch_items, k=min(random.randint(1, len(branch_items)), len(branch_items)))
            for sub_module, leaves in chosen_branch:
                if isinstance(leaves, dict):
                    leaf_alias = str(leaves.get("__alias__") or leaves.get("alias") or "")
                    leaf_direct_imports = leaves.get("__imports__", [])
                    if leaf_alias:
                        lines.append(f"import {main_module}.{sub_module} as {leaf_alias}")
                    if isinstance(leaf_direct_imports, (list, tuple, set)) and leaf_direct_imports:
                        chosen_direct = random.sample(
                            list(leaf_direct_imports),
                            k=min(random.randint(1, len(leaf_direct_imports)), len(leaf_direct_imports)),
                        )
                        for item in chosen_direct:
                            if isinstance(item, dict):
                                leaf_name = str(item.get("name") or item.get("sub_module") or item.get("leaf") or "")
                                leaf_alias = str(item.get("alias") or item.get("as") or "")
                                if leaf_name:
                                    if leaf_alias:
                                        lines.append(f"from {main_module}.{sub_module} import {leaf_name} as {leaf_alias}")
                                    else:
                                        lines.append(f"from {main_module}.{sub_module} import {leaf_name}")
                            else:
                                lines.append(f"from {main_module}.{sub_module} import {item}")
                    leaf_branch = {k: v for k, v in leaves.items() if k not in reserved_keys}
                    if not leaf_branch:
                        continue
                    leaves = leaf_branch
                if leaves in (None, [], {}, ""):
                    lines.append(f"from {main_module}.{sub_module} import {sub_module}")
                    continue
                if isinstance(leaves, list):
                    chosen_leaf = random.sample(leaves, k=min(random.randint(1, len(leaves)), len(leaves)))
                    for leaf in chosen_leaf:
                        if isinstance(leaf, dict):
                            leaf_name = str(leaf.get("name") or leaf.get("sub_module") or leaf.get("leaf") or "")
                            leaf_alias = str(leaf.get("alias") or leaf.get("as") or "")
                            if leaf_name:
                                if leaf_alias:
                                    lines.append(f"from {main_module}.{sub_module} import {leaf_name} as {leaf_alias}")
                                else:
                                    lines.append(f"from {main_module}.{sub_module} import {leaf_name}")
                        else:
                            lines.append(f"from {main_module}.{sub_module} import {leaf}")
                    continue
                if isinstance(leaves, dict):
                    nested = list(leaves.items())
                    chosen_nested = random.sample(nested, k=min(random.randint(1, len(nested)), len(nested)))
                    for leaf_group, leaf_items in chosen_nested:
                        if isinstance(leaf_items, list) and leaf_items:
                            chosen_leaf = random.sample(leaf_items, k=min(random.randint(1, len(leaf_items)), len(leaf_items)))
                            for leaf in chosen_leaf:
                                if isinstance(leaf, dict):
                                    leaf_name = str(leaf.get("name") or leaf.get("sub_module") or leaf.get("leaf") or "")
                                    leaf_alias = str(leaf.get("alias") or leaf.get("as") or "")
                                    if leaf_name:
                                        if leaf_alias:
                                            lines.append(
                                                f"from {main_module}.{sub_module}.{leaf_group} import {leaf_name} as {leaf_alias}"
                                            )
                                        else:
                                            lines.append(f"from {main_module}.{sub_module}.{leaf_group} import {leaf_name}")
                                else:
                                    lines.append(f"from {main_module}.{sub_module}.{leaf_group} import {leaf}")
                        else:
                            lines.append(f"from {main_module}.{sub_module}.{leaf_group} import {leaf_group}")
            continue

        if isinstance(subtree, str):
            lines.append(f"import {subtree}")

    return lines + [""]


def _python_import_header() -> list[str]:
    chance_mult = _snippet_mult()
    tree = {
        "re": None,
        "random": None,
        "json": None,
        "pathlib": None,
        "os": None,
        "sys": None,
        "math": None,
        "statistics": None,
        "typing": None,
        "numpy": {"__alias__": "np"},
        "pandas": {"__alias__": "pd", "__imports__": ["DataFrame", "Series", "read_csv", "to_parquet", "to_json", "read_excel"
        ]},
        "matplotlib": {"pyplot": {"__alias__": "plt", "__imports__": ["plot", "figure", "subplots"]}},
        "seaborn": {"__alias__": "sns", "__imports__": ["set_theme", "heatmap", "lineplot"]},
        "requests": None,
        "tqdm": {"__alias__": "tqdm", "__imports__": ["tqdm"]},
        "torch": ["nn", "tensor", "optim"],
        "sklearn": {
            "model_selection": [{"name": "train_test_split", "alias": "tts"}, "GridSearchCV", "KFold"],
            "metrics": ["accuracy_score", "classification_report", "f1_score"],
            "preprocessing": ["StandardScaler", "OneHotEncoder"],
            'text': ["CountVectorizer", "TfidfVectorizer", "ENGLISH_STOP_WORDS"],
        },
        "dataclasses": ["dataclass"],
        "collections": ["defaultdict", "Counter", "deque"],
        "itertools": ["chain", "islice", "product"],
        "datetime": ["datetime", "timedelta"],
        "pydantic": ["BaseModel", "Field"],
        "rich": {"console": ["Console"], "progress": ["track"]},
        "faker": ["Faker"],
        "scipy": {"stats": ["ttest_ind", "zscore"]},
        "bs4": ["BeautifulSoup"],
        "transformers": ["pipeline", "AutoTokenizer", "AutoModel", ],
        "nltk": {"tokenize": ["word_tokenize", "sent_tokenize"], "corpus": ["stopwords"]},
        "torchvision": {"models": ["resnet50", "vit_b_16"], "transforms": ["Compose", "Resize", "ToTensor"]},
        "fastapi": ["FastAPI", "Depends", "HTTPException", "Query"],
        "sqlalchemy": {"orm": ["Session", "declarative_base", "relationship"], "sql": ["select", "insert", "update"]},
        "pyspark": {"sql": ["SparkSession", "functions", "types"]},
        "django": {"db": ["models"], "http": ["JsonResponse", "HttpResponse"]},
        "flask": ["Flask", "request", "jsonify", "render_template"],
        "pytest": ["fixture", "mark", "raises"],
        "typing": ["Any", "Callable", "TypeVar", "Generic", "Protocol", "runtime_checkable", "Union", "Optional", "Iterable", "Sequence", "Mapping", "MutableMapping", "Dict", "List", "Set", "Tuple", "NamedTuple", "TypedDict", "cast", "overload", "final", "Literal"],
        _module_path(): None,
        _module_path(): {"tools": [_camel(), _camel()]},
    }
    return _format_import_tree(tree, max_main=2 if chance_mult < 0.8 else 3, chance_mult=chance_mult)


def _js_import_header() -> list[str]:
    chance_mult = _snippet_mult()
    spec = {
        "main_module": "",
        "sub_modules": [
            {"sub_module": "fs", "alias": "fs"},
            {"sub_module": "path", "alias": "path"},
            {"sub_module": "crypto", "alias": "crypto"},
            {"sub_module": "luxon", "alias": "DateTime"},
            {"sub_module": "uuid", "alias": "uuidv4"},
            {"sub_module": "os", "alias": "os"},
            {"sub_module": _module_path(), "alias": _camel()},
            {"sub_module": _module_path(), "alias": _ident()},
            {"sub_module": "axios", "alias": "axios"},
            {"sub_module": "express", "alias": "express"},
            {"sub_module": "dotenv", "alias": "dotenv"},
            {"sub_module": "lodash", "alias": "_"},
            
        ],
        "templates": [
            'import {alias_name} from "node:{sub_module}";',
            'import {{ {alias_name} }} from "{sub_module}";',
            'const {alias_name} = require("{sub_module}");',
            'const {alias_name} = require("node:{sub_module}");',
        ],
        "min_items": 1,
        "max_items": 2 if chance_mult < 0.8 else 3,
        "chance_mult": chance_mult,
    }
    return _format_imports(spec)


def _go_import_header() -> list[str]:
    chance_mult = _snippet_mult()
    spec = {
        "main_module": "fmt",
        "sub_modules": [
            {"sub_module": "context", "alias": "context"},
            {"sub_module": "encoding/json", "alias": "json"},
            {"sub_module": "net/http", "alias": "http"},
            {"sub_module": "os", "alias": "os"},
            {"sub_module": "strings", "alias": "strings"},
            {"sub_module": "time", "alias": "time"},
            {"sub_module": "github.com/google/uuid", "alias": "uuid"},
            {"sub_module": "golang.org/x/sync/errgroup", "alias": "errgroup"},
            {"sub_module": _package_path(), "alias": _ident()},
            {"sub_module": f"{_package_path()}/{_ident()}", "alias": _ident()},
        ],
        "templates": [
            '{alias_prefix}"{sub_module}"',
        ],
        "min_items": 1,
        "max_items": 2 if chance_mult < 0.85 else 3,
        "chance_mult": chance_mult,
    }
    lines = _format_imports(spec)
    return ["import (", '    "fmt"'] + [f"    {imp}" for imp in lines[:-1]] + [")", ""]


def _java_import_header() -> list[str]:
    chance_mult = _snippet_mult()
    spec = {
        "main_module": "java.util",
        "sub_modules": [
            {"sub_module": "*", "template": "import {main_module}.{sub_module};"},
            {"sub_module": "ArrayList", "template": "import {main_module}.{sub_module};"},
            {"sub_module": "HashMap", "template": "import {main_module}.{sub_module};"},
            {"sub_module": "HashSet", "template": "import {main_module}.{sub_module};"},
            {"sub_module": "LinkedHashMap", "template": "import {main_module}.{sub_module};"},
            {"sub_module": "List", "template": "import {main_module}.{sub_module};"},
            {"sub_module": "Map", "template": "import {main_module}.{sub_module};"},
            {"sub_module": "Set", "template": "import {main_module}.{sub_module};"},
            {"sub_module": "Collections", "template": "import {main_module}.{sub_module};"},
            {"sub_module": "Optional", "template": "import {main_module}.{sub_module};"},
            {"sub_module": "java.time.*", "template": "import {sub_module};"},
            {"sub_module": "java.io.*", "template": "import {sub_module};"},
            {"sub_module": "java.nio.file.*", "template": "import {sub_module};"},
            {"sub_module": "java.util.concurrent.*", "template": "import {sub_module};"},
            {"sub_module": "java.util.stream.*", "template": "import {sub_module};"},
            {"sub_module": "org.slf4j.Logger", "template": "import {sub_module};"},
            {"sub_module": "org.slf4j.LoggerFactory", "template": "import {sub_module};"},
            {"sub_module": "com.fasterxml.jackson.databind.ObjectMapper", "template": "import {sub_module};"},
            {"sub_module": f"{_module_path()}.*", "template": "import {sub_module};"},
            {"sub_module": f"{_module_path()}.{_camel()}", "template": "import {sub_module};"},
        ],
        "templates": [
            "import {main_module}.{sub_module};",
            "import {sub_module};",
        ],
        "min_items": 1,
        "max_items": 2 if chance_mult < 0.8 else 3,
        "chance_mult": chance_mult,
    }
    return _format_imports(spec)


def _html_attrs() -> dict[str, str]:
    """Return a small bundle of realistic-looking HTML attributes."""
    domain = fake.domain_name()
    return {
        "class": fake.word(),
        "id": fake.slug(),
        "href": f"https://{domain}/",
        "src": f"https://{domain}/{fake.slug()}.png",
        "content": fake.word(),
        "name": fake.word(),
        "property": random.choice(["og:title", "og:description", "og:image", "og:type"]),
        "lang": random.choice(["en", "fr", "de", "es", "pl", "pt", "zh", "ja"]),
    }


def generate_html_artifact() -> str:
    """Generate an HTML snippet with tags preserved and text content removed."""
    a = _html_attrs()
    inline_styles = [
        f'style="display: {random.choice(["block", "flex", "grid", "inline-block"])}; '
        f'margin: {_css_length()}; '
        f'padding: {_css_length()}; '
        f'color: {_hex_color()}; '
        f'background-color: {random.choice([_hex_color(), "transparent", _rgb_color()])}; '
        f'font-family: {random.choice(["Inter, system-ui, sans-serif", "Roboto, Arial, sans-serif", "Georgia, serif", "Segoe UI, Tahoma, sans-serif"])};"',
        f'style="width: {random.choice([_css_length(), "100%", "min-content", "fit-content"])}; '
        f'height: {random.choice([_css_length(), "auto", "min-content", "fit-content"])}; '
        f'border-radius: {_css_length()}; '
        f'box-shadow: {random.choice(["0 1px 2px rgba(0,0,0,.1)", "0 4px 12px rgba(0,0,0,.15)", "none"])};"',
        f'style="font-size: {random.choice(["0.875rem", "1rem", "1.125rem", "16px", "18px"])}; '
        f'line-height: {random.choice(["1.2", "1.4", "1.6", "2"])}; '
        f'text-align: {random.choice(["left", "center", "right"])};"',
    ]
    inline_style = random.choice(inline_styles)

    templates = [
        (
            f'<!DOCTYPE html><html lang="{a["lang"]}"><head>'
            f'<meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title></title>'
            f'<meta name="description" content="">'
            f'<meta property="{a["property"]}" content="{a["content"]}">'
            f'<link rel="canonical" href="{a["href"]}">'
            f'</head><body><div class="{a["class"]}" id="{a["id"]}" {inline_style}><span></span></div></body></html>'
        ),
        (
            f'<header class="{a["class"]}" {inline_style}><nav><ul>'
            f'<li></li><li></li><li></li>'
            f'</ul></nav></header>'
        ),
        (
            f'<article id="{a["id"]}"><section class="{a["class"]}" {inline_style}>'
            f'<h1></h1><p></p><p></p><aside></aside>'
            f'</section></article>'
        ),
        (
            f'<!-- {fake.slug()} --><script type="application/ld+json">{{}}</script>'
            f'<style></style><noscript></noscript>'
        ),
        (
            f'<div class="{a["class"]}" {inline_style}><div><span></span><span></span></div>'
            f'<footer><a href="{a["href"]}"></a></footer></div>'
        ),
        (
            f'<meta charset="utf-8"><meta name="{a["name"]}" content="{a["content"]}">'
            f'<link rel="preload" as="image" href="{a["src"]}">'
        ),
    ]

    return random.choice(templates)


def _css_selector() -> str:
    parts = random.choice([
        [f".{fake.word()}"],
        [f"#{fake.slug()}"],
        [f"body"],
        [f"html"],
        [f"main"],
        [f"header"],
        [f"nav"],
        [f"section"],
        [f"article"],
        [f".{fake.word()}", f".{fake.word()}"],
        [f"[data-{_ident()}]"],
    ])
    return "".join(parts)


def _hex_color() -> str:
    """Return a plausible CSS hex color."""
    mode = random.random()
    if mode < 0.2:
        return random.choice(["#000", "#111", "#222", "#333", "#eee", "#fff"])
    return "#" + "".join(random.choice("0123456789abcdef") for _ in range(6))


def _rgb_color() -> str:
    """Return a plausible CSS rgb()/rgba() color."""
    if random.random() < 0.5:
        return f"rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})"
    return (
        f"rgba({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)}, "
        f"{random.uniform(0.05, 1.0):.2f})"
    )


def _css_length() -> str:
    """Return a plausible CSS length or size value."""
    unit = random.choice(["px", "rem", "em", "%", "vh", "vw"])
    if unit == "%":
        return f"{random.randint(0, 100)}%"
    if unit in {"vh", "vw"}:
        return f"{random.randint(0, 100)}{unit}"
    if unit in {"px", "rem", "em"}:
        val = random.choice([0, 0.125, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 6, 8, 12, 16, 24, 32])
        if unit == "px":
            return f"{int(val * 16) if val < 1 else int(val) if float(val).is_integer() else val}px"
        return f"{val:g}{unit}"
    return str(random.randint(0, 100))


def _css_value(prop: str) -> str:
    if "color" in prop:
        return random.choice([_hex_color(), _hex_color(), _hex_color(), "rgb(34, 34, 34)", "hsl(210, 50%, 45%)"])
    if prop in {"margin", "padding", "border-radius", "width", "height", "top", "left", "right", "bottom"}:
        return random.choice(["0", _css_length(), _css_length(), "auto", "fit-content", "min-content", "max-content"])
    if prop in {"font-size"}:
        return random.choice([_css_length(), _css_length(), "0.875rem", "1rem", "1.125rem", "1.25rem", "16px", "20px"])
    if prop in {"line-height"}:
        return random.choice(["1", "1.1", "1.2", "1.4", "1.6", "2", _css_length()])
    if prop in {"display"}:
        return random.choice(["block", "inline-block", "flex", "grid", "none"])
    if prop in {"position"}:
        return random.choice(["relative", "absolute", "fixed", "sticky"])
    if prop in {"font-family"}:
        return random.choice([
            '"Inter", system-ui, sans-serif',
            '"Roboto", Arial, sans-serif',
            '"Helvetica Neue", Helvetica, Arial, sans-serif',
            '"Segoe UI", Tahoma, sans-serif',
            '"Georgia", serif',
            '"Times New Roman", Times, serif',
            '"SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif',
            '"Fira Sans", "Open Sans", sans-serif',
            '"Source Sans Pro", Arial, sans-serif',
            '"IBM Plex Sans", sans-serif',
        ])
    if prop in {"overflow", "text-align", "font-weight", "justify-content", "align-items"}:
        return random.choice(["hidden", "left", "center", "bold", "space-between", "flex-start", "flex-end"])
    if prop in {"background", "background-color"}:
        return random.choice([
            "transparent",
            _hex_color(),
            _hex_color(),
            f"linear-gradient({random.choice(['90deg', '180deg', '135deg', 'to right', 'to bottom'])}, {random.choice([_hex_color(), _rgb_color()])}, {random.choice([_hex_color(), _rgb_color()])})",
            f"radial-gradient(circle, {random.choice([_hex_color(), _rgb_color()])}, {random.choice([_hex_color(), _rgb_color()])})",
        ])
    if prop in {"z-index"}:
        return str(random.randint(0, 999))
    if prop in {"opacity"}:
        return f"{random.uniform(0.1, 1.0):.2f}"
    if prop in {"gap", "row-gap", "column-gap", "letter-spacing", "word-spacing", "max-width", "min-width", "max-height", "min-height"}:
        return random.choice([_css_length(), _css_length(), "auto", "none"])
    if prop in {"transition"}:
        return random.choice([
            f"all {random.choice(['.15s', '.2s', '.25s', '.3s'])} {random.choice(['ease', 'ease-in', 'ease-out', 'ease-in-out', 'linear'])}",
            f"opacity {random.choice(['.15s', '.2s', '.25s', '.3s'])} {random.choice(['ease', 'ease-out', 'ease-in-out'])}",
            f"transform {random.choice(['.15s', '.2s', '.25s'])} {random.choice(['ease', 'ease-out', 'ease-in-out'])}",
            f"color {random.choice(['.15s', '.2s', '.25s'])} {random.choice(['linear', 'ease', 'ease-in'])}",
            f"background-color {random.choice(['.15s', '.2s', '.25s'])} {random.choice(['ease', 'ease-in-out'])}",
        ])
    if prop in {"box-shadow"}:
        return random.choice([
            f"0 {_css_length()} {_css_length()} {_hex_color()}",
            f"0 {_css_length()} {_css_length()} {_css_length()} {_hex_color()}",
            f"inset 0 {_css_length()} {_css_length()} {_hex_color()}",
            f"0 {_css_length()} {_css_length()} {_hex_color()}, 0 {_css_length()} {_css_length()} {_hex_color()}",
        ])
    return random.choice([fake.word(), "inherit", "unset", "initial"])


def generate_css_artifact() -> str:
    """Generate a plausible CSS snippet with common selectors and properties."""
    selectors = [_css_selector() for _ in range(random.randint(1, 3))]
    props = [
        "display",
        "margin",
        "padding",
        "font-family",
        "color",
        "background-color",
        "border-radius",
        "font-size",
        "line-height",
        "width",
        "height",
        "position",
        "overflow",
        "text-align",
        "opacity",
        "z-index",
    ]

    blocks = []
    for sel in selectors:
        decls = []
        for prop in random.sample(props, k=random.randint(3, 7)):
            decls.append(f"  {prop}: {_css_value(prop)};")
        if random.random() < 0.4:
            decls.append(f"  {random.choice(['border', 'outline'])}: {_css_length()} solid {_hex_color()};")
        if random.random() < 0.3:
            extra_prop = random.choice(['transition', 'box-shadow'])
            decls.append(f"  {extra_prop}: {_css_value(extra_prop)};")
        blocks.append(f"{sel} {{\n" + "\n".join(decls) + "\n}")

    if random.random() < 0.5:
        media_sel = random.choice(["(max-width: 768px)", "(min-width: 1024px)", "(prefers-reduced-motion: reduce)"])
        blocks.append(
            f"@media {media_sel} {{\n"
            f"  {random.choice(selectors)} {{\n"
            f"    display: {random.choice(['block', 'flex', 'grid'])};\n"
            f"    width: {random.choice(['100%', 'auto', 'min-content'])};\n"
            f"  }}\n"
            f"}}"
        )

    if random.random() < 0.4:
        blocks.append(
            ":root {\n"
            f"  --{_ident()}: {_css_value('background-color')};\n"
            f"  --{_ident()}: {_css_value('color')};\n"
            "}"
        )

    return "\n\n".join(blocks)


def _python_snippet() -> str:
    fn = _ident()
    arg = _ident()
    chance_mult = _snippet_mult()
    kind = random.choices(
        ["function", "class", "async", "script"],
        weights=[4, 2, 2, 3] if chance_mult > 0.6 else [5, 1, 1, 4],
        k=1,
    )[0]
    header = _python_import_header()
    arg_type = _python_type(allow_optional=True)
    return_type = _python_type(allow_optional=True)

    if kind == "class":
        cls = _camel()
        flow = "\n".join(_python_control_flow("        ")) if _chance(0.75, chance_mult) else ""
        return "\n".join(header + [
            f"class {cls}:",
            "    def __init__(self, enabled: bool = True):",
            "        self.enabled = enabled",
            f"        self.label = {repr(fake.word())}",
            "",
            f"    def {fn}(self, {arg}: {arg_type}) -> {return_type}:",
            f'        result = {{"status": "ok", "count": {random.randint(0, 9)}, "value": {_value("python")}}}',
            f"        if len({arg}) {_comparison_op('c')} {random.randint(0, 20)}:",
            f'            print("[{fake.word()}]", len({arg}))',
            *([flow] if flow else []),
            f'        print(f"[{fake.word()}] {fn}({{{arg}}}) -> {{result}}")',
            "        return result",
        ])

    if kind == "async":
        out = _ident()
        flow = "\n".join(_python_control_flow("    ")) if _chance(0.75, chance_mult) else ""
        return "\n".join(header + [
            f"async def {fn}({arg}: {arg_type}, limit: int = 0) -> {return_type}:",
            "    await asyncio.sleep(0)",
            f'    {out} = {{"status": "ok", "count": {random.randint(0, 9)}, "value": {_value("python")}}}',
            f"    if len({arg}) {_comparison_op('c')} {random.randint(0, 20)}:",
            f'        print("[{fake.word()}]", len({arg}))',
            *([flow] if flow else []),
            f'    print(f"[{fake.word()}] {fn}({{{arg}}}) -> {{{out}}}")',
            f"    return {out}",
        ])

    if kind == "script":
        flow = "\n".join(_python_control_flow("    ")) if _chance(0.6, chance_mult) else ""
        return "\n".join(header + [
            "if __name__ == \"__main__\":",
            "    import sys",
            f'    print("[{fake.word()}]", sys.argv[1:] if len(sys.argv) > 1 else [])',
            *([flow] if flow else []),
            f"    raise SystemExit({random.randint(0, 3)})",
        ])

    out = _ident()
    flow = "\n".join(_python_control_flow("    ")) if _chance(0.7, chance_mult) else ""
    return "\n".join(header + [
        f"def {fn}({arg}: {arg_type}, limit: int = 0) -> {return_type}:",
        f'    {out} = {{"status": "ok", "count": {random.randint(0, 9)}, "value": {_value("python")}}}',
        f"    if len({arg}) {_comparison_op('c')} {random.randint(0, 20)}:",
        f'        print("[{fake.word()}]", len({arg}))',
        *([flow] if flow else []),
        f'    print(f"[{fake.word()}] {fn}({{{arg}}}) -> {{{out}}}")',
        f"    return {out}",
    ])


def _js_snippet() -> str:
    fn = _ident()
    arg = _ident()
    chance_mult = _snippet_mult()
    kind = random.choices(
        ["function", "arrow", "class", "async", "module"],
        weights=[4, 4, 2, 2, 2] if chance_mult > 0.6 else [5, 5, 1, 1, 1],
        k=1,
    )[0]
    header = _js_import_header()
    flow = "\n".join(_js_control_flow("    ")) if _chance(0.75, chance_mult) else ""

    if kind == "arrow":
        return "\n".join(header + [
            f"const {fn} = ({arg}) => {{",
            *([flow] if flow else []),
            _block([
                f"const result = {{ ok: true, type: 'event', value: {_value('js')} }};",
                f"const size = String({arg} ?? '').length;",
                f"if (size {_comparison_op('c')} {random.randint(0, 20)}) {{",
                _block([
                    f'console.log("[{fake.word()}]", size);',
                ], indent="        "),
                "}",
                f'console.log("[{fake.word()}]", {arg}, result);',
                "return result;",
            ]),
            "};",
        ])

    if kind == "class":
        cls = _camel()
        ctor_flow = "\n".join(_js_control_flow("        ")) if _chance(0.7, chance_mult) else ""
        method_flow = "\n".join(_js_control_flow("        ")) if _chance(0.7, chance_mult) else ""
        constructor_body = "\n".join([
            f"        this.{arg} = {arg};",
            f"        this.label = '{fake.word()}';",
            *([ctor_flow] if ctor_flow else []),
        ])
        method_body = "\n".join([
            "        const result = { ok: true, value: " + _value("js") + " };",
            *([method_flow] if method_flow else []),
            f'        console.log("[{fake.word()}]", {arg}, result);',
            "        return result;",
        ])
        return "\n".join(header + [
            f"class {cls} {{",
            f"constructor({arg}) {{",
            constructor_body,
            "}",
            "",
            f"{fn}() {{",
            method_body,
            "}",
            "}",
        ])

    if kind == "async":
        return "\n".join(header + [
            f"async function {fn}({arg}) {{",
            *([flow] if flow else []),
            _block([
                f"const response = await fetch({repr(f'https://{fake.domain_name()}/api/{fake.word()}')});",
                f"const size = String({arg} ?? '').length;",
                f"if (size {_comparison_op('c')} {random.randint(0, 20)}) {{",
                _block([
                    f'console.log("[{fake.word()}]", size);',
                ], indent="        "),
                "}",
                f'console.log("[{fake.word()}]", response.status, {arg});',
                "return response.json();",
            ]),
            "}",
        ])

    if kind == "module":
        return "\n".join(header + [
            f"export const {fn} = ({arg}) => {{",
            *([flow] if flow else []),
            _block([
                "try {",
                _block([
                    f"const size = String({arg} ?? '').length;",
                    f"if (size {_comparison_op('c')} {random.randint(0, 20)}) {{",
                    _block([
                        f'console.log("[{fake.word()}]", size);',
                    ], indent="        "),
                    "}",
                    f"return {{ ok: true, value: {_value('js')} }};",
                ], indent="        "),
                "} catch (err) {",
                _block([
                    "console.error(err);",
                    "return null;",
                ], indent="        "),
                "}",
            ]),
            "};",
        ])

    return "\n".join(header + [
        f"function {fn}({arg}) {{",
        *([flow] if flow else []),
        _block([
            f"const result = {{ ok: true, type: 'event', value: {_value('js')} }};",
            f"const size = String({arg} ?? '').length;",
            f"if (size {_comparison_op('c')} {random.randint(0, 20)}) {{",
            _block([
                f'console.log("[{fake.word()}]", size);',
            ], indent="        "),
            "}",
            f'console.log("[{fake.word()}]", {arg}, result);',
            "return result;",
        ]),
        "}",
    ])


def _sql_snippet() -> str:
    chance_mult = _snippet_mult()
    table_a = _ident()
    table_b = _ident()
    alias_a = fake.word()
    alias_b = fake.word()
    col_a = _ident()
    col_b = _ident()
    metric = _ident()
    agg_alias = _ident()
    agg_kind = random.choice(["COUNT", "SUM", "AVG", "MIN", "MAX"])
    if agg_kind == "COUNT":
        agg_expr = random.choice([
            "*",
            alias_a,
            f"{alias_a}.{col_a}",
        ])
    else:
        agg_expr = random.choice([
            f"{alias_a}.{metric}",
            f"{alias_b}.{col_b}",
            f"ABS({alias_a}.{metric})",
        ])
    comp_op = _comparison_op("sql")
    text_op = _comparison_op("text")
    text_rhs = _comparison_rhs("text")
    joiner = random.choice(["AND", "OR"])
    join_type = random.choice([
        "INNER JOIN",
        "LEFT JOIN",
        "RIGHT JOIN",
        "FULL JOIN",
        "FULL OUTER JOIN",
        "CROSS JOIN",
        "NATURAL JOIN",
    ])
    pieces = [
        f"SELECT {alias_a}.{col_a}, {alias_b}.{metric}, {agg_kind}({agg_expr}) AS {agg_alias}",
        f"FROM {table_a} {alias_a}",
        (
            f"{join_type} {table_b} {alias_b} ON {alias_a}.{col_a} {comp_op} {alias_b}.{col_b}"
            if join_type not in {"CROSS JOIN", "NATURAL JOIN"}
            else f"{join_type} {table_b} {alias_b}"
        ),
    ]
    if _chance(0.8, chance_mult):
        pieces.append(
            f"WHERE {alias_a}.{metric} IS NOT NULL {joiner} {alias_b}.{col_b} {text_op} {text_rhs}"
        )
    if _chance(0.5, chance_mult):
        pieces.append(
            f"{joiner} {alias_a}.created_at {_comparison_op('sql')} '{random.randint(2020, 2026)}-01-01'"
        )
    if _chance(0.6, chance_mult):
        having_rhs = random.randint(1, 20)
        if agg_kind == "AVG":
            having_rhs = random.randint(1, 100)
        pieces.extend([
            f"GROUP BY {alias_a}.{col_a}, {alias_b}.{metric}",
            f"HAVING {agg_kind}({agg_expr}) {_comparison_op('sql')} {having_rhs}",
        ])
    else:
        pieces.append(f"GROUP BY {alias_a}.{col_a}")
    if _chance(0.7, chance_mult):
        pieces.append(f"ORDER BY {agg_alias} {random.choice(['DESC', 'ASC'])}")
    if _chance(0.5, chance_mult):
        pieces.append(f"LIMIT {random.randint(5, 500)}")
    return "\n".join(pieces) + ";"


def _bash_snippet() -> str:
    chance_mult = _snippet_mult()
    var = _ident().upper()
    path = f"/tmp/{_ident()}"
    workdir = f"/var/tmp/{_ident()}"
    use_workdir = random.random() < 0.75
    use_pattern = random.random() < 0.55
    use_mode = random.random() < 0.8
    artifact_kind = random.choice(["shell", "python", "java", "go", "c", "cpp", "utility"])
    if artifact_kind == "shell":
        runner = random.choice(["bash", "sh"])
        script = random.choice(["run.sh", f"{_ident()}.sh", f"{_ident()}.bash"])
    elif artifact_kind == "python":
        runner = random.choice(["python3", "python"])
        script = random.choice(["run.py", f"{_ident()}.py"])
    elif artifact_kind == "java":
        runner = "java"
        script = random.choice([f"{_ident()}.java", f"{_camel()}.java"])
    elif artifact_kind == "go":
        runner = "go"
        script = random.choice([f"{_ident()}.go", f"{_camel()}.go"])
    elif artifact_kind == "c":
        runner = "gcc"
        script = random.choice([f"{_ident()}.c", f"{_ident()}.h"])
    elif artifact_kind == "cpp":
        runner = random.choice(["g++", "clang++", "gcc"])
        script = random.choice([f"{_ident()}.cpp", f"{_ident()}.cc", f"{_ident()}.hpp"])
    else:
        runner = random.choice([
            "grep",
            "find",
            "xargs",
            "sort",
            "uniq",
            "awk",
            "sed",
            "tar",
            "make",
            "perl",
            _ident(),
            _ident() + random.choice(["", "", "_cmd", "-tool"]),
        ])
        script = random.choice([
            f"{_ident()}.sh",
            f"{_ident()}.txt",
            f"{_ident()}.log",
            f"{_ident()}.csv",
        ])
    script_path = random.choice([
        f"./{script}",
        f"/opt/{_ident()}/{script}",
        f"$WORKDIR/{script}",
        script,
    ])
    lang = random.choice(["en", "de", "fr", "es", "pl", "pt", "zh", "ja", "tr", "ru"])
    mode = random.choice(["fast", "safe", "debug", "train", "eval"])
    extra = random.choice([
        f"--lang {lang}",
        f"-Dlang={lang}",
        f"LANG={lang}",
        f"--locale {lang}",
    ])
    entry_candidates = [
        (
            f'{runner} {script_path} --input "$' + var + f'" {extra}'
            + (f" --mode {mode}" if use_mode else "")
            if artifact_kind != "utility" or runner in {"grep", "find", "xargs", "sort", "uniq", "awk", "sed", "tar", "make", "perl"}
            else f'{runner} "{script_path}" --input "$' + var + f'" {extra}'
        ),
        f'{"awk -f" if runner == "awk" else "sed -n"} "{script_path}"'
        + (f' "$WORKDIR"' if use_workdir else ""),
        f'find "{workdir}" -type f | head -n {random.randint(3, 10)}' if use_workdir else f'find "{path}" -type f | head -n {random.randint(3, 10)}',
        f'grep -Rni "{fake.word()}" "{workdir if use_workdir else path}" | head -n {random.randint(3, 10)}',
        f'printf "%s\\n" "{fake.word()}" "{fake.word()}" | sort | uniq -c',
        f'xargs -I{{}} echo {{}} < "{script_path}"',
    ]
    entrypoint = random.choice(entry_candidates)
    control_flow = "\n".join(_bash_control_flow("    ", path=path, workdir=workdir, var=var)) if _chance(0.75, chance_mult) else ""
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'{var}="{path}"',
    ]
    if use_workdir:
        lines.append(f'WORKDIR="{workdir}"')
    if use_pattern:
        lines.append('PATTERN="${PATTERN:-[[:alpha:]]+}"')
    if use_mode:
        lines.append('MODE="${MODE:-debug}"')
    lines.extend([
        f'echo "[INFO] starting {fake.word()}"',
    ])
    if use_workdir:
        lines.append(f'mkdir -p "${{{var}}}" "$WORKDIR"')
    else:
        lines.append(f'mkdir -p "${{{var}}}"')
    if control_flow:
        lines.append(control_flow)
    lines.append(entrypoint)
    return "\n".join(lines)


def _bash_command(op: str, *, path: str, workdir: str, var: str) -> str:
    """Return a plausible shell command using common Linux utilities."""
    candidate = random.choice([
        "grep",
        "awk",
        "sed",
        "find",
        "sort",
        "uniq",
        "cut",
        "head",
        "tail",
        "wc",
        "tr",
        "xargs",
        "printf",
        "ls",
        "du",
        "stat",
    ])
    if op == "probe":
        if candidate == "grep":
            return f'grep -En "{fake.word()}" "$PATTERN" "{path}" || true'
        if candidate == "awk":
            return f'awk \'{ "{" }print NR ":" $0{ "}" }\' "{path}" | head -n 5'
        if candidate == "sed":
            return f'sed -n "1,5p" "{path}"'
        if candidate == "find":
            return f'find "{workdir}" -type f | head -n 5'
        if candidate == "sort":
            return f'find "{workdir}" -type f | sort | uniq -c'
        if candidate == "uniq":
            return f'printf "%s\\n" "{fake.word()}" "{fake.word()}" "{fake.word()}" | sort | uniq -c'
        if candidate == "cut":
            return f'printf "%s:%s\\n" "{fake.word()}" "{fake.word()}" | cut -d: -f1'
        if candidate == "head":
            return f'head -n 5 "{path}"'
        if candidate == "tail":
            return f'tail -n 5 "{path}"'
        if candidate == "wc":
            return f'wc -l "{path}"'
        if candidate == "tr":
            return f'printf "%s\\n" "{fake.word()}" | tr "[:lower:]" "[:upper:]"'
        if candidate == "xargs":
            return f'printf "%s\\n" "{fake.word()}" "{fake.word()}" | xargs -I{{}} echo {{}}'
        if candidate == "du":
            return f'du -sh "{workdir}"'
        if candidate == "stat":
            return f'stat -c "%n %s" "{path}"'
        return f'printf "%s\\n" "{fake.word()}"'
    if op == "update":
        return random.choice([
            f'COUNT=$((COUNT + 1))',
            f'printf "[TRACE] %s\\n" "$COUNT"',
            f'printf "%s\\n" "$PATTERN"',
            f'echo "[DEBUG] {fake.word()} $COUNT"',
        ])
    if op == "echo":
        return random.choice([
            f'echo "[TRACE] {fake.word()} $COUNT"',
            f'printf "[TRACE] %s\\n" "$MODE"',
            f'printf "%s\\n" "{fake.word()}"',
        ])
    return f'printf "%s\\n" "{fake.word()}"'


def _bash_control_flow(indent: str = "    ", *, path: str, workdir: str, var: str) -> list[str]:
    """Generate a shell control-flow block with common Linux utilities."""
    mode = random.choice(["if", "loop", "switch"])
    count_limit = random.randint(1, 5)
    lines = [
        f"{indent}COUNT=0",
        f"{indent}FILES=$(find \"$WORKDIR\" -type f | head -n {count_limit})",
    ]

    if mode == "if":
        lines.extend([
            f"{indent}if {_bash_command('probe', path=path, workdir=workdir, var=var)}; then",
            f"{indent}    {_bash_command('update', path=path, workdir=workdir, var=var)}",
            f"{indent}    {_bash_command('echo', path=path, workdir=workdir, var=var)}",
            f"{indent}else",
            f"{indent}    {_bash_command('probe', path=path, workdir=workdir, var=var)}",
            f"{indent}fi",
            f"{indent}if [ \"$COUNT\" -gt {count_limit} ]; then",
            f"{indent}    printf '[INFO] count=%s\\n' \"$COUNT\"",
            f"{indent}fi",
        ])
        return lines

    if mode == "loop":
        loop_mode = random.choice(["for", "while"])
        if loop_mode == "for":
            lines.extend([
                f"{indent}for ITEM in $FILES; do",
                f"{indent}    {_bash_command('probe', path=path, workdir=workdir, var=var)}",
                f"{indent}    {_bash_command('update', path=path, workdir=workdir, var=var)}",
                f"{indent}    if [ -n \"$ITEM\" ]; then",
                f"{indent}        {_bash_command('echo', path=path, workdir=workdir, var=var)}",
                f"{indent}    fi",
                f"{indent}done",
            ])
        else:
            lines.extend([
                f"{indent}while [ \"$COUNT\" -lt {count_limit} ]; do",
                f"{indent}    {_bash_command('probe', path=path, workdir=workdir, var=var)}",
                f"{indent}    {_bash_command('update', path=path, workdir=workdir, var=var)}",
                f"{indent}done",
            ])
        return lines

    case_token = random.choice(["start", "scan", "cleanup"])
    lines.extend([
        f"{indent}case \"$MODE\" in",
        f"{indent}    start)",
        f"{indent}        {_bash_command('probe', path=path, workdir=workdir, var=var)}",
        f"{indent}        {_bash_command('update', path=path, workdir=workdir, var=var)}",
        f"{indent}        ;;",
        f"{indent}    scan)",
        f"{indent}        find \"$WORKDIR\" -type f | sort | uniq -c",
        f"{indent}        ;;",
        f"{indent}    cleanup)",
        f"{indent}        rm -rf \"$WORKDIR\"/*",
        f"{indent}        ;;",
        f"{indent}    *)",
        f"{indent}        echo \"[WARN] unknown mode: $MODE\"",
        f"{indent}        ;;",
        f"{indent}esac",
        f"{indent}case \"$COUNT\" in",
        f"{indent}    0) echo \"[INFO] empty\" ;;",
        f"{indent}    [1-{count_limit}]) echo \"[INFO] small\" ;;",
        f"{indent}    *) echo \"[INFO] large\" ;;",
        f"{indent}esac",
        f"{indent}echo \"[TRACE] {case_token}\"",
    ])
    return lines


def _go_snippet() -> str:
    chance_mult = _snippet_mult()
    fn = _camel()
    arg = _ident()
    recv = _ident()
    kind = random.choice(["struct", "interface", "func", "method"])
    ret_type = random.choice(["Config", "string", "int", "bool", "error"])
    flow = "\n".join(_go_control_flow("    ")) if _chance(0.75, chance_mult) else ""
    zero_value = {
        "Config": "Config{}",
        "string": '""',
        "int": "0",
        "bool": "false",
        "error": "nil",
    }[ret_type]

    header = _go_import_header() + [
        "type Config struct {",
        "    Name string",
        "    Count int",
        "}",
        "",
    ]

    if kind == "struct":
        header.extend([
            f"type {fn} struct {{",
            "    Enabled bool",
            "    Label string",
            "}",
        ])
        body = [
            f"func New{fn}() *{fn} {{",
            *([flow] if flow else []),
            _block([
                f'return &{fn}{{Enabled: {_value("go", "bool")}, Label: {_value("go", "string")}}}',
            ]),
            "}",
        ]
    elif kind == "interface":
        header.extend([
            f"type {fn} interface {{",
            f"    Do({arg} string) {ret_type}",
            "}",
        ])
        body = [
            f"func {fn.lower()}({arg} string) {ret_type} {{",
            *([flow] if flow else []),
            _block([
                f'fmt.Println("[trace]", {arg})',
                f"if len({arg}) {_comparison_op('c')} {random.randint(0, 20)} {{",
                _block([
                    f'fmt.Println("[debug]", len({arg}))',
                ], indent="        "),
                "}",
                f"return {zero_value}",
            ]),
            "}",
        ]
    elif kind == "method":
        header.extend([
            f"type {fn} struct {{",
            "    Count int",
            "}",
        ])
        body = [
            f"func ( {recv} *{fn} ) {arg.title()}({arg} string) ({ret_type}, error) {{",
            *([flow] if flow else []),
            _block([
                f'fmt.Printf("[debug] %s=%v\\n", "{arg}", {arg})',
                f"if {recv} == nil {{ return {zero_value}, nil }}",
                f"if len({arg}) {_comparison_op('c')} {random.randint(0, 20)} {{",
                _block([
                    f'fmt.Println("[trace]", len({arg}))',
                ], indent="        "),
                "}",
                f"return {zero_value}, nil",
            ]),
            "}",
        ]
    else:
        body = [
            f"func {fn}({arg} string) ({ret_type}, error) {{",
            *([flow] if flow else []),
            _block([
                'fmt.Println("[debug]", ' + arg + ")",
                f"if len({arg}) {_comparison_op('c')} {random.randint(0, 20)} {{",
                _block([
                    f'fmt.Println("[trace]", len({arg}))',
                ], indent="        "),
                "}",
                f"return {zero_value}, nil",
            ]),
            "}",
        ]

    return "\n".join(header + body)


def _java_snippet() -> str:
    chance_mult = _snippet_mult()
    cls = _camel()
    fn = _ident()
    class_kind = random.choice([
        "class",
        "public class",
        "final class",
        "abstract class",
        "record",
        "interface",
    ])
    return_type = random.choice([
        "String",
        "int",
        "long",
        "boolean",
        "double",
        "void",
    ])
    param_type = random.choice([
        "String",
        "int",
        "long",
        "Object",
        "List<String>",
    ])
    maybe_static = "static " if random.random() < 0.7 else ""
    header = _java_import_header()
    flow = "\n".join(_java_control_flow("        ")) if _chance(0.75, chance_mult) else ""
    extra_lines = random.choice([
        ['        System.err.println("[debug] " + input);'],
        [
            '        if (input == null) return;' if return_type == "void" else
            f'        if (input == null) return {"false" if return_type == "boolean" else "0" if return_type in {"int", "long", "double"} else "null"};'
        ],
        [
            f'        if (String.valueOf(input).length() {_comparison_op("c")} {random.randint(0, 20)}) {{',
            '            System.out.println("[trace] " + input.length());',
            '        }',
        ],
        [f'        System.out.println("[info] {fake.word()}");'],
    ])
    ret_expr = {
        "String": _value("java", "string"),
        "int": str(random.randint(-999, 999)),
        "long": f"{random.randint(1, 9_999)}L",
        "boolean": _value("java", "bool"),
        "double": f"{random.uniform(-99, 99):.2f}",
        "void": "",
    }[return_type]
    body = [
        f"{class_kind} {cls} {{",
        f"    public {maybe_static}{return_type} {fn}({param_type} input) {{",
        f'        System.out.println("[trace] " + input);',
        *([flow] if flow else []),
        *extra_lines,
        *( [f"        return {ret_expr};"] if return_type != "void" else [] ),
        "    }",
        "}",
    ]
    return "\n".join(header + body)


def _c_cpp_snippet() -> str:
    chance_mult = _snippet_mult()
    kind = random.choice(["c", "cpp"])
    fn = _ident()
    var = _ident()
    header_lines: list[str] = []
    flow = "\n".join(_c_control_flow("    ")) if _chance(0.75, chance_mult) else ""

    if kind == "c":
        scalar_type = _c_type()
        count_type = random.choice(["int", "long", "unsigned int", "size_t", "double"])
        label_type = random.choice(["char *", "const char *", "unsigned char *"])
        header_lines.extend([
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <string.h>",
        ])
        if scalar_type == "bool":
            header_lines.append("#include <stdbool.h>")
        if random.random() < 0.7:
            header_lines.append(f"#define {_ident().upper()} {random.randint(1, 100)}")
        if random.random() < 0.5:
            header_lines.append(f"typedef struct {_camel()} {{")
            header_lines.append(f"    {count_type} count;")
            header_lines.append(f"    {label_type} label;")
            header_lines.append(f"}} {_camel()};")
            header_lines.append("")
        cmp_left = f"strlen({var})" if "char" in scalar_type else var
        zero_value = _zero_value_for_type(scalar_type, language="c")
        body_lines = [f"{scalar_type} {fn}({scalar_type} {var}) {{"]
        if "char" in scalar_type:
            body_lines.append(f"    if ({var} == NULL) return {zero_value};")
        if random.random() < 0.5:
            body_lines.append(f'    const char * note = {_value("c", "string")};')
        body_lines.extend([
            f'    printf("[debug] {_c_printf_format(scalar_type)}\\n", {var});',
            f"    if ({_comparison_expr(cmp_left, style='c')}) return {zero_value};",
            *([flow] if flow else []),
            f"    return {zero_value};",
            "}",
        ])
        body = [
            *body_lines,
        ]
        if random.random() < 0.5:
            body.extend([
                "",
                "int main(int argc, char **argv) {",
                _block([
                    "if (argc < 2) {",
                    _block([
                        'fprintf(stderr, "usage: %s <input>\\n", argv[0]);',
                        "return 1;",
                    ], indent="        "),
                    "}",
                    (
                        f"return {fn}(argv[1]) ? 0 : 1;"
                        if "char" in scalar_type
                        else (
                            f"return {fn}(({scalar_type})atof(argv[1]));"
                            if scalar_type in {"double", "float"}
                            else (
                                f"return {fn}(({scalar_type})strtoul(argv[1], NULL, 10));"
                                if "unsigned int" in scalar_type or "size_t" in scalar_type or "long" in scalar_type or scalar_type == "int"
                                else f"return {fn}((bool)(atoi(argv[1]) != 0));"
                            )
                        )
                    ),
                ]),
                "}",
            ])
        return "\n".join(header_lines + [""] + body)

    elem_type = _cpp_type()
    vec_type = f"std::vector<{elem_type}>"
    header_lines.extend([
        "#include <iostream>",
        "#include <vector>",
        "#include <string>",
        "#include <map>",
    ])
    if random.random() < 0.6:
        header_lines.append("using namespace std;")
    if random.random() < 0.5:
        header_lines.extend([
            f"template <typename T>",
            f"struct {_camel()} {{",
            f"    {random.choice(['T', 'int', 'double', 'bool'])} value;",
            f"    {random.choice(['std::string', 'string'])} label;",
            "};",
            "",
        ])
    if random.random() < 0.5:
        header_lines.extend([
            f"namespace {_camel()} {{",
            f"    static const {random.choice(['int', 'long', 'double'])} {_ident()} = {random.randint(1, 999)};",
            "}",
            "",
        ])
    body = [
        f"{vec_type} {fn}(const {vec_type}& {var}) {{",
        *([flow] if flow else []),
        _block([
            f"std::cout << \"[trace] size=\" << {var}.size() << std::endl;",
            f"std::string note = {_value('cpp', 'string')};",
            f"{vec_type} out;",
            f"for (auto value : {var}) {{",
            _block([
                (
                    "out.push_back(value);"
                    if elem_type == "std::string"
                    else (
                        "out.push_back(!value);"
                        if elem_type == "bool"
                        else f"out.push_back(value + {random.randint(1, 3)});"
                    )
                ),
            ], indent="        "),
            "}",
            f"if (static_cast<int>({var}.size()) {_comparison_op('c')} {random.randint(0, 12)}) {{",
            _block([
                "return out;",
            ], indent="        "),
            "}",
            "return out;",
        ]),
        "}",
    ]
    if random.random() < 0.5:
        body.extend([
            "",
            f"class {_camel()} {{",
            "public:",
            f"    explicit {_camel()}({random.choice(['int', 'long', 'double', 'size_t'])} n) : count(n) {{}}",
            f"    {random.choice(['int', 'long', 'double', 'size_t'])} count;",
            "};",
        ])
    return "\n".join(header_lines + [""] + body)


def _rust_snippet() -> str:
    chance_mult = _snippet_mult()
    kind = random.choice(["fn", "struct", "impl"])
    name = _camel()
    fn = _ident()
    arg = _ident()
    ty = random.choice(["i32", "i64", "f32", "f64"])
    return_kind = random.choice(["result", "option", "vec"])
    ret_ty = {
        "result": "Result<i32, String>",
        "option": "Option<i32>",
        "vec": "Vec<i32>",
    }[return_kind]
    flow = "\n".join(_rust_control_flow("        " if kind == "impl" else "    ")) if _chance(0.75, chance_mult) else ""

    header = []
    if random.random() < 0.6:
        header.append("use std::collections::HashMap;")
    if random.random() < 0.4:
        header.append("use std::fmt::Debug;")

    if kind == "struct":
        return "\n".join(header + [
            f"struct {name} {{",
            f"    enabled: bool,",
            f"    count: {random.choice(['i32', 'i64', 'usize'])},",
            "}",
            "",
            f"fn new_{fn}() -> {name} {{",
            _block([
                f"{name} {{ enabled: true, count: {random.randint(0, 9)} }}",
            ]),
            "}",
        ])

    if kind == "impl":
        return "\n".join(header + [
            f"struct {name} {{",
            f"    value: {ty},",
            "}",
            "",
            f"impl {name} {{",
            f"    fn {fn}(&self, {arg}: {ty}) -> {ret_ty} {{",
            *([flow] if flow else []),
            _block([
                f"if {arg} {_comparison_op('c')} {_numeric_literal('int')} {{",
                _block([
                    f"return {'Ok(0)' if return_kind == 'result' else 'Some(0)' if return_kind == 'option' else 'vec![0, 1]'};",
                ], indent="        "),
                "}",
                (
                    "Ok(0)"
                    if return_kind == "result"
                    else "Some(0)"
                    if return_kind == "option"
                    else "vec![0, 1, 2]"
                ),
            ], indent="        "),
            "    }",
            "}",
        ])

    result_expr = (
        "Ok(0)"
        if return_kind == "result"
        else "Some(0)"
        if return_kind == "option"
        else "vec![0, 1, 2]"
    )
    fallback_expr = (
        "Err(String::from(\"oops\"))"
        if return_kind == "result"
        else "None"
        if return_kind == "option"
        else "vec![3, 4, 5]"
    )
    return "\n".join(header + [
        f"fn {fn}({arg}: {ty}) -> {ret_ty} {{",
        *([flow] if flow else []),
        _block([
            f"if {arg} {_comparison_op('c')} {_numeric_literal('int')} {{",
            _block([
                f"return {result_expr};",
            ], indent="        "),
            "}",
            f"{fallback_expr}",
        ], indent="    "),
        "}",
    ])


def _yaml_snippet() -> str:
    enabled = random.choice(["true", "false"])
    return "\n".join([
        f"name: {fake.word()}",
        f"enabled: {enabled}",
        f"timeout_ms: {random.randint(50, 5000)}",
        "logging:",
        f"  level: {random.choice(['debug', 'info', 'warn'])}",
        f"  format: {random.choice(['json', 'text'])}",
    ])


def _log_snippet() -> str:
    level = random.choice(["INFO", "WARN", "ERROR", "DEBUG"])
    rid = fake.uuid4()[:8]
    ts = (
        datetime(2026, 1, 1, tzinfo=timezone.utc)
        + timedelta(
            days=random.randint(0, 364),
            seconds=random.randint(0, 86399),
            milliseconds=random.randint(0, 999),
        )
    ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return "\n".join([
        f"{level} {ts} request_id={rid} event={fake.word()}",
        f"{level} worker={random.randint(1, 32)} status={random.choice(['ok', 'retry', 'fail'])}",
    ])


def _short_code_artifact() -> str:
    """Return a shorter code-like fragment."""
    choice = random.choice(["python", "js", "go", "java", "c", "cpp", "sql", "bash", "log", "yaml"])
    if choice == "python":
        return random.choice([
            f"def {_ident()}({ _ident() }: {_python_type(allow_optional=False)}): return {_value('python', 'number')}",
            f"{_ident()} = {_value('python', 'collection')}",
            f"if {_ident()} {_comparison_op('c')} {random.randint(0, 9)}: print({_value('python', 'string')})",
        ])
    if choice == "js":
        return random.choice([
            f"const {_ident()} = ({_ident()}) => {_value('js', 'number')};",
            f"if ({_ident()} {_comparison_op('c')} {random.randint(0, 9)}) console.log({_value('js', 'string')});",
            f"export const {_ident()} = {_value('js', 'object')};",
        ])
    if choice == "go":
        return random.choice([
            f"func {_camel()}() bool {{ return {_value('go', 'bool')} }}",
            f"var {_ident()} = {_value('go', 'collection')}",
            f"if {_ident()} {_comparison_op('c')} {random.randint(0, 9)} {{ fmt.Println({_value('go', 'string')}) }}",
        ])
    if choice == "java":
        return random.choice([
            f"int {_ident()} = {random.randint(0, 9)};",
            f"if ({_ident()} {_comparison_op('c')} {random.randint(0, 9)}) System.out.println({_value('java', 'string')});",
            f"List<String> {_ident()} = List.of({_value('java', 'string')}, {_value('java', 'string')});",
        ])
    if choice == "c":
        return random.choice([
            f"int {_ident()} = {random.randint(0, 9)};",
            f"if ({_ident()} {_comparison_op('c')} {random.randint(0, 9)}) printf({_value('c', 'string')});",
            f"char *{_ident()} = {_value('c', 'string')};",
        ])
    if choice == "cpp":
        return random.choice([
            f"auto {_ident()} = std::vector<int>{{{random.randint(0, 9)}, {random.randint(0, 9)}}};",
            f"if ({_ident()} {_comparison_op('c')} {random.randint(0, 9)}) std::cout << {_value('cpp', 'string')};",
            f"std::string {_ident()} = {_value('cpp', 'string')};",
        ])
    if choice == "sql":
        return random.choice([
            f"SELECT {_ident()} FROM {_ident()} WHERE {_ident()} {_comparison_op('sql')} {random.randint(0, 9)};",
            f"UPDATE {_ident()} SET {_ident()} = {_random.choice([0,1])};",
            f"INSERT INTO {_ident()} ({_ident()}) VALUES ({random.randint(0, 9)});",
        ])
    if choice == "bash":
        return random.choice([
            f'echo "{fake.word()}"',
            f'grep -n "{fake.word()}" "{fake.file_name()}" || true',
            f'find /tmp -type f | head -n {random.randint(1, 5)}',
        ])
    if choice == "log":
        return random.choice([
            f"INFO {fake.word()} id={fake.uuid4()[:8]}",
            f"WARN {fake.word()} retry={random.randint(1, 5)}",
            f"DEBUG {fake.word()} status={random.choice(['ok', 'fail'])}",
        ])
    return random.choice([
        f"name: {fake.word()}",
        f"enabled: {random.choice(['true', 'false'])}",
        f"count: {random.randint(0, 99)}",
    ])


def generate_code_artifact() -> str:
    """Generate a plausible-but-loose generic code artifact snippet."""
    size = random.choices(["short", "medium", "long"], weights=[0.35, 0.4, 0.25], k=1)[0]
    if size == "short":
        return _short_code_artifact()

    templates = [
        _python_snippet,
        _js_snippet,
        _c_cpp_snippet,
        _rust_snippet,
        _sql_snippet,
        _bash_snippet,
        _go_snippet,
        _java_snippet,
        _yaml_snippet,
        _log_snippet,
    ]
    if size == "medium":
        return random.choice(templates)()

    return random.choice(templates + [_python_snippet, _js_snippet, _c_cpp_snippet, _rust_snippet, _bash_snippet, _go_snippet, _java_snippet])()

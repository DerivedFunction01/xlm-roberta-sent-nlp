import random
import math

def ri(a, b):
    return random.randint(a, b)


def rc(seq):
    return random.choice(seq)


def rf(a, b):
    return round(random.uniform(a, b), ri(1, 3))


VARS = ["x", "y", "z", "n", "t", "u", "v", "w", "a", "b"]
GREEK = ["α", "β", "γ", "θ", "λ", "μ", "φ", "ω", "ρ", "σ", "δ", "ε"]
CONSTS = ["π", "e", "√2", "√3"]


def var():
    return rc(VARS)


def greek():
    return rc(GREEK)


def vorg():
    return rc(VARS + GREEK)


def coeff():
    return ri(2, 12)


def pint():
    return ri(1, 20)


def nint():
    return ri(-20, -1)


def bint():
    return ri(1, 100)


def frac():
    n, d = ri(1, 9), ri(2, 9)
    return f"{n}/{d}"


def signed():
    n = ri(1, 50)
    return f"{n}" if random.random() > 0.3 else f"-{n}"


def power(base=None, exp_range=(2, 5)):
    b = base or vorg()
    e = ri(*exp_range)
    return f"{b}^{e}"


def sqrt(inner=None):
    i = inner or rc([f"{ri(1,100)}", f"{var()}^2", f"{var()}+{ri(1,9)}"])
    return f"√({i})"


def trig():
    fn = rc(["sin", "cos", "tan", "cot", "sec", "csc"])
    arg = rc([var(), greek(), f"{ri(2,6)}{var()}", f"{var()}+{ri(1,9)}", f"{ri(1,6)}π"])
    return f"{fn}({arg})"


def log_expr():
    base = rc([str(ri(2, 10)), "e", "10"])
    arg = rc([f"{ri(1,100)}", f"{var()}^{ri(2,4)}", f"{var()}+{ri(1,9)}"])
    if base == "e":
        return f"ln({arg})"
    if base == "10":
        return f"log({arg})"
    return f"log_{base}({arg})"


def exp_expr():
    base = rc(["e", str(ri(2, 9)), greek()])
    exp = rc([var(), f"{ri(2,5)}{var()}", f"{var()}^{ri(2,3)}", f"{var()}+{ri(1,5)}"])
    return f"{base}^{exp}"


def poly_term(v=None):
    v = v or var()
    e = ri(1, 4)
    c = ri(1, 9)
    options = [f"{c}{v}^{e}", f"{v}^{e}", f"{c}{v}", v]
    return rc(options)


def poly(v=None, terms=None):
    v = v or var()
    n = terms or ri(2, 4)
    parts = [poly_term(v) for _ in range(n)]
    expr = parts[0]
    for p in parts[1:]:
        sign = rc([" + ", " - "])
        expr += sign + p
    return expr


def matrix_entry():
    return rc([str(ri(-9, 9)), var(), f"{ri(1,5)}{var()}"])


def matrix(rows=None, cols=None):
    r = rows or ri(2, 3)
    c = cols or ri(2, 3)
    rows_str = [", ".join(matrix_entry() for _ in range(c)) for _ in range(r)]
    return "[ " + " | ".join(rows_str) + " ]"


def vec(dim=None):
    d = dim or ri(2, 4)
    components = ", ".join(rc([str(ri(-9, 9)), var()]) for _ in range(d))
    return f"⟨{components}⟩"


def limit_point():
    return rc(["0", "1", "-1", "∞", "-∞", f"{ri(1,9)}", greek()])


def interval():
    a, b = ri(-10, 0), ri(1, 10)
    lb = rc(["[", "("])
    rb = rc(["]", ")"])
    return f"{lb}{a}, {b}{rb}"


def arithmetic():
    templates = [
        lambda: f"{bint()} + {bint()} = {bint()}",
        lambda: f"{bint()} - {bint()} = {signed()}",
        lambda: f"{bint()} × {bint()} = {bint()}",
        lambda: f"{bint()} ÷ {bint()} = {frac()}",
        lambda: f"{frac()} + {frac()} = {frac()}",
        lambda: f"{frac()} × {frac()} = {frac()}",
        lambda: f"{bint()}% of {bint()} = {rf(0,100)}",
        lambda: f"|{signed()}| = {ri(1,50)}",
        lambda: f"⌊{rf(-10,10)}⌋ = {ri(-10,10)}",
        lambda: f"⌈{rf(-10,10)}⌉ = {ri(-10,10)}",
        lambda: f"{ri(1,9)}! = {bint()}",
        lambda: f"gcd({bint()}, {bint()}) = {ri(1,20)}",
        lambda: f"lcm({bint()}, {bint()}) = {bint()}",
        lambda: f"{sqrt()} = {rf(0,15)}",
        lambda: f"{ri(2,9)}^{ri(2,8)} = {bint()}",
    ]
    return rc(templates)()


def algebra():
    v = var()
    templates = [
        lambda: f"{coeff()}{v} + {pint()} = {bint()}",
        lambda: f"{coeff()}{v}^2 - {pint()}{v} + {pint()} = 0",
        lambda: f"({v} + {pint()})({v} - {pint()}) = 0",
        lambda: f"{v}^2 = {bint()}",
        lambda: f"{frac()} = {coeff()}{v} - {pint()}",
        lambda: f"{poly(v, 3)} = {signed()}",
        lambda: f"|{coeff()}{v} - {pint()}| ≤ {pint()}",
        lambda: f"|{v} + {pint()}| > {pint()}",
        lambda: f"{coeff()}{v} + {pint()} ≥ {bint()}",
        lambda: f"{v}^{ri(3,5)} - {pint()} = {signed()}",
        lambda: f"({v} + {pint()})^{ri(2,4)} = {bint()}",
        lambda: f"{coeff()}/{v} = {frac()}",
        lambda: f"√({coeff()}{v} + {pint()}) = {ri(1,10)}",
        lambda: f"{exp_expr()} = {bint()}",
        lambda: f"{log_expr()} = {rf(-3,5)}",
        lambda: f"{poly(v,2)} / ({v} + {pint()}) = {v} + {pint()}",
    ]
    return rc(templates)()


def systems():
    v1, v2 = random.sample(VARS[:6], 2)
    templates = [
        lambda: f"{coeff()}{v1} + {coeff()}{v2} = {bint()}\n{coeff()}{v1} - {coeff()}{v2} = {bint()}",
        lambda: f"{v1}^2 + {v2}^2 = {bint()}\n{v1} + {v2} = {pint()}",
        lambda: f"{coeff()}{v1} + {pint()} = {coeff()}{v2}\n{v2} = {coeff()}{v1} - {pint()}",
        lambda: f"{coeff()}{v1}^2 - {coeff()}{v2} = {bint()}\n{coeff()}{v1} + {coeff()}{v2} = {bint()}",
    ]
    return rc(templates)()


def calculus_diff():
    v = rc(["x", "t", "θ"])
    f = rc(["f", "g", "h", "F"])
    inner = rc([trig(), log_expr(), exp_expr(), power(v), poly(v, 2)])
    outer = rc([trig(), log_expr(), exp_expr(), power(v), poly(v, 2)])
    templates = [
        lambda: f"d/d{v} [{inner}] = {ri(-9,9)}",
        lambda: f"{f}'({v}) = {inner}",
        lambda: f"d²/d{v}² [{inner}] = {outer}",
        lambda: f"∂/∂{v} [{poly(v,2)} + {trig()}] = {poly(v,2)}",
        lambda: f"d/d{v} [{outer} · {inner}] = {poly(v,3)}",
        lambda: f"d/d{v} [{inner} / {outer}] = {poly(v,2)}",
        lambda: f"d/d{v} [{outer}({inner})] = {poly(v,2)} · {trig()}",
        lambda: f"lim_({v}→{limit_point()}) {inner} = {rf(-10,10)}",
        lambda: f"lim_({v}→{limit_point()}) [{inner}/{outer}] = {rf(-5,5)}",
        lambda: f"{f}''({v}) + {coeff()}{f}'({v}) - {pint()}{f}({v}) = 0",
    ]
    return rc(templates)()


def calculus_integ():
    v = rc(["x", "t", "θ", "u"])
    a, b = ri(-5, 0), ri(1, 10)
    inner = rc([trig(), exp_expr(), power(v), poly(v, 2), f"{coeff()}/{v}"])
    templates = [
        lambda: f"∫ {inner} d{v} = {poly(v,2)} + C",
        lambda: f"∫_{a}^{b} {inner} d{v} = {rf(-50,50)}",
        lambda: f"∮ {trig()} d{v} = {rf(-10,10)}",
        lambda: f"∫∫ {inner} d{v} d{rc([w for w in VARS if w != v])} = {rf(-20,20)}",
        lambda: f"∫ {inner} · {trig()} d{v} = {poly(v,2)} + C",
        lambda: f"∫_{a}^∞ {exp_expr()} d{v} = {rf(0,10)}",
        lambda: f"∫ {frac()} / ({poly(v,2)}) d{v} = {log_expr()} + C",
    ]
    return rc(templates)()


def geometry():
    r = rc([str(ri(1, 20)), greek(), f"{ri(1,9)}{greek()}"])
    templates = [
        lambda: f"A = π{r}²",
        lambda: f"C = 2π{r}",
        lambda: f"A = {frac()}bh",
        lambda: f"V = {frac()}π{r}³",
        lambda: f"V = {ri(1,9)} × {ri(1,9)} × {ri(1,9)}",
        lambda: f"a² + b² = c²",
        lambda: f"{ri(1,9)}² + {ri(1,9)}² = {ri(1,200)}",
        lambda: f"A = {frac()}(b₁ + b₂)h",
        lambda: f"s = {rf(0,20)}, A = s² = {rf(0,400)}",
        lambda: f"d = √((x₂-x₁)² + (y₂-y₁)²) = {rf(0,30)}",
        lambda: f"m = (y₂-y₁)/(x₂-x₁) = {frac()}",
        lambda: f"A = {ri(1,9)} · {ri(1,9)} · {trig()} / 2 = {rf(0,100)}",
        lambda: f"sin({greek()}) = {frac()}, cos({greek()}) = {frac()}",
        lambda: f"sin²({greek()}) + cos²({greek()}) = 1",
        lambda: f"Law of Cos: c² = a² + b² - 2ab·{trig()}",
        lambda: f"Arc len = {r}·{greek()} = {rf(0,100)}",
        lambda: f"SA = 4π{r}² = {rf(10,2000)}",
        lambda: f"(x - {signed()})² + (y - {signed()})² = {ri(1,20)}²",
        lambda: f"x²/{ri(1,9)}² + y²/{ri(1,9)}² = 1",
        lambda: f"x²/{ri(1,9)}² - y²/{ri(1,9)}² = 1",
    ]
    return rc(templates)()


def trig_identities():
    a, b = greek(), greek()
    templates = [
        lambda: f"sin²({a}) + cos²({a}) = 1",
        lambda: f"tan({a}) = sin({a}) / cos({a})",
        lambda: f"sin(2{a}) = 2·sin({a})·cos({a})",
        lambda: f"cos(2{a}) = cos²({a}) - sin²({a})",
        lambda: f"sin({a}+{b}) = sin({a})cos({b}) + cos({a})sin({b})",
        lambda: f"cos({a}-{b}) = cos({a})cos({b}) + sin({a})sin({b})",
        lambda: f"tan({a}+{b}) = (tan({a})+tan({b}))/(1-tan({a})tan({b}))",
        lambda: f"sin({a}) = {frac()}  ⟹  {a} = {rf(0,6.28):.3f}",
        lambda: f"2sin²({a}) - 1 = cos(2{a})",
        lambda: f"cosh²({a}) - sinh²({a}) = 1",
        lambda: f"e^(i{a}) = cos({a}) + i·sin({a})",
    ]
    return rc(templates)()


def linear_algebra():
    templates = [
        lambda: f"A = {matrix(2,2)},  det(A) = {ri(-50,50)}",
        lambda: f"A·{vec(2)} = {vec(2)}",
        lambda: f"A = {matrix(3,3)},  rank(A) = {ri(1,3)}",
        lambda: f"A⁻¹ = {matrix(2,2)}",
        lambda: f"A^T = {matrix(ri(2,3), ri(2,3))}",
        lambda: f"‖{vec(3)}‖ = {rf(0,20)}",
        lambda: f"{vec(3)} · {vec(3)} = {ri(-50,50)}",
        lambda: f"{vec(3)} × {vec(3)} = {vec(3)}",
        lambda: f"Av = λv,  λ = {ri(-9,9)}",
        lambda: f"det({matrix(2,2)}) = {ri(-50,50)}",
        lambda: f"trace(A) = {ri(-20,20)}",
        lambda: f"proj_{vec(2)}({vec(2)}) = {vec(2)}",
    ]
    return rc(templates)()


def statistics():
    data = sorted([ri(1, 100) for _ in range(ri(4, 8))])
    data_str = ", ".join(map(str, data))
    v = vorg()
    templates = [
        lambda: f"μ = ({data_str}) / {len(data)} = {rf(10,90)}",
        lambda: f"σ² = {rf(0,100)},  σ = {rf(0,10)}",
        lambda: f"P({v} ≤ {ri(50,99)}) = {rf(0,1):.3f}",
        lambda: f"P(A ∩ B) = {rf(0,1):.3f}",
        lambda: f"P(A | B) = P(A∩B)/P(B) = {frac()}",
        lambda: f"E[{v}] = {rf(-10,50)}",
        lambda: f"Var({v}) = E[{v}²] - (E[{v}])² = {rf(0,100)}",
        lambda: f"Cov({v},{rc(VARS)}) = {rf(-10,10)}",
        lambda: f"z = ({v} - μ) / σ = {rf(-3,3):.2f}",
        lambda: f"χ² = Σ(O - E)² / E = {rf(0,30):.2f}",
        lambda: f"r = {rf(-1,1):.3f}",
        lambda: f"C({ri(5,20)},{ri(1,4)}) = {ri(1,5000)}",
        lambda: f"P({ri(5,20)},{ri(1,4)}) = {ri(1,100000)}",
        lambda: f"f({v}) = {frac()} e^(-{frac()}{v})",
    ]
    return rc(templates)()


def number_theory():
    p = rc([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
    templates = [
        lambda: f"{bint()} ≡ {ri(0,p-1)} (mod {p})",
        lambda: f"{bint()} mod {p} = {ri(0,p-1)}",
        lambda: f"φ({p}) = {p-1}",
        lambda: f"{p}^{ri(2,6)} ≡ {ri(0,p-1)} (mod {bint()})",
        lambda: f"gcd({bint()}, {bint()}) = {ri(1,30)}",
        lambda: f"{bint()} = {' × '.join(str(rc([2,3,5,7,11,13])) for _ in range(ri(2,4)))}",
        lambda: f"∑_{{k=1}}^{ri(5,50)} k = {bint()}",
        lambda: f"∏_{{k=1}}^{ri(3,8)} k = {bint()}",
    ]
    return rc(templates)()


def sequences_series():
    a, d, r = ri(1, 20), ri(1, 10), ri(2, 9)
    n = ri(5, 20)
    v = var()
    templates = [
        lambda: f"a_n = {a} + ({n}-1)·{d} = {a + (n-1)*d}",
        lambda: f"S_n = n/2·(2·{a} + (n-1)·{d}) = {bint()}",
        lambda: f"a_n = {a} · {r}^(n-1)",
        lambda: f"S_n = {a}·(1 - {r}^n)/(1-{r}) = {bint()}",
        lambda: f"S_∞ = {a}/(1 - {frac()}) = {rf(1,100)}",
        lambda: f"∑_{{n=1}}^∞ {frac()}/{poly(v,2)} = {frac()}",
        lambda: f"∑_{{n=0}}^∞ {frac()}^n = {rf(1,20)}",
        lambda: f"a_{{n+1}} = a_n + {d},  a_1 = {a}",
        lambda: f"a_{{n+2}} = a_{{n+1}} + a_n,  a_1={ri(1,5)}, a_2={ri(1,5)}",
        lambda: f"lim_(n→∞) ({frac()})^n = 0",
    ]
    return rc(templates)()


def complex_numbers():
    a, b = ri(-9, 9), ri(-9, 9)
    c, d_ = ri(-9, 9), ri(-9, 9)
    r = rf(0, 10)
    th = rc(["π/6", "π/4", "π/3", "π/2", "2π/3", "3π/4", "π"])
    templates = [
        lambda: f"({a}+{b}i) + ({c}+{d_}i) = {a+c}+{b+d_}i",
        lambda: f"({a}+{b}i)·({c}+{d_}i) = {a*c-b*d_}+{a*d_+b*c}i",
        lambda: f"|{a}+{b}i| = √{a**2+b**2} = {math.sqrt(a**2+b**2):.3f}",
        lambda: f"z = {rf(0,10):.2f}(cos({th}) + i·sin({th}))",
        lambda: f"e^(i·{th}) = cos({th}) + i·sin({th})",
        lambda: f"z² = ({a}+{b}i)² = {a**2-b**2}+{2*a*b}i",
        lambda: f"1/({a}+{b}i) = {frac()} + {frac()}i",
        lambda: f"Im({a}+{b}i) = {b}",
        lambda: f"Re({a}+{b}i) = {a}",
    ]
    return rc(templates)()


def differential_equations():
    v = rc(["x", "t"])
    f = rc(["y", "f", "u", "N"])
    templates = [
        lambda: f"d{f}/d{v} = {coeff()}{f}",
        lambda: f"{f}'' + {coeff()}{f}' + {pint()}{f} = 0",
        lambda: f"{f}'' - {pint()}{f} = {trig()}",
        lambda: f"d²{f}/d{v}² = {poly(v,2)}",
        lambda: f"d{f}/d{v} + {coeff()}{f} = {exp_expr()}",
        lambda: f"{coeff()}{f}'' + {pint()}{f}' - {pint()}{f} = {ri(0,50)}",
        lambda: f"∂²u/∂{v}² = c²·∂²u/∂t²",
        lambda: f"∂u/∂t = k·∂²u/∂{v}²",
        lambda: f"∇²{f} = 0",
        lambda: f"{f}(0) = {ri(0,10)}, {f}'(0) = {ri(0,10)}",
    ]
    return rc(templates)()


def set_theory_logic():
    v = vorg()
    templates = [
        lambda: f"|A ∪ B| = |A| + |B| - |A ∩ B|",
        lambda: f"A ⊆ B,  B ⊆ C  ⟹  A ⊆ C",
        lambda: f"(A ∪ B)ᶜ = Aᶜ ∩ Bᶜ",
        lambda: f"(A ∩ B)ᶜ = Aᶜ ∪ Bᶜ",
        lambda: f"|P(A)| = 2^{ri(2,8)} = {ri(4,256)}",
        lambda: f"A × B = {{(a,b) | a∈A, b∈B}}, |A×B| = {ri(1,10)}×{ri(1,10)}",
        lambda: f"∀{v} ∈ ℝ: {v}² ≥ 0",
        lambda: f"∃{v} ∈ ℤ: {coeff()}{v} + {pint()} = {bint()}",
        lambda: f"p ∧ q ⟹ p",
        lambda: f"¬(p ∨ q) ⟺ ¬p ∧ ¬q",
        lambda: f"p → q ≡ ¬p ∨ q",
    ]
    return rc(templates)()


DOMAINS = {
    "arithmetic": (arithmetic, 15),
    "algebra": (algebra, 15),
    "systems": (systems, 5),
    "calculus_diff": (calculus_diff, 12),
    "calculus_integ": (calculus_integ, 10),
    "geometry": (geometry, 12),
    "trig_identities": (trig_identities, 8),
    "linear_algebra": (linear_algebra, 8),
    "statistics": (statistics, 8),
    "number_theory": (number_theory, 5),
    "sequences_series": (sequences_series, 8),
    "complex_numbers": (complex_numbers, 6),
    "differential_eqs": (differential_equations, 6),
    "set_theory_logic": (set_theory_logic, 5),
}

_fns = [fn for fn, _ in DOMAINS.values()]
_weights = [w for _, w in DOMAINS.values()]


def generate_synthetic_math(domain: str | None = None) -> str:
    if domain is not None:
        if domain not in DOMAINS:
            raise ValueError(f"Unknown domain '{domain}'. Choose from: {list(DOMAINS)}")
        return DOMAINS[domain][0]()
    return random.choices(_fns, weights=_weights, k=1)[0]()


def generate_batch(n: int = 10, domain: str | None = None) -> list[str]:
    return [generate_synthetic_math(domain) for _ in range(n)]

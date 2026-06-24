r"""
Out-of-process semantic oracle: cross-check this package's differential ops
against the pip ``DifferentialAlgebra`` (the same BLAD engine via its slow
sympy string path).

The oracle runs DA in a *subprocess* and marshals operands via the sympy bridge
(BLAD jet notation ``u[x,x]`` <-> sympy ``Derivative(u(x), (x, 2))``), so the two
BLAD copies stay cleanly separated regardless of dlopen flags.  See
``~/project/docs/blad-native-binding-design.md`` (Backwards compatibility &
coexistence) for why static+hidden makes in-process defensible but
out-of-process is the belt-and-suspenders choice.

This module is only used by the ``TESTS::`` cross-checks and by callers who want
an independent confirmation; it is not on the hot path.
"""

import json
import subprocess
import sys


_ORACLE_DRIVER = r'''
import sys, json
import DifferentialAlgebra as DA
from sympy import Function, Symbol, Derivative, sympify, srepr

req = json.load(sys.stdin)
ders = [Symbol(d) for d in req["derivations"]]
heads = {h: Function(h) for h in req["heads"]}
params = [Symbol(p) for p in req["parameters"]]

def jet_to_sympy(name):
    # "u[x,x]" -> Derivative(u(x_,...), (x,2),...) ; "u" -> u(x,y,...)
    if "[" not in name:
        head = name
        if head in heads:
            return heads[head](*ders)
        return Symbol(head)
    head, rest = name.split("[", 1)
    dlist = rest.rstrip("]").split(",")
    base = heads[head](*ders)
    # count each derivation
    from collections import Counter
    c = Counter(dlist)
    pieces = []
    for d, k in c.items():
        pieces.append((Symbol(d), k))
    return Derivative(base, *pieces)

R = DA.DifferentialRing(derivations=ders,
                        blocks=[heads[h] for h in req["heads"]],
                        parameters=params)

def parse(expr_terms):
    # expr_terms: list of [coeff_str, [[name, deg], ...]]
    e = 0
    for coeff, term in expr_terms:
        m = sympify(coeff)
        for name, deg in term:
            m = m * jet_to_sympy(name) ** deg
        e = e + m
    return e

op = req["op"]
A = parse(req["A"])
if op == "differentiate":
    res = R.differentiate(A, Symbol(req["der"]))
elif op == "separant":
    res = R.separant(A)
elif op == "initial":
    res = R.initial(A)
elif op == "prem":
    B = parse(req["B"])
    h, r = R.differential_prem(A, [B])
    res = r
else:
    raise ValueError("unknown op " + op)

# return the sympy srepr; the caller compares after re-parsing both sides into
# its own engine.  Simpler & robust: return expanded string form.
from sympy import expand
print(json.dumps({"result": str(expand(res))}))
'''


def _terms_payload(dpoly):
    """Convert a DifferentialPolynomial to the oracle's term payload."""
    from . import _blad
    out = []
    for coeff, term in _blad.read_terms(dpoly._h()):
        out.append([str(coeff), [[nm, int(d)] for nm, d in term]])
    return out


def _run_oracle(req):
    proc = subprocess.run(
        [sys.executable, "-c", _ORACLE_DRIVER],
        input=json.dumps(req),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("oracle subprocess failed:\n%s" % proc.stderr)
    return json.loads(proc.stdout)["result"]


def _ring_meta(R):
    return dict(
        derivations=R.derivations(),
        heads=R.indeterminates(),
        parameters=R.parameters(),
    )


def oracle_check(op, A, B=None, der=None):
    r"""
    Run ``op`` on ``A`` (and ``B``) through both this package and the DA oracle
    and return ``(ours_sympy_str, oracle_sympy_str)`` -- two expanded sympy
    strings the caller can compare.

    EXAMPLES::

        sage: from sage_differential_polynomial import DifferentialPolynomialRing
        sage: from sage_differential_polynomial.oracle import oracle_check
        sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
        sage: A = R('u[x,x]^2 + 3*u[x] - 5'); B = R('u[x,x] - u')
        sage: ours, ora = oracle_check('prem', A, B)   # optional - DifferentialAlgebra
        sage: ours == ora                              # optional - DifferentialAlgebra
        True
    """
    R = A.parent()
    req = _ring_meta(R)
    req["op"] = op
    req["A"] = _terms_payload(A)
    if B is not None:
        req["B"] = _terms_payload(B)
    if der is not None:
        req["der"] = der
    oracle_str = _run_oracle(req)

    if op == "differentiate":
        ours = A.differentiate(der)
    elif op == "separant":
        ours = A.separant()
    elif op == "initial":
        ours = A.initial()
    elif op == "prem":
        ours = A.prem(B)
    else:
        raise ValueError("unknown op %r" % op)

    ours_str = _ours_to_sympy_str(ours)
    return ours_str, oracle_str


def _ours_to_sympy_str(dpoly):
    """Render our element as the same expanded sympy string the oracle emits."""
    from sympy import sympify, Function, Symbol, Derivative, expand
    from collections import Counter
    from . import _blad
    R = dpoly.parent()
    ders = [Symbol(d) for d in R.derivations()]
    heads = {h: Function(h) for h in R.indeterminates()}

    def jet(name):
        if "[" not in name:
            if name in heads:
                return heads[name](*ders)
            return Symbol(name)
        head, rest = name.split("[", 1)
        c = Counter(rest.rstrip("]").split(","))
        return Derivative(heads[head](*ders), *[(Symbol(d), k) for d, k in c.items()])

    e = 0
    for coeff, term in _blad.read_terms(dpoly._h()):
        m = sympify(str(coeff))
        for nm, d in term:
            m = m * jet(nm) ** d
        e = e + m
    return str(expand(e))

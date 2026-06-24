r"""
Regression / integration tests for the BLAD-backed differential-polynomial ring.

These are the durable correctness checks behind the package: round-trip
fidelity, prolongation, separant/initial, algebraic prem, the composition of
prem + prolongation into a full *differential* reduction (the operation
``regularchains-sage`` needs), and oracle agreement on small differential
systems.

Run as Sage doctests::

    sage -t src/sage_differential_polynomial/regression.py
"""

from sage.rings.rational_field import QQ

from . import _blad
from .differential_polynomial_ring import DifferentialPolynomialRing


def differential_reduce(A, reductors, derivations):
    r"""
    Full differential pseudo-reduction of ``A`` by a single ``reductor`` (the
    operation ``regularchains-sage``'s ``_native_diff_prem`` orchestrates).

    Reduces ``A`` by the reductor *and all its derivatives* that appear in
    ``A``, highest-leader-first: a derivative ``D`` strictly above the reductor's
    leader is eliminated by the reductor prolonged to leader ``D``; the leader
    itself is reduced last.  Built entirely from this package's primitives
    (``prem`` + ``differentiate``).

    INPUT:

    - ``A`` -- the dividend (a :class:`DifferentialPolynomial`).
    - ``reductors`` -- a list with one reductor (v1: single reductor).
    - ``derivations`` -- the derivation names.

    EXAMPLES:

    The full differential reduction matches BLAD's ``differential_prem`` (run
    through the package's primitives)::

        sage: from sage_differential_polynomial import DifferentialPolynomialRing
        sage: from sage_differential_polynomial.regression import differential_reduce
        sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
        sage: A = R('u[x,x,x]^2 + u[x,x] - u')
        sage: B = R('u[x,x] - u^2')
        sage: differential_reduce(A, [B], ['x'])
        4*u_x^2*u^2 + u^2 - u
    """
    if len(reductors) != 1:
        raise NotImplementedError("v1 differential_reduce takes one reductor")
    B = reductors[0]
    R = A.parent()
    bl = B.leader()                       # e.g. 'u[x,x]'
    if bl is None:
        return A
    head, mdB = _split_leader(bl, derivations)

    # appearing leaders of A that are derivatives of B's leader, highest first
    targets = []
    for nm in sorted(A._jet_names(), key=R._ranking_key):
        h2, md2 = _split_leader(nm, derivations)
        if h2 != head:
            continue
        if all(md2.get(d, 0) >= mdB.get(d, 0) for d in derivations):
            targets.append((nm, md2))

    r = A
    for nm, md2 in targets:
        if r.is_zero():
            break
        # prolong B by theta = md2 - mdB
        qD = B
        for d in derivations:
            for _ in range(md2.get(d, 0) - mdB.get(d, 0)):
                qD = qD.differentiate(d)
        r = r.prem(qD)
    return r


def _split_leader(name, derivations):
    """``u[x,x]`` -> ('u', {'x': 2}); ``u`` -> ('u', {})."""
    if "[" not in name:
        return name, {}
    head, rest = name.split("[", 1)
    md = {}
    for d in rest.rstrip("]").split(","):
        md[d] = md.get(d, 0) + 1
    return head, md


def run_regression(verbose=True):
    r"""
    Run the full regression battery, returning ``(passed, total)``.

    EXAMPLES::

        sage: from sage_differential_polynomial.regression import run_regression
        sage: p, t = run_regression(verbose=False)
        sage: p == t
        True
    """
    R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
    checks = []

    def chk(name, cond):
        checks.append((name, bool(cond)))
        if verbose:
            print(("  ok   " if cond else "  FAIL ") + name)

    p = R('u[x,x]^2 + 3*u[x] - 5')
    q = R('u[x,x] - u')

    chk("roundtrip", R(p._blad_repr()) == p)
    chk("leader", p.leader() == 'u[x,x]')
    chk("separant", str(p.separant()) == '2*u_x_x')
    chk("initial", str(R('5*u[x,x]^2*u + 3').initial()) == '5*u')
    chk("diff", str(R('u^2').differentiate('x')) == '2*u_x*u')
    chk("diff2_prolong", str(R('u^2').differentiate('x', 2)) ==
        '2*u_x_x*u + 2*u_x^2')
    chk("algebraic_prem", str(p.prem(q)) == '3*u_x + u^2 - 5')
    chk("ring_add", (p - p).is_zero())
    chk("ring_mul_nonzero", (p * q).number_of_terms() > 0)

    A = R('u[x,x,x]^2 + u[x,x] - u')
    B = R('u[x,x] - u^2')
    chk("differential_reduce",
        str(differential_reduce(A, [B], ['x'])) == '4*u_x^2*u^2 + u^2 - u')

    passed = sum(1 for _, c in checks if c)
    return passed, len(checks)

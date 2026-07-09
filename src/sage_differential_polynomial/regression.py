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

    # -- Phase A primitives ------------------------------------------------
    chk("factor_derivative", R.factor_derivative('u[x,x]') == ((('x', 2),), 'u'))
    chk("factor_derivative_head", R.factor_derivative('u') == ((), 'u'))
    chk("sort_descending",
        R.sort(['u', 'u[x,x]', 'u[x]']) == ['u[x,x]', 'u[x]', 'u'])
    chk("sort_ascending",
        R.sort(['u', 'u[x,x]', 'u[x]'], 'ascending') == ['u', 'u[x]', 'u[x,x]'])
    chk("leading_rank", str(R('u[x,x]^3 + u').leading_rank()) == 'u_x_x^3')
    chk("appearing_derivatives",
        R('u[x,x]^2 + 3*u[x] - 5').appearing_derivatives(as_names=True)
        == ['u[x,x]', 'u[x]'])

    # The full *differential* pseudo-remainder (Phase A's substantive add): a
    # fixed-point Ritt reduction by a reductor and ALL its derivatives.  Matches
    # the prem+prolong orchestration (differential_reduce) and BLAD.
    rdp, hdp = R.differential_prem(A, [B])
    chk("differential_prem", str(rdp) == '4*u_x^2*u^2 + u^2 - u')
    chk("differential_prem_matches_reduce",
        rdp == differential_reduce(A, [B], ['x']))
    # a deeper case that needs the fixed-point iteration (a reduction step
    # introduces a lower derivative that must itself be reduced):
    A2 = R('u[x,x,x,x] - u')
    B2 = R('u[x,x] + u[x] - 1')
    r2, _h2 = R.differential_prem(A2, [B2])
    chk("differential_prem_fixedpoint", str(r2) == '-u_x - u + 1')

    passed = sum(1 for _, c in checks if c)
    return passed, len(checks)


def run_gc_regression(verbose=True):
    r"""
    Regression battery for the in-epoch arena garbage collector
    (:meth:`~differential_polynomial_ring.DifferentialPolynomialRing.gc`).

    Checks that ``gc()`` invalidates the BLAD handles and rolls back the arena,
    yet every live element rematerializes transparently from its durable string
    snapshot; that arithmetic results built *before* a GC are usable after; and
    that repeated GC cycles are stable and actually reclaim the arena.

    EXAMPLES::

        sage: from sage_differential_polynomial.regression import run_gc_regression
        sage: p, t = run_gc_regression(verbose=False)
        sage: p == t
        True
    """
    from . import _blad
    R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
    checks = []

    def chk(name, cond):
        checks.append((name, bool(cond)))
        if verbose:
            print(("  ok   " if cond else "  FAIL ") + name)

    chk("checkpoint_ready", _blad.gc_checkpoint_ready())

    p = R('u[x,x]^2 + 3*u[x] - 5')
    q = R('u[x,x] - u')

    # handle-only arithmetic results created before a GC
    s = p * q + p                     # _wrap_handle result, no string yet
    d = p - q
    s_before = str(s)

    # grow the arena, snapshot usage, then GC
    acc = R.zero()
    for i in range(500):
        acc = acc + p
    used_before = _blad.stack_usage()
    R.gc()
    used_after = _blad.stack_usage()

    chk("arena_grew", used_before > 0)
    chk("arena_reclaimed", used_after < used_before)

    # elements built before the GC rematerialize transparently
    chk("rematerialize_transparent", str(s) == s_before)
    chk("premade_arith_usable_after_gc", (d == p - q))
    chk("accumulator_correct_after_gc", acc == R(500) * p)

    # a fresh handle produced *after* the GC also survives the next GC
    t2 = acc * q
    t2_before = str(t2)
    R.gc()
    chk("post_gc_result_survives_next_gc", str(t2) == t2_before)

    # repeated GC cycles are stable and keep the arena bounded
    stable = True
    for _ in range(10):
        w = R.zero()
        for _ in range(200):
            w = w + p * q
        R.gc()
        if not (w == R(200) * (p * q)):
            stable = False
            break
    chk("repeated_gc_cycles_stable", stable)

    # equality / arithmetic across a GC boundary between the two operands
    a = p * p
    R.gc()
    b = p * p
    chk("cross_gc_equality", a == b)

    passed = sum(1 for _, c in checks if c)
    return passed, len(checks)

r"""
Reduction by a SET of reductors -- the cases a single sweep gets wrong.

:meth:`~sage_differential_polynomial.differential_polynomial_ring.DifferentialPolynomialRing.differential_prem`
iterates to a fixed point over the reductor list.  This module holds the
regression cases that make that iteration necessary, and an independent
reference implementation to check it against.

Why this is a module of its own
===============================

Every case here needs **two differential indeterminates**, and v1 supports a
single live ring per session (BLAD has one global differential ring).  The
rest of the package doctests on ``DifferentialPolynomialRing(QQ, ['u'], ['x'])``,
so a two-head ring cannot share a file with them -- ``sage -t`` runs each file
in its own process, which is exactly the isolation these cases need.

One head is not merely inconvenient, it is insufficient: with a single head,
``_differential_prem_one`` is already a fixed point over every derivative of
that head, so the lowest-leader reductor alone finishes the job and no second
sweep is ever needed.  It takes two heads for one reductor's prolongation to
land on another reductor's leader.

The two cases
=============

Both reduce a polynomial by a two-element set, in both list orders, and in
both cases ONE sweep leaves a reducible jet standing.

**Case A -- the tail re-exposes a lower leader.**  ``{v[x] - v, u[x] - v[x]}``
reducing ``u[x]``.  Eliminating ``u_x`` by ``u[x] - v[x]`` introduces ``v_x``,
which ``v[x] - v`` handles -- but under the order ``[v[x] - v, u[x] - v[x]]``
that reductor has already been passed::

    sage: from sage_differential_polynomial import DifferentialPolynomialRing
    sage: R = DifferentialPolynomialRing(QQ, ['u', 'v'], ['x'])
    sage: B1 = R('v[x] - v'); B2 = R('u[x] - v[x]'); p = R('u[x]')
    sage: R.differential_prem(p, [B1, B2])[0]
    v
    sage: R.differential_prem(p, [B2, B1])[0]
    v

**Case B -- autoreduced, and highest-leader-first is the order that fails.**
``{w[x] - w, u - w}`` reducing ``u[x]``.  The set is autoreduced: ``u - w``
holds ``w``, not a proper derivative of ``w_x``, and has degree 0 in ``w_x``;
``w[x] - w`` has no ``u`` at all.  Eliminating ``u_x`` prolongs ``u - w`` to
``u_x - w_x``, introducing ``w_x`` -- and ``w_x`` outranks ``u``, so under
highest-leader-first it was reduced first and is already gone by.  This is the
case that shows no static ordering suffices; Case A fails under the ascending
order, Case B under the descending one.

Named ``u`` and ``v`` here (rather than ``u`` and ``w``) so that both cases
share the single two-head ring this module is allowed::

    sage: Ah = R('v[x] - v'); Al = R('u - v'); q = R('u[x]')
    sage: R.sort([Ah.leader(), Al.leader()], "descending")
    ['v[x]', 'u']
    sage: R.differential_prem(q, [Ah, Al])[0]        # highest-leader-first
    v
    sage: R.differential_prem(q, [Al, Ah])[0]
    v

Order independence
==================

The property the fixed point buys, and the one a single sweep violates: the
remainder does not depend on the order of the reductor list::

    sage: all(R.differential_prem(p, list(o))[0] == R.differential_prem(p, list(o[::-1]))[0]
    ....:     for o in [(B1, B2), (Ah, Al)])
    True

Agreement with an independent reference
=======================================

:func:`differential_reduce_set` iterates the package's single-reductor
reference (:func:`~sage_differential_polynomial.regression.differential_reduce`,
built from ``prem`` + ``differentiate``) rather than going through
``_differential_prem_one``, so it is an independent check of the whole path::

    sage: from sage_differential_polynomial.regression_set import differential_reduce_set
    sage: differential_reduce_set(p, [B1, B2], ['x']) == R.differential_prem(p, [B1, B2])[0]
    True
    sage: differential_reduce_set(q, [Ah, Al], ['x']) == R.differential_prem(q, [Ah, Al])[0]
    True

No static order suffices
========================

``max_passes`` counts sweeps, the last of which confirms convergence, so a
reduction needing no second productive sweep settles in 2.  Reading the
minimum off each case shows the two cases fail under OPPOSITE orders --
which is the whole argument against fixing this with a sort::

    sage: def min_passes(dividend, red):
    ....:     for n in range(1, 8):
    ....:         try:
    ....:             R.differential_prem(dividend, red, max_passes=n)
    ....:             return n
    ....:         except RuntimeError:
    ....:             pass
    sage: R.sort([B2.leader(), B1.leader()], "descending")     # case A order
    ['u[x]', 'v[x]']
    sage: min_passes(p, [B2, B1])       # descending: settles at once
    2
    sage: min_passes(p, [B1, B2])       # ascending: needs a second sweep
    3
    sage: R.sort([Ah.leader(), Al.leader()], "descending")     # case B order
    ['v[x]', 'u']
    sage: min_passes(q, [Al, Ah])       # ascending: settles at once
    2
    sage: min_passes(q, [Ah, Al])       # DESCENDING: needs a second sweep
    3

Non-termination is loud
=======================

An under-reduced remainder is never returned silently::

    sage: R.differential_prem(p, [B1, B2], max_passes=1)
    Traceback (most recent call last):
    ...
    RuntimeError: differential_prem did not converge in 1 passes...
"""

from .regression import differential_reduce


def differential_reduce_set(A, reductors, derivations, max_passes=64):
    r"""
    Reference full reduction of ``A`` by a SET of reductors.

    Iterates the single-reductor reference
    :func:`~sage_differential_polynomial.regression.differential_reduce` over
    the list until the remainder stops changing.  Built from ``prem`` +
    ``differentiate`` by way of that reference, so it shares no code with
    :meth:`DifferentialPolynomialRing._differential_prem_one` and can be used
    to cross-check it.

    INPUT:

    - ``A`` -- the dividend

    - ``reductors`` -- a list of reductors

    - ``derivations`` -- the derivation names

    - ``max_passes`` -- integer (default: 64); sweep cap, as on
      :meth:`DifferentialPolynomialRing.differential_prem`

    OUTPUT: the reduced polynomial (no cofactor -- this is a cross-check, not
    a replacement)

    EXAMPLES::

        sage: from sage_differential_polynomial import DifferentialPolynomialRing
        sage: from sage_differential_polynomial.regression_set import differential_reduce_set
        sage: R = DifferentialPolynomialRing(QQ, ['u', 'v'], ['x'])
        sage: differential_reduce_set(R('u[x]'), [R('v[x] - v'), R('u[x] - v[x]')], ['x'])
        v
    """
    r = A
    for _ in range(max_passes):
        before = r
        for B in reductors:
            r = differential_reduce(r, [B], derivations)
            if r.is_zero():
                return r
        if r == before:
            return r
    raise RuntimeError(
        "differential_reduce_set did not converge in %d passes" % max_passes)

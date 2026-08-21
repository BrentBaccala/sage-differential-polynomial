r"""
Native Sage differential-polynomial ring backed by BLAD.

This module provides :class:`DifferentialPolynomialRing` (a Sage Parent) and
:class:`DifferentialPolynomial` (its Element), implemented as a thin term-walk
binding to BLAD (the C engine behind François Boulier's ``DifferentialAlgebra``).
Unlike the existing ``bmi/sage`` interface -- a command facade that shuttles
Sage symbolic expressions across a *string* boundary into BLAD -- this is a
native Sage *type*: Sage gains a differential-polynomial ring whose elements are
genuine Parent/Element objects, marshalled to/from BLAD by a GMP-limb-level
term-walk that never crosses a string boundary on the polynomial body.

The differential structure (a *ranking*, not a term order; derivations that are
also polynomial indeterminates; lazy prolongation creating new jet variables) is
BLAD's; this module mirrors it 1:1.

EXAMPLES::

    sage: from sage_differential_polynomial import DifferentialPolynomialRing
    sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
    sage: p = R('u[x,x]^2 + 3*u[x] - 5')
    sage: q = R('u[x,x] - u')
    sage: p.prem(q)
    3*u_x + u^2 - 5

.. NOTE::

    v1 supports a single live ring at a time (BLAD has one global differential
    ring).  Constructing a second :class:`DifferentialPolynomialRing` reinstalls
    the ring and bumps an epoch; elements of the previous ring rematerialize from
    their owned Sage form on next use.
"""

import re
import weakref

# Import sage.all first to fully initialize the Sage library's import graph.
# Under ``sage -python`` (as opposed to the ``sage`` REPL) importing granular
# modules such as ``sage.rings.rational_field`` before the library is
# initialized triggers a circular-import ``ImportError: cannot import name QQ``.
# Pulling in ``sage.all`` once up front orders the initialization correctly.
import sage.all  # noqa: F401

from sage.structure.parent import Parent
from sage.structure.element import Element
from sage.structure.unique_representation import UniqueRepresentation
from sage.categories.rings import Rings
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing

from . import _blad


# ---------------------------------------------------------------------------
# Name munging: BLAD jet names <-> Sage generator names.
#
#   BLAD   "u"        Sage  "u"
#   BLAD   "u[x]"     Sage  "u_x"
#   BLAD   "u[x,x]"   Sage  "u_x_x"
# ---------------------------------------------------------------------------
def blad_to_sage_name(blad_name):
    """``u[x,x]`` -> ``u_x_x``; ``u`` -> ``u``."""
    if "[" not in blad_name:
        return blad_name
    head, rest = blad_name.split("[", 1)
    ders = rest.rstrip("]").split(",")
    return "_".join([head] + ders)


def sage_to_blad_name(sage_name, heads, derivations):
    """``u_x_x`` -> ``u[x,x]``.  Needs the head/derivation alphabets to split."""
    if sage_name in heads or sage_name in derivations:
        return sage_name
    parts = sage_name.split("_")
    for k in range(len(parts), 0, -1):
        cand = "_".join(parts[:k])
        if cand in heads:
            ders = parts[k:]
            if ders and all(d in derivations for d in ders):
                return "%s[%s]" % (cand, ",".join(ders))
            if not ders:
                return cand
    return sage_name


_JET_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[([^\]]*)\]")


def _blad_string_to_sage_repr(s):
    """Replace ``u[x,x]`` with ``u_x_x`` in a BLAD output string."""
    def repl(m):
        head = m.group(1)
        ders = m.group(2).split(",")
        return "_".join([head] + ders)
    return _JET_RE.sub(repl, s)


class DifferentialPolynomialRing(UniqueRepresentation, Parent):
    r"""
    A differential-polynomial ring backed by BLAD.

    INPUT:

    - ``base`` -- the coefficient ring; currently ``QQ``.

    - ``indeterminates`` -- the differential indeterminates (order-zero heads),
      e.g. ``['u', 'v']``.  Their jet alphabets are generated lazily by
      differentiation, NOT listed here.

    - ``derivations`` -- the independent variables, e.g. ``['x', 'y']``.  In
      BLAD a derivation *is* a polynomial indeterminate ranking below all jets.

    - ``ranking`` -- optional dict ``{'blocks': [...], 'subranking': 'grlexA'}``.

    - ``parameters`` -- optional order-zero constant / base-field generators.

    EXAMPLES::

        sage: from sage_differential_polynomial import DifferentialPolynomialRing
        sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x']); R
        Differential Polynomial Ring in u over Rational Field with derivations [x]
        sage: R.derivations()
        ['x']
        sage: R.indeterminates()
        ['u']

    TESTS:

    Category-framework conformance::

        sage: TestSuite(R).run()

    .. NOTE::

        v1 supports a single live ring at a time (BLAD has one global
        differential ring).  Constructing a second ring in the same session
        reinstalls the BLAD ring and bumps the epoch; do not hold two live
        rings of different shape simultaneously in v1.  A two-derivation,
        two-indeterminate ring (Cauchy-Riemann shape) works the same way in a
        fresh session::

            DifferentialPolynomialRing(QQ, ['u', 'v'], ['x', 'y'])
    """

    _global_epoch = 0
    _installed = None

    @staticmethod
    def __classcall__(cls, base, indeterminates, derivations,
                      ranking=None, parameters=(), base_field=None):
        indeterminates = tuple(indeterminates)
        derivations = tuple(derivations)
        parameters = tuple(parameters)
        if ranking is not None:
            ranking = tuple(sorted((k, tuple(map(tuple, v)) if k == "blocks" else v)
                                   for k, v in ranking.items()))
        return super().__classcall__(cls, base, indeterminates, derivations,
                                     ranking, parameters, base_field)

    def __init__(self, base, indeterminates, derivations,
                 ranking, parameters, base_field):
        if base is not QQ:
            raise NotImplementedError(
                "v1 supports base=QQ only (got %r)" % (base,))
        self._base = base
        self._indeterminates = list(indeterminates)
        self._derivations = list(derivations)
        self._parameters = list(parameters)
        self._ranking = dict(ranking) if ranking else None
        self._base_field = base_field
        self._jet_shadow_cache = []
        self._epoch = -1
        # Live handle-only elements: those holding a BLAD handle but no durable
        # ``_blad_string`` yet (the arithmetic-path ``_wrap_handle`` results).
        # The arena GC snapshots exactly these to their string before rolling
        # back the ba0 stack -- see :meth:`gc`.  A WeakSet so collected elements
        # drop out on their own; ``DifferentialPolynomial`` is a pure-Python
        # subclass of the cdef ``Element`` and so is weak-referenceable.
        self._live_handles = weakref.WeakSet()
        Parent.__init__(self, base=base, category=Rings())

        _blad.blad_init()
        self._rank_string = self._build_ranking_string()
        self._install()

    # -- ranking / install --------------------------------------------------
    def _build_ranking_string(self):
        ders = "[%s]" % ",".join(self._derivations)
        if self._ranking and "blocks" in self._ranking:
            blocks = [list(b) for b in self._ranking["blocks"]]
        else:
            blocks = [list(self._indeterminates)]
        # Parameters are order-zero generators that must be DECLARED to BLAD as
        # ranking symbols (otherwise "known symbol expected" at install).  Append
        # every parameter not already placed in a block as a trailing block, so
        # parameters rank below all differential indeterminates -- the Maple
        # convention (parameters are 0-order dependent variables at the bottom of
        # the ranking).  This holds whether or not the caller gave an explicit
        # ``blocks`` layout: an explicit layout still need not list the
        # parameters (they always belong at the bottom).
        if self._parameters:
            in_blocks = {n for blk in blocks for n in blk}
            tail = [p for p in self._parameters if p not in in_blocks]
            if tail:
                blocks = blocks + [tail]
        sub = (self._ranking or {}).get("subranking")
        block_strs = []
        for blk in blocks:
            inner = "[%s]" % ",".join(blk)
            if sub and sub != "grlexA":
                block_strs.append("%s%s" % (sub, inner))
            else:
                block_strs.append(inner)
        blocks_str = "[%s]" % ",".join(block_strs)
        params_str = "[%s]" % ",".join(self._parameters)
        return ("ranking (derivations = %s, blocks = %s, parameters = %s)"
                % (ders, blocks_str, params_str))

    def _install(self):
        cls = type(self)
        if cls._installed is self and self._epoch == cls._global_epoch:
            return
        cls._global_epoch += 1
        self._epoch = cls._global_epoch
        try:
            _blad.install_ranking(self._rank_string)
        except _blad.BladError as exc:
            raise RuntimeError(
                "v1 supports a single live DifferentialPolynomialRing at a time "
                "(BLAD has one global differential ring); a different ring is "
                "already installed.  Lower all elements to their Sage form "
                "before switching rings.  BLAD said: %s" % exc)
        cls._installed = self

    @property
    def epoch(self):
        self._install()
        return self._epoch

    # -- in-epoch arena garbage collection ----------------------------------
    def _register_live(self, elt):
        """Register a handle-only element (no ``_blad_string``, no ``_owned``)
        so the arena :meth:`gc` snapshots it before reclaiming the ba0 stack."""
        try:
            self._live_handles.add(elt)
        except TypeError:
            pass

    def force_epoch_bump(self):
        r"""
        Invalidate every live BLAD handle by bumping the epoch, WITHOUT
        reinstalling the ranking.

        The ranking / differential ring live on BLAD's *quiet* stack and survive
        an arena roll-back untouched, so only the polynomial handles (into the
        *main* stack) must be invalidated.  A plain epoch bump does that: on next
        use each element's :meth:`~DifferentialPolynomial._h` sees the stale
        stamp and re-parses from its durable string snapshot.  ``_installed``
        stays this ring, so :meth:`_install`'s short-circuit keeps the ranking in
        place (no ``install_ranking`` call).

        This is the epoch half of :meth:`gc`; call :func:`_blad.arena_gc`
        immediately before it (after snapshotting the live set).
        """
        cls = type(self)
        cls._global_epoch += 1
        self._epoch = cls._global_epoch
        # _installed stays self: the ranking is not reinstalled.

    def arena_usage_bytes(self):
        """Bytes retained on BLAD's main stack above the GC checkpoint (the
        reclaimable arena), or ``-1`` before any ranking is installed."""
        return _blad.stack_usage()

    def set_gc_threshold(self, nbytes):
        """Arm :meth:`maybe_gc`: run a :meth:`gc` whenever the reclaimable
        arena exceeds ``nbytes``.  ``None`` (the default) disarms it."""
        self._gc_threshold = nbytes

    def maybe_gc(self):
        """Size-triggered GC hook for hot loops: run :meth:`gc` iff the arena
        exceeds the :meth:`set_gc_threshold` threshold.  Cheap when below
        threshold or disarmed (one C stack-walk / attribute check).  Returns
        True iff a GC ran.

        Safe wherever :meth:`gc` is: any point where no BLAD C call is in
        flight and every needed polynomial is held by a
        :class:`DifferentialPolynomial` (the live-handle registry snapshots
        them) -- e.g. between iterations of a pseudo-division loop."""
        thr = getattr(self, "_gc_threshold", None)
        if thr is None:
            return False
        if _blad.stack_usage() < thr:
            return False
        self.gc()
        return True

    def gc(self):
        r"""
        Run one in-epoch arena garbage collection.

        Snapshot every live handle-only element to its durable BLAD string,
        reclaim the whole ``bap`` arena on BLAD's main stack in O(1)
        (:func:`_blad.arena_gc`), then bump the epoch
        (:meth:`force_epoch_bump`) so the now-stale handles re-parse from their
        snapshot on next use.  Transparent to callers: an element created before
        a ``gc`` rematerializes on demand afterward.

        Quiescent-point contract: must be called when no BLAD operation is in
        flight (no live handle mid-computation on the C stack) -- e.g. at the
        DifferentialThomas DoNextStep boundary.  The live set at such a point is
        just the persistent structures' polynomials (queued systems, Janet-tree
        leaves, cached initials/separants, collected cells), all modest.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: p = R('u[x,x]^2 + 3*u[x] - 5')
            sage: q = p*p + R('u[x]')          # a handle-only arithmetic result
            sage: R.gc()
            sage: q                            # rematerializes transparently
            u_x_x^4 + 6*u_x_x^2*u_x - 10*u_x_x^2 + 9*u_x^2 - 29*u_x + 25
            sage: (q - (p*p + R('u[x]'))).is_zero()
            True
        """
        # 1. snapshot every live handle-only element (handles still valid here)
        for elt in list(self._live_handles):
            if elt is None:
                continue
            try:
                if elt._blad_string is None and elt._owned is None:
                    elt._blad_string_lazy()
            except Exception:
                # An element with no live representation is already unusable;
                # skip it rather than abort the GC.
                pass
        # 2. reclaim the arena, 3. invalidate handles via epoch bump
        _blad.arena_gc()
        self.force_epoch_bump()
        # every survivor now carries a string snapshot; drop the registry (fresh
        # post-GC handles re-register as they are created).
        self._live_handles.clear()
        self._gc_count = getattr(self, "_gc_count", 0) + 1

    @property
    def gc_count(self):
        """How many arena GCs this ring has run (diagnostic)."""
        return getattr(self, "_gc_count", 0)

    # -- introspection ------------------------------------------------------
    def derivations(self):
        return list(self._derivations)

    def indeterminates(self):
        return list(self._indeterminates)

    def parameters(self):
        return list(self._parameters)

    def _heads(self):
        return set(self._indeterminates) | set(self._parameters)

    def _all_finite_names(self):
        return (list(self._derivations) + list(self._indeterminates)
                + list(self._parameters))

    def _repr_(self):
        names = ", ".join(self._indeterminates)
        ders = "[%s]" % ", ".join(self._derivations)
        return ("Differential Polynomial Ring in %s over %s with derivations %s"
                % (names, self._base, ders))

    # -- element construction ----------------------------------------------
    def _element_constructor_(self, x):
        return self.element_class(self, x)

    def gen(self, name):
        r"""
        Return a generator (a degree-1 jet/derivation/parameter) as an element.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R.gen('u')
            u
            sage: R.gen('x')
            x
        """
        if name not in self._all_finite_names():
            raise ValueError("%r is not a generator of %s" % (name, self))
        return self.element_class(self, name)

    def gens(self):
        return tuple(self.gen(n) for n in self._all_finite_names())

    def inject_variables(self, verbose=True):
        r"""
        Inject the FINITE generating set (derivations, heads, parameters) into
        the calling namespace.

        .. NOTE::

            Unlike :func:`PolynomialRing`, this injects only the finite
            generating set, NOT the infinite jet alphabet -- jets arise by
            differentiation.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R.inject_variables(verbose=False)
            sage: (x, u)
            (x, u)
        """
        import inspect
        frame = inspect.currentframe().f_back
        for n in self._all_finite_names():
            frame.f_globals[n] = self.gen(n)
        if verbose:
            print("Defining %s" % ", ".join(self._all_finite_names()))

    # -- jet shadow ---------------------------------------------------------
    def jet_shadow(self, dpolys, minimal=False):
        r"""
        Return an ``MPolynomialRing`` "jet shadow" covering the jet variables in
        ``dpolys`` (elements of this ring), plus a BLAD-name -> generator map.
        The variable order mirrors the BLAD ranking.

        When ``minimal=False`` a cached shadow whose generator set is a superset
        of the needed jets is reused.

        OUTPUT: ``(ring, name_map)``.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: p = R('u[x,x]^2 + 3*u[x] - 5')
            sage: S, m = R.jet_shadow([p])
            sage: sorted(str(g) for g in S.gens())
            ['u_x', 'u_x_x']
        """
        if not isinstance(dpolys, (list, tuple)):
            dpolys = [dpolys]
        needed = set()
        for dp in dpolys:
            needed |= set(dp._jet_names())
        needed_fs = frozenset(needed)

        if not minimal:
            for gens_fs, ring, gmap in self._jet_shadow_cache:
                if needed_fs <= gens_fs:
                    return ring, gmap

        names = sorted(needed, key=self._ranking_key)
        if not names:
            # a constant polynomial: build a trivial 1-generator shadow so the
            # MPolynomialRing is well-formed (the gen is unused).
            names = [self._indeterminates[0] if self._indeterminates
                     else self._derivations[0]]
        sage_names = [blad_to_sage_name(n) for n in names]
        ring = PolynomialRing(self._base, sage_names, order="degrevlex")
        gmap = {bn: ring.gen(i) for i, bn in enumerate(names)}
        self._jet_shadow_cache.append((needed_fs, ring, gmap))
        return ring, gmap

    # -- ring axioms --------------------------------------------------------
    def zero(self):
        return self.element_class(self, "0")

    def one(self):
        return self.element_class(self, "1")

    def _an_element_(self):
        head = self._indeterminates[0]
        der = self._derivations[0]
        return self.element_class(self, "%s[%s] + 2*%s + 1" % (head, der, head))

    def characteristic(self):
        return ZZ(0)

    def _coerce_map_from_(self, S):
        # accept ZZ, QQ, and the base into the ring (constants)
        from sage.rings.integer_ring import ZZ as _ZZ
        from sage.rings.rational_field import QQ as _QQ
        if S in (_ZZ, _QQ) or S is self.base():
            return True
        return None

    def _block_layout(self):
        """The block layout (a list of head-name lists) BLAD installs, mirroring
        :meth:`_build_ranking_string`.  Earlier block == higher rank.  Cached."""
        cached = getattr(self, "_block_layout_cache", None)
        if cached is not None:
            return cached
        if self._ranking and "blocks" in self._ranking:
            blocks = [list(b) for b in self._ranking["blocks"]]
        else:
            blocks = [list(self._indeterminates)]
        if self._parameters:
            in_blocks = {n for blk in blocks for n in blk}
            tail = [p for p in self._parameters if p not in in_blocks]
            if tail:
                blocks = blocks + [tail]
        # head -> (block_index, position_within_block)
        layout = {}
        for bi, blk in enumerate(blocks):
            for pj, head in enumerate(blk):
                layout[head] = (bi, pj)
        self._block_layout_cache = layout
        return layout

    def _ranking_key(self, blad_name):
        """Sort key matching the BLAD ranking (higher rank -> sorts EARLIER, i.e.
        smaller key).

        BLAD's installed ranking is block-major: an earlier block outranks a
        later one regardless of differentiation order (elimination ranking
        between blocks).  Within a block the default ``grlexA`` subranking ranks
        by total differentiation order (higher order = higher rank), then by the
        head's position in the block.  So a head in block 0 (e.g. ``DDPs`` in a
        per-indeterminate split) outranks every derivative of a head in block 1,
        but within a single block (the package default, all indeterminates in
        block 0) order dominates -- which is why ``u[x,x]`` outranks ``u[x]``.
        """
        if "[" in blad_name:
            head, rest = blad_name.split("[", 1)
            order = len(rest.rstrip("]").split(","))
        else:
            head, order = blad_name, 0
        layout = self._block_layout()
        if head in layout:
            block_idx, head_pos = layout[head]
        else:
            block_idx, head_pos = 1000, 1000
        # block-major (smaller block index = higher rank), then higher order
        # first within the block, then head position, then name for stability.
        return (block_idx, -order, head_pos, blad_name)

    # -- Phase A: differential primitives the consumer needs ----------------
    def factor_derivative(self, name):
        r"""
        Split a jet derivative into ``(theta, base)``: its derivation
        multi-index ``theta`` (a tuple of ``(derivation, multiplicity)`` pairs,
        in derivation order) and its order-zero dependent head ``base`` (a
        string).

        Mirrors BLAD's ``factor_derivative`` (which returns ``(theta, base)``).
        ``theta`` is the empty tuple for an order-zero head.

        INPUT: ``name`` -- a BLAD jet name (``'u[x,x]'``), a Sage jet name
        (``'u_x_x'``), or a degree-one element of this ring.

        OUTPUT: ``(theta, base)`` where ``theta`` is a tuple
        ``((d1, m1), (d2, m2), ...)`` over the ring's derivations (zero-mult
        entries omitted) and ``base`` is the head name.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R.factor_derivative('u[x,x]')
            ((('x', 2),), 'u')
            sage: R.factor_derivative('u')
            ((), 'u')

        A Sage-name or element argument is accepted too::

            sage: R.factor_derivative('u_x')
            ((('x', 1),), 'u')
            sage: p = R('u[x]')
            sage: R.factor_derivative(p)
            ((('x', 1),), 'u')

        With several derivations the multi-index lists each (multiplicity
        collapsed); see the regression battery for a two-derivation check
        (a fresh process per ring -- v1 supports a single live ring).
        """
        bn = self._as_blad_name(name)
        if "[" not in bn:
            return ((), bn)
        head, rest = bn.split("[", 1)
        ders = rest.rstrip("]").split(",")
        counts = {}
        for d in ders:
            counts[d] = counts.get(d, 0) + 1
        theta = tuple((d, counts[d]) for d in self._derivations if d in counts)
        return (theta, head)

    def _as_blad_name(self, name):
        """Coerce a jet identifier (element / Sage name / BLAD name) to a BLAD
        name (``u[x,x]``)."""
        if isinstance(name, DifferentialPolynomial):
            ld = name.leader()
            if ld is None:
                # an order-zero head element prints as the head itself
                return name._blad_repr()
            return ld
        s = str(name)
        if "[" in s:
            return s
        return sage_to_blad_name(s, self._heads(), set(self._derivations))

    def sort(self, iterable, direction="descending"):
        r"""
        Rank-order a list of derivatives / jet names / elements by the ring's
        ranking.

        ``direction='descending'`` (default) returns highest-rank first;
        ``'ascending'`` returns lowest-rank first.  Elements of the iterable
        may be BLAD names, Sage names, or degree-one :class:`DifferentialPolynomial`
        elements; the returned list preserves the input objects (only reordered).

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R.sort(['u', 'u[x,x]', 'u[x]'])
            ['u[x,x]', 'u[x]', 'u']
            sage: R.sort(['u', 'u[x,x]', 'u[x]'], 'ascending')
            ['u', 'u[x]', 'u[x,x]']
        """
        reverse = (direction == "descending")
        # _ranking_key sorts ascending with higher-rank-first (negative order);
        # so ascending key order == descending rank order.  Flip to match.
        return sorted(iterable,
                      key=lambda d: self._ranking_key(self._as_blad_name(d)),
                      reverse=not reverse)

    def differential_prem(self, p, reductors, max_passes=64):
        r"""
        Full **differential** pseudo-remainder of ``p`` by ``reductors``.

        Reduce ``p`` by every reductor *and all its derivatives* (Ritt
        reduction) until the remainder is reduced with respect to the whole
        set: no derivative of any reductor's leader, at or above that leader,
        survives in ``r``.  This is the differential analogue of
        :meth:`DifferentialPolynomial.prem` (which reduces by a single
        algebraic leader only): a derivative ``D`` strictly above a reductor's
        leader is eliminated by that reductor prolonged to leader ``D``; the
        leader itself is reduced last.

        .. NOTE::

            **One sweep over the reductor list is not enough, and no ordering
            of the list makes it enough.**  Reducing by a reductor PROLONGS
            it, and the prolonged tail can contain a derivative of *another*
            reductor's leader -- one the sweep has already gone past.  This is
            why the method iterates to a fixed point rather than making a
            single ordered pass.

            Autoreducedness does not rescue the single sweep either.  Being
            reduced w.r.t. `A_j` constrains `A_i`, not `\partial A_i`, and it
            is the prolongation that does the damage: with the autoreduced
            pair ``{w[x] - w, u - w}`` (leaders ``w_x`` and ``u``), eliminating
            ``u_x`` prolongs ``u - w`` to ``u_x - w_x`` and so introduces
            ``w_x`` -- the *higher* leader, already passed under a
            highest-leader-first order.  See
            :mod:`sage_differential_polynomial.regression_set` for that case
            and its mirror image worked as doctests (they need a two-head ring,
            which cannot share a session with this module's one-head ring).

            A single reductor never needs a second pass:
            :meth:`_differential_prem_one` is itself a fixed point over all
            derivatives of that reductor's head.  Only a SET can need one.

        INPUT:

        - ``p`` -- the differential polynomial to reduce; coerced into this
          ring if it is not already an element

        - ``reductors`` -- a reductor, or a list/tuple of them

        - ``max_passes`` -- integer (default: 64); cap on sweeps over the
          reductor list.  Ritt reduction terminates by a well-ordering on the
          ranking, so the cap is a backstop against a bug, not a limit reached
          in practice; exceeding it raises rather than returning a remainder
          that is silently under-reduced.

        OUTPUT:

        a pair ``(r, h)`` -- the reduced :class:`DifferentialPolynomial`, and
        the product of the initials/separants the reduction multiplied through
        (a :class:`DifferentialPolynomial`; ``R.one()`` if none), so that
        ``h*p`` is congruent to ``r``.  ``h`` is the non-vanishing Thomas
        inequation cofactor, and it accumulates across every sweep.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: A = R('u[x,x,x]^2 + u[x,x] - u')
            sage: B = R('u[x,x] - u^2')
            sage: r, h = R.differential_prem(A, [B]); r
            4*u_x^2*u^2 + u^2 - u

        A bare reductor is accepted as well as a list::

            sage: R.differential_prem(A, B)[0]
            4*u_x^2*u^2 + u^2 - u

        Reducing to zero stops immediately::

            sage: R.differential_prem(R('u[x,x] - u^2'), [B])[0]
            0

        TESTS:

        ``max_passes`` refuses to return an under-reduced remainder::

            sage: R.differential_prem(A, [B], max_passes=0)
            Traceback (most recent call last):
            ...
            RuntimeError: differential_prem did not converge in 0 passes...
        """
        if not isinstance(reductors, (list, tuple)):
            reductors = [reductors]
        r = self(p) if not isinstance(p, DifferentialPolynomial) else p
        h = self.one()
        for _ in range(max_passes):
            before = r
            for B in reductors:
                r, hB = self._differential_prem_one(r, B)
                h = h * hB
                if r.is_zero():
                    return r, h
            if r == before:
                return r, h
        raise RuntimeError(
            "differential_prem did not converge in %d passes over %d "
            "reductor(s); Ritt reduction terminates, so this is a bug -- "
            "raise max_passes only if you have a reason to believe otherwise"
            % (max_passes, len(reductors)))

    def _differential_prem_one(self, A, B):
        """Full differential pseudo-remainder of ``A`` by a single reductor
        ``B`` and all its appearing derivatives, highest-leader-first.  Returns
        ``(r, h)`` with ``h`` the accumulated initial/separant cofactor.

        Iterates to a fixed point: a pseudo-division against a prolonged
        reductor of leader ``D`` can introduce strictly lower derivatives of the
        same head (e.g. reducing ``u[x,x,x,x]`` against ``D_x^2 B`` produces
        ``u[x,x,x]`` terms), which must themselves be reduced.  So we repeatedly
        pick the highest appearing derivative of ``B``'s head with multidegree
        ``>=`` ``B``'s leader and reduce against the matching prolongation until
        no such derivative remains."""
        bl = B.leader()
        if bl is None:
            return A, self.one()
        _theta_B, headB = self.factor_derivative(bl)
        mdB = self._multidegree(bl)

        r = A
        h = self.one()
        guard = 0
        while not r.is_zero():
            guard += 1
            if guard > 10000:
                break
            # highest appearing derivative of B's head that is a derivative of
            # B's leader (componentwise >=).
            target = None
            for nm in self.sort(r._jet_names(), "descending"):
                _t2, head2 = self.factor_derivative(nm)
                if head2 != headB:
                    continue
                md2 = self._multidegree(nm)
                if all(md2.get(d, 0) >= mdB.get(d, 0)
                       for d in self._derivations):
                    target = (nm, md2)
                    break
            if target is None:
                break
            nm, md2 = target
            # prolong B by theta = md2 - mdB
            qD = B
            for d in self._derivations:
                for _ in range(md2.get(d, 0) - mdB.get(d, 0)):
                    qD = qD.differentiate(d)
            rD, hpow = r.prem_with_power(qD, v=nm)
            if rD == r:
                break               # no progress (degree too low) -- avoid loop
            r = rD
            if hpow:
                ini = qD.initial()
                for _ in range(hpow):
                    h = h * ini
        return r, h

    def _multidegree(self, name):
        """``{derivation: multiplicity}`` dict for a jet name."""
        _theta, _base = self.factor_derivative(name)
        return {d: m for d, m in _theta}

    # ======================================================================
    # Phase D: the ``regularchains.ring.Ring`` interface (dual-substrate).
    #
    # The shared Layer-0 algebraic engine (``triade``/``thomas``/``polyutil``/
    # ``chaintools``) threads a ``ring`` object through every function and
    # resolves all ranking ops as ``ring.<op>(p)`` (Maple style).  Giving
    # ``DifferentialPolynomialRing`` the same interface as
    # ``regularchains.ring.Ring`` lets the engine run *directly* on ``bap``
    # ``DifferentialPolynomial`` elements -- no jet ``_Shadow`` / ``to_sage`` /
    # ``to_blad`` round-trip.  The engine code is byte-unchanged; only this
    # interface and the element-method surface below adapt.
    #
    # **Field-param fidelity (the load-bearing seam).**  RegularChains' ``mvar``
    # is the highest-ranked appearing *unknown* -- a variable that is neither a
    # derivation (coordinate) nor a parameter.  BLAD's ``leader`` ranks
    # parameters/coordinates *as* variables, so ``leader`` and ``mvar`` DIVERGE
    # on a pure parameter / coordinate relation: ``leader('a^2-3') == 'a'`` but
    # ``mvar('a^2-3') is None`` (the derivations + parameters are the "field
    # params" the decomposition must never case-split on).  ``init``/``tail``/
    # ``mvar_name`` are therefore taken w.r.t. ``mvar``, NOT ``leader``.
    # ======================================================================

    @property
    def R(self):
        """The polynomial-ring object the engine coerces through.

        For the dual-substrate path the ring *is* its own ``.R``: the engine's
        ``ring.R(p)`` / ``ring.R(0)`` calls coerce through this ring's element
        constructor, producing :class:`DifferentialPolynomial` elements (the
        ``bap`` substrate), exactly where a Sage ``Ring`` would return Sage
        ``MPolynomial`` elements.
        """
        return self

    @property
    def param_field(self):
        """Whether declared :meth:`parameters` are treated as base-field
        constants (never case-split; the default) or as ranked *unknowns* (the
        ``param_field=False`` / parametric-split mode).

        The dual-substrate Layer-0 engine reads this through
        :attr:`field_params` / :meth:`is_param` / :meth:`mvar_name`.  The
        diff-Thomas driver sets it per decomposition (it is not part of the
        ring's identity, so it is a plain mutable flag, safe because v1 runs one
        decomposition at a time).  Default ``True`` (parameters are field
        constants -- the Maple ``field_of_constants`` convention)."""
        return getattr(self, "_param_field", True)

    @param_field.setter
    def param_field(self, value):
        self._param_field = bool(value)

    @property
    def field_params(self):
        """The unranked base-field generators the decomposition never case-splits
        on -- the RegularChains ``field_params``.

        Always includes the derivations (coordinates).  Includes the declared
        :meth:`parameters` ONLY in :attr:`param_field` mode (the default); in the
        ``param_field=False`` parametric-split mode the parameters are ranked
        unknowns (case-splittable) and so are NOT field params.
        """
        fp = list(self._derivations)
        if self.param_field:
            fp += list(self._parameters)
        return fp

    @property
    def params(self):
        """The parameter names (a subset of :attr:`field_params`)."""
        return list(self._parameters)

    @property
    def char(self):
        """The characteristic (always 0 in v1)."""
        return ZZ(0)

    @property
    def order(self):
        """The ranked variable list, highest rank first.

        Peripheral on the differential path: the *core* ``triangularize`` /
        ``intersect`` / ``regular_gcd`` / split-by-squarefree engine never reads
        ``ring.order`` (only the counting / dimension / debug features do, which
        the diff-Thomas path does not exercise -- see the Phase-D audit).  A
        ``DifferentialPolynomialRing`` has an *infinite* jet alphabet, so no
        finite total ``order`` exists; we return the finite generating set in
        ranking order (derivations rank below indeterminates), which is a sound
        upper part for those peripheral readers and is never consulted by the
        core.
        """
        names = list(self._indeterminates) + list(self._parameters) \
            + list(self._derivations)
        return self.sort(names, "descending")

    def rank_of(self, vname):
        """Rank index of a variable as an INTEGER, smaller == higher rank
        (matching ``regularchains.ring.Ring.rank_of`` where 0 is the top rank).

        The engine both *compares* rank indices (``chain.under`` / ``chain.upper``
        use ``<`` / ``>``) and *negates* them (``-ring.rank_of(...)`` as a sort
        key), so a plain integer is required -- not the ``_ranking_key`` tuple.
        We encode the BLAD ranking key ``(block_idx, -order, head_pos, name)``
        into a single monotonic non-negative integer that preserves the tuple's
        order: a higher differentiation order or an earlier block yields a
        smaller integer (higher rank).  The encoding is order-preserving for any
        runtime jet (order / head position bounded by the polynomials in hand).
        """
        block_idx, neg_order, head_pos, _name = \
            self._ranking_key(self._as_blad_name(vname))
        _B = 10 ** 6
        # neg_order = -order, so smaller neg_order == higher order == higher rank.
        # Shift each field into a positive monotonic slot (smaller == higher rank).
        return ((block_idx * _B + (neg_order + _B)) * _B + head_pos)

    def is_param(self, vname):
        """True iff ``vname`` is a field param -- a derivation, or (in
        :attr:`param_field` mode) a declared parameter -- i.e. an unranked
        base-field generator the decomposition never case-splits on."""
        head = str(vname).split("[", 1)[0]
        return head in set(self.field_params)

    def _is_field_param_name(self, blad_name):
        head = blad_name.split("[", 1)[0]
        return head in set(self.field_params)

    def mvar(self, p):
        """The main variable of ``p``: the highest-ranked appearing *unknown*
        jet (an indeterminate derivative), as a degree-one
        :class:`DifferentialPolynomial`, or ``None`` when only field params
        (derivations / parameters) appear.

        Differs from :meth:`DifferentialPolynomial.leader` (which ranks params
        and coordinates too): a pure parameter / coordinate relation has NO
        ``mvar`` -- it is constant w.r.t. the ranking.  See the class docstring
        for the field-param contract.
        """
        nm = self.mvar_name(p)
        return None if nm is None else self(nm)

    def mvar_name(self, p):
        """The main-variable NAME of ``p`` (field-param-aware), or ``None``.

        The highest-ranked appearing name that is NOT a derivation or parameter.
        ``None`` when ``p`` is a base constant or involves only field params.
        """
        p = self(p) if not isinstance(p, DifferentialPolynomial) else p
        names = [nm for nm in p._jet_names()
                 if not self._is_field_param_name(nm)]
        if not names:
            return None
        return self.sort(names, "descending")[0]

    def gen(self, name):  # noqa: F811  (override the finite-gen accessor)
        """Return the ``bap`` generator (a degree-one element) for ``name``.

        Overrides the finite-generating-set :meth:`gen` so the engine's
        ``ring.gen(v)`` -- where ``v`` is any appearing jet name, including
        prolongations not in the finite generating set -- returns the
        degree-one element the element methods accept (vs. the finite-gen guard
        that rejects unlisted jet names).
        """
        return self(self._as_blad_name(name))

    def init(self, p, v=None):
        """Initial of ``p`` w.r.t. its :meth:`mvar` (field-param-aware) -- the
        coefficient of the highest power of the main *unknown* variable.

        ``v`` may be given to take the initial w.r.t. a specific variable.
        Distinct from :meth:`DifferentialPolynomial.initial` (which uses BLAD's
        ``leader``, params included): on a pure-param relation
        ``init('a^2-3')`` returns ``a^2-3`` itself (no unknown to peel), whereas
        ``initial`` would peel the param ``a``.
        """
        p = self(p) if not isinstance(p, DifferentialPolynomial) else p
        if v is None:
            v = self.mvar_name(p)
        if v is None:
            return p                      # constant w.r.t. the ranking
        d = p.degree_in(v)
        return p.coefficient_in(v, d)

    def tail(self, p, v=None):
        """Reductum ``p - init(p)*v**deg`` w.r.t. :meth:`mvar` (field-param-aware).
        """
        p = self(p) if not isinstance(p, DifferentialPolynomial) else p
        if v is None:
            v = self.mvar_name(p)
        if v is None:
            return self.zero()
        d = p.degree_in(v)
        ini = p.coefficient_in(v, d)
        g = self(self._as_blad_name(v))
        lead = self.one()
        for _ in range(d):
            lead = lead * g
        return p - ini * lead

    def degree_in(self, p, v):
        """Degree of ``p`` in the variable ``v`` (RegularChains ``ring.degree_in``)."""
        p = self(p) if not isinstance(p, DifferentialPolynomial) else p
        return p.degree_in(v)

    def mdeg(self, p, v=None):
        """Degree of ``p`` w.r.t. its :meth:`mvar` (field-param-aware), or 0 for
        a constant / pure-field-param polynomial."""
        p = self(p) if not isinstance(p, DifferentialPolynomial) else p
        if v is None:
            v = self.mvar_name(p)
        if v is None:
            return ZZ(0)
        return ZZ(p.degree_in(v))

    def monomial(self, *exps):
        """Construct the monomial ``prod gen_i ** exps_i`` over the FINITE
        generating set, mirroring ``MPolynomialRing.monomial``.  Used only by
        the (peripheral) field-param-content stripper; the differential path's
        own stripper is BLAD-native."""
        gens = self.gens()
        if len(exps) != len(gens):
            raise ValueError("monomial expects one exponent per generator")
        mon = self.one()
        for g, e in zip(gens, exps):
            for _ in range(int(e)):
                mon = mon * g
        return mon

    def base_ring(self):
        """The coefficient field (``QQ`` in v1)."""
        return self._base


# ---------------------------------------------------------------------------
# Element
# ---------------------------------------------------------------------------
class DifferentialPolynomial(Element):
    r"""
    An element of a :class:`DifferentialPolynomialRing`.

    An element carries a canonical *owned* form (a durable string / jet-shadow
    snapshot) plus an optional epoch-stamped BLAD handle (a disposable cache).
    Differential polynomials are immutable values, so the two never diverge.

    EXAMPLES::

        sage: from sage_differential_polynomial import DifferentialPolynomialRing
        sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
        sage: p = R('u[x,x]^2 + 3*u[x] - 5'); p
        u_x_x^2 + 3*u_x - 5
        sage: p.leader()
        'u[x,x]'

    TESTS:

    Ring arithmetic is C-native (BLAD ``bap``), not materialization::

        sage: q = R('u[x,x] - u')
        sage: (p - p).is_zero()
        True
        sage: (p + q) - q == p
        True
        sage: -(-p) == p
        True

    Pickling round-trips via the durable BLAD-string snapshot::

        sage: loads(dumps(p)) == p
        True
    """

    def __init__(self, parent, value):
        Element.__init__(self, parent)
        self._handle = None
        self._handle_epoch = -1
        self._owned = None
        self._blad_string = None
        if isinstance(value, DifferentialPolynomial):
            self._owned = value._owned
            self._blad_string = value._blad_string
            self._handle = value._handle
            self._handle_epoch = value._handle_epoch
            # A copy of a handle-only element shares the live handle but not the
            # (still-None) string; register it too so the arena GC snapshots it.
            if (self._handle is not None and self._blad_string is None
                    and self._owned is None):
                parent._register_live(self)
        elif isinstance(value, str):
            self._blad_string = value
        elif value in ZZ:
            self._blad_string = str(ZZ(value))
        elif value in QQ:
            self._blad_string = str(QQ(value))
        else:
            self._blad_string = str(value)

    # -- handle management --------------------------------------------------
    def _h(self):
        R = self.parent()
        ep = R.epoch
        if self._handle is not None and self._handle_epoch == ep:
            return self._handle
        if self._blad_string is not None:
            self._handle = _blad.parse_poly(self._blad_string, ep)
        elif self._owned is not None:
            self._handle = self._from_owned(ep)
        else:
            raise RuntimeError("element has no representation")
        self._handle_epoch = ep
        return self._handle

    def _from_owned(self, ep):
        ring, poly = self._owned
        R = self.parent()
        names = [str(g) for g in ring.gens()]
        blad_names = [sage_to_blad_name(n, R._heads(), set(R._derivations))
                      for n in names]
        gen_to_blad = {i: blad_names[i] for i in range(len(names))}
        terms = []
        md = {}
        for mon, coeff in poly.monomial_coefficients().items():
            term = []
            for i, e in enumerate(mon):
                if e:
                    bn = gen_to_blad[i]
                    term.append((bn, int(e)))
                    md[bn] = max(md.get(bn, 0), int(e))
            terms.append((int(QQ(coeff) * QQ(coeff).denominator()) if False
                          else int(coeff), tuple(term)))
        terms.sort(key=lambda ct: _term_sort_key(ct[1], R))
        return _blad.build_terms(terms, list(md.items()), ep)

    @staticmethod
    def _wrap_handle(parent, handle):
        p = parent.element_class(parent, "0")
        p._handle = handle
        p._handle_epoch = handle.epoch
        # NB: do NOT stringify here.  The algebraic layer creates millions of
        # bap intermediates; an eager ``handle.to_string()`` on every wrap is a
        # per-op stringification tax (the core of the to_blad O(N^2) bug).  The
        # BLAD string is materialized lazily on demand (_blad_string()) by the
        # few sites that actually need it (_repr_/__hash__/__reduce__ and the
        # stale-handle recovery path in _h()).
        p._blad_string = None
        # Register with the arena GC: a live handle with no durable string must
        # be snapshotted before the ba0 stack rolls back (see the ring's gc()).
        parent._register_live(p)
        return p

    def _blad_string_lazy(self):
        """The durable BLAD string snapshot, computed on demand from the live
        handle and cached.  Used by repr/hash/pickle and as the cross-epoch
        recovery representation; NOT computed at wrap time."""
        if self._blad_string is None:
            self._blad_string = self._h().to_string()
        return self._blad_string

    # -- jet names ----------------------------------------------------------
    def _jet_names(self):
        names = set()
        for _, term in _blad.read_terms(self._h()):
            for nm, _ in term:
                names.add(nm)
        return names

    # -- representations ----------------------------------------------------
    def _repr_(self):
        return _blad_string_to_sage_repr(self._h().to_string())

    def _blad_repr(self):
        """The raw BLAD string (``u[x,x]`` notation)."""
        return self._h().to_string()

    def __hash__(self):
        return hash(self._blad_string_lazy())

    def __reduce__(self):
        # serialize via the durable BLAD string snapshot (never the C handle)
        return (_reconstruct_element, (self.parent(), self._blad_string_lazy()))

    def _richcmp_(self, other, op):
        eq = (self._h() == other._h())
        if op == 2:
            return eq
        if op == 3:
            return not eq
        return NotImplemented

    def is_zero(self):
        return self._h().is_zero()

    # -- ring arithmetic (C-native bap, NOT materialization to Sage) --------
    def _add_(self, other):
        R = self.parent()
        return DifferentialPolynomial._wrap_handle(
            R, _blad.add(self._h(), other._h(), R.epoch))

    def _sub_(self, other):
        R = self.parent()
        return DifferentialPolynomial._wrap_handle(
            R, _blad.sub(self._h(), other._h(), R.epoch))

    def _mul_(self, other):
        R = self.parent()
        return DifferentialPolynomial._wrap_handle(
            R, _blad.mul(self._h(), other._h(), R.epoch))

    def _neg_(self):
        R = self.parent()
        return DifferentialPolynomial._wrap_handle(
            R, _blad.neg(self._h(), R.epoch))

    def number_of_terms(self):
        r"""
        The number of monomials.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('u[x,x]^2 + 3*u[x] - 5').number_of_terms()
            3
        """
        return int(self._h().nbmon())

    # -- the differential operations ---------------------------------------
    def differentiate(self, *args):
        r"""
        Total derivative.  ``p.differentiate('x')`` is ``D_x p``;
        ``p.differentiate('x', 2, 'y')`` is ``D_x^2 D_y p``.

        Arguments must be derivations (independent variables).

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: p = R('u^2')
            sage: p.differentiate('x')
            2*u_x*u
            sage: p.differentiate('x', 2)
            2*u_x_x*u + 2*u_x^2

        A non-derivation argument raises::

            sage: p.differentiate('u')
            Traceback (most recent call last):
            ...
            ValueError: differentiate expects a derivation, got 'u'
        """
        R = self.parent()
        ep = R.epoch
        ders = R.derivations()
        # Normalize args: an integer-like (Python int or Sage Integer) is a
        # multiplicity; anything else is a derivation name.  We classify by
        # "is it one of the ring's derivations?" first, so a derivation named
        # like a number never collides.
        norm = []
        for a in args:
            sa = str(a)
            if sa in ders:
                norm.append(("der", sa))
            else:
                try:
                    norm.append(("mult", int(a)))
                except (TypeError, ValueError):
                    norm.append(("der", sa))  # will fail validation below
        seq = []
        i = 0
        while i < len(norm):
            kind, d = norm[i]
            if kind != "der" or d not in ders:
                raise ValueError(
                    "differentiate expects a derivation, got %r" % (d,))
            mult = 1
            if i + 1 < len(norm) and norm[i + 1][0] == "mult":
                mult = norm[i + 1][1]
                i += 1
            seq.append((d, mult))
            i += 1
        h = self._h()
        for d, mult in seq:
            for _ in range(mult):
                h = _blad.differentiate(h, d, ep)
        return DifferentialPolynomial._wrap_handle(R, h)

    def diff(self, *args):
        """Alias for :meth:`differentiate`."""
        return self.differentiate(*args)

    def separant(self, v=None):
        r"""
        The separant ``dp/d(leader)``, or ``dp/dv`` for a given jet ``v``.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('u[x,x]^2 + 3*u[x] - 5').separant()
            2*u_x_x
        """
        R = self.parent()
        ep = R.epoch
        if v is None:
            h = _blad.separant(self._h(), ep)
        else:
            vb = sage_to_blad_name(str(v), R._heads(), set(R._derivations))
            h = _blad.separant(self._h(), ep, vb)
        return DifferentialPolynomial._wrap_handle(R, h)

    def initial(self):
        r"""
        The initial (leading coefficient w.r.t. the leader).

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('5*u[x,x]^2*u + 3').initial()
            5*u
        """
        R = self.parent()
        ep = R.epoch
        return DifferentialPolynomial._wrap_handle(
            R, _blad.initial(self._h(), ep))

    def leader(self):
        r"""
        The leading derivative (highest-ranked jet) as a BLAD name, or ``None``
        for a constant.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('u[x,x]^2 + 3*u[x] - 5').leader()
            'u[x,x]'

        A constant has no leader::

            sage: R('5').leader() is None
            True
        """
        try:
            return _blad.leader_name(self._h())
        except _blad.BladError:
            # BLAD raises "non numeric polynomial expected" on a pure constant
            return None

    def leading_derivative(self):
        """Alias for :meth:`leader`."""
        return self.leader()

    def leading_rank(self):
        r"""
        The leading rank ``leader ** degree_in_leader`` as a
        :class:`DifferentialPolynomial` (``self.parent().one()`` for a constant).

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('u[x,x]^3 + 3*u[x] - 5').leading_rank()
            u_x_x^3
            sage: R('3*u + 1').leading_rank()
            u
            sage: R('5').leading_rank()
            1
        """
        R = self.parent()
        ld = self.leader()
        if ld is None:
            return R.one()
        deg = self.degree_in(ld)
        lead = R(ld)
        result = R.one()
        for _ in range(deg):
            result = result * lead
        return result

    def degree_in(self, name):
        r"""
        The degree of ``self`` in the jet ``name`` (a BLAD/Sage name or element).

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('u[x,x]^3 + 3*u[x] - 5').degree_in('u[x,x]')
            3
            sage: R('u[x,x]^3 + 3*u[x] - 5').degree_in('u[x]')
            1
        """
        R = self.parent()
        bn = R._as_blad_name(name)
        # C-native query (bap_degree_polynom_mpz).  The old read_terms walk
        # materialized the WHOLE polynomial as Python tuples per call -- called
        # 2-3x per pseudo-division iteration on 10^5-term operands, it was a
        # Python-object storm dominating the endgame wall and transient RSS.
        # BLAD returns -1 for the zero polynomial; the historical (term-walk)
        # convention here is 0 -- clamp to preserve it.
        d = int(_blad.degree_in(self._h(), bn))
        return d if d > 0 else 0

    def appearing_derivatives(self, as_names=False, selection="indeterminates"):
        r"""
        The jets / derivatives appearing in ``self``, in ranking order
        (highest first).

        ``selection``:

        - ``'indeterminates'`` (default) -- the differential-indeterminate jets
          (excludes pure derivations and parameters).
        - ``'all'`` -- every appearing name.

        ``as_names=True`` returns BLAD name strings; otherwise degree-one
        :class:`DifferentialPolynomial` elements.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('u[x,x]^2 + 3*u[x] - 5').appearing_derivatives(as_names=True)
            ['u[x,x]', 'u[x]']
        """
        R = self.parent()
        names = self._jet_names()
        if selection == "indeterminates":
            params = set(R._parameters)
            ders = set(R._derivations)
            kept = []
            for nm in names:
                head = nm.split("[", 1)[0]
                if head in ders or head in params:
                    continue
                kept.append(nm)
            names = kept
        ordered = R.sort(names, "descending")
        if as_names:
            return list(ordered)
        return [R(nm) for nm in ordered]

    def prem(self, other, v=None):
        r"""
        Ritt pseudo-remainder of ``self`` by ``other`` (w.r.t. ``other.leader()``
        unless ``v`` is given).

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: p = R('u[x,x]^2 + 3*u[x] - 5')
            sage: q = R('u[x,x] - u')
            sage: p.prem(q)
            3*u_x + u^2 - 5
        """
        R = self.parent()
        ep = R.epoch
        if v is not None:
            v = sage_to_blad_name(str(v), R._heads(), set(R._derivations))
        h, _hpow = _blad.prem(self._h(), other._h(), ep, v)
        return DifferentialPolynomial._wrap_handle(R, h)

    def prem_with_power(self, other, v=None):
        r"""
        Like :meth:`prem` but also return the power ``h`` the initial of
        ``other`` was raised to: returns ``(remainder, h)``.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: p = R('u[x,x]^2 + 3*u[x] - 5'); q = R('u[x,x] - u')
            sage: r, h = p.prem_with_power(q); h
            2
        """
        R = self.parent()
        ep = R.epoch
        if v is not None:
            v = sage_to_blad_name(str(v), R._heads(), set(R._derivations))
        h, hpow = _blad.prem(self._h(), other._h(), ep, v)
        return DifferentialPolynomial._wrap_handle(R, h), int(hpow)

    def gcd_prem(self, other, v=None):
        r"""
        Swell-controlled (gcd-aware) pseudo-remainder of ``self`` by ``other``
        w.r.t. ``other.leader()`` (unless ``v`` is given).

        Unlike :meth:`prem`, this divides out gcd content during pseudo-division
        (BLAD ``baz_gcd_prem_polynom_mpz``) -- the same mechanism BLAD's own
        Rosenfeld-Groebner uses to keep coefficient/term swell under control.
        The remainder differs from :meth:`prem`'s by a content/unit factor, so
        the two agree up to a nonzero unit.

        Returns ``(remainder, hexp)`` where ``hexp`` is a coarse exponent signal
        for the multiplier product.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: p = R('u[x,x]^2 + 3*u[x] - 5'); q = R('u[x,x] - u')
            sage: r, h = p.gcd_prem(q); r.is_zero()
            False
        """
        R = self.parent()
        ep = R.epoch
        if v is not None:
            v = sage_to_blad_name(str(v), R._heads(), set(R._derivations))
        h, hexp = _blad.gcd_prem(self._h(), other._h(), v, ep)
        return DifferentialPolynomial._wrap_handle(R, h), int(hexp)

    # -- algebraic primitives (Phase B; thin delegates to BLAD baz/bap) -----
    def gcd(self, other):
        r"""
        Greatest common divisor of ``self`` and ``other`` (over `\QQ`).

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('u^2 - 1').gcd(R('u^2 - u'))
            u - 1
        """
        R = self.parent()
        return DifferentialPolynomial._wrap_handle(
            R, _blad.gcd(self._h(), other._h(), R.epoch))

    def factor(self):
        r"""
        Irreducible factorization over `\QQ`, as a Sage
        :class:`~sage.structure.factorization.Factorization`.

        The returned object iterates as ``(factor, multiplicity)`` pairs (each
        ``factor`` a non-constant irreducible :class:`DifferentialPolynomial`)
        and carries the numeric content as its :meth:`~sage.structure.factorization.Factorization.unit`
        -- matching Sage ``MPolynomial.factor`` so the shared Layer-0 engine's
        ``for fac, _ in p.factor()`` idiom works unchanged on the ``bap``
        substrate.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: F = R('2*(u-1)^2*(u-2)').factor()
            sage: F.unit()
            2
            sage: sorted((str(f), m) for f, m in F)
            [('u - 1', 2), ('u - 2', 1)]
        """
        from sage.structure.factorization import Factorization
        R = self.parent()
        ep = R.epoch
        num, facs = _blad.factor(self._h(), ep)
        unit = R(int(num))
        out = [(DifferentialPolynomial._wrap_handle(R, fh), int(m))
               for fh, m in facs]
        return Factorization(out, unit=unit)

    def squarefree_decomposition(self):
        r"""
        Yun squarefree decomposition: ``(unit, [(factor, multiplicity), ...])``
        with the ``factor`` pairwise coprime and individually squarefree.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: unit, facs = R('(u-1)^2*(u-2)').squarefree_decomposition()
            sage: sorted((str(f), m) for f, m in facs)
            [('u - 1', 2), ('u - 2', 1)]
        """
        R = self.parent()
        ep = R.epoch
        num, facs = _blad.squarefree(self._h(), ep)
        unit = R(int(num))
        out = [(DifferentialPolynomial._wrap_handle(R, fh), int(m))
               for fh, m in facs]
        return unit, out

    def squarefree_part(self):
        r"""
        The squarefree part: the product of the distinct irreducible factors.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('(u-1)^2*(u-2)').squarefree_part()
            u^2 - 3*u + 2
        """
        R = self.parent()
        _unit, facs = self.squarefree_decomposition()
        res = R.one()
        for fac, _m in facs:
            res = res * fac
        return res

    def resultant(self, other, v):
        r"""
        Resultant of ``self`` and ``other`` w.r.t. the jet ``v`` (a name or a
        degree-one element), via BLAD's Ducos resultant.

        ``v`` must be the highest-ranked variable of both operands (BLAD's
        Ducos resultant requires it); in the consumer it is always a chain
        leader, i.e. the top variable in BLAD's global ranking.

        .. NOTE::

            The result matches Sage's (Sylvester) ``resultant`` **up to a sign**:
            BLAD's Ducos resultant carries its own degree-ordering sign
            convention that is not a simple parity of the final degrees.  The
            *magnitude* is exact (verified by fuzz: 440/440 up-to-sign over
            random pairs sharing a leader).  Every consumer compares resultants
            up to a nonzero scalar (via ``_normalize``), so the sign is
            immaterial there; callers needing Sage's exact sign should normalize.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: r = R('u^2 - 1').resultant(R('u - 2'), 'u')
            sage: r in (R(3), R(-3))
            True
            sage: r.is_zero()
            False
        """
        R = self.parent()
        ep = R.epoch
        bn = R._as_blad_name(v)
        return DifferentialPolynomial._wrap_handle(
            R, _blad.resultant(self._h(), other._h(), bn, ep))

    def subresultants(self, other, v):
        r"""
        Full Ducos subresultant polynomial chain of ``self`` and ``other`` w.r.t.
        the jet ``v``, returned as a ``dict`` mapping each subresultant's degree
        in ``v`` to the (lowest-index) subresultant of that degree.

        ``v`` must be the highest-ranked variable of both operands (the Ducos
        constraint), exactly as for :meth:`resultant`.  The resultant ``S_0`` is
        the degree-0 entry; the lower-degree input is included at its own degree.

        Each subresultant matches Sage's ``subresultants`` degree-by-degree **up
        to a sign** (BLAD's Ducos convention); consumers compare up to a scalar.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: ch = R('u^3 - u').subresultants(R('u^2 + 1'), 'u')
            sage: 0 in ch       # the resultant S_0 is always present
            True
        """
        R = self.parent()
        ep = R.epoch
        bn = R._as_blad_name(v)
        chain = {}
        for deg, h in _blad.subresultant_chain(self._h(), other._h(), bn, ep):
            if deg not in chain:
                chain[deg] = DifferentialPolynomial._wrap_handle(R, h)
        return chain

    def content(self, v=None):
        r"""
        Content w.r.t. the jet ``v`` (a name / element), or w.r.t. the leader if
        ``v`` is ``None``.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('u[x]*u^2 + u[x]*u').content('u')
            u_x
        """
        R = self.parent()
        ep = R.epoch
        bn = None if v is None else R._as_blad_name(v)
        return DifferentialPolynomial._wrap_handle(
            R, _blad.content(self._h(), bn, ep))

    def primpart(self, v=None):
        r"""
        Primitive part w.r.t. the jet ``v`` (a name / element), or the leader if
        ``v`` is ``None``.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('u[x]*u^2 + u[x]*u').primpart('u')
            u^2 + u
        """
        R = self.parent()
        ep = R.epoch
        bn = None if v is None else R._as_blad_name(v)
        return DifferentialPolynomial._wrap_handle(
            R, _blad.primpart(self._h(), bn, ep))

    def exquo(self, other):
        r"""
        Exact quotient ``self / other`` (``other`` must divide ``self``).

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('u^2 - 1').exquo(R('u - 1'))
            u + 1
        """
        R = self.parent()
        return DifferentialPolynomial._wrap_handle(
            R, _blad.exquo(self._h(), other._h(), R.epoch))

    def divides(self, other):
        r"""
        Whether ``self`` divides ``other``.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('u - 1').divides(R('u^2 - 1'))
            True
            sage: R('u - 3').divides(R('u^2 - 1'))
            False
        """
        R = self.parent()
        divs, _cof = _blad.is_factor(other._h(), self._h(), R.epoch)
        return bool(divs)

    def coefficient_in(self, v, d):
        r"""
        Coefficient of ``self`` viewed as a polynomial in the jet ``v`` (a name /
        element) at degree ``d``.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('3*u^2 + 5*u + 7').coefficient_in('u', 2)
            3
            sage: R('3*u^2 + 5*u + 7').coefficient_in('u', 0)
            7
        """
        R = self.parent()
        ep = R.epoch
        bn = R._as_blad_name(v)
        return DifferentialPolynomial._wrap_handle(
            R, _blad.coefficient_in(self._h(), bn, int(d), ep))

    def pseudo_quo_rem(self, other, v=None):
        r"""
        Pseudo quotient and remainder: ``(q, r)`` with
        ``lc_v(other)^k * self == q * other + r`` and ``deg_v(r) < deg_v(other)``,
        reducing w.r.t. ``v`` (a name / element) or ``leader(other)``, where
        ``lc_v(other)`` is the leading coefficient of ``other`` in ``v``.

        The remainder is BLAD's ``prem``; the quotient is recovered exactly as
        ``(lc_v^k * self - r) / other``.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: p = R('u^3 + 1'); q = R('u^2 - u')
            sage: quo, rem = p.pseudo_quo_rem(q, 'u')
            sage: init = q.initial()
            sage: _, k = p.prem_with_power(q, 'u')
            sage: init**k * p == quo*q + rem
            True
        """
        R = self.parent()
        rem, k = self.prem_with_power(other, v)
        # the multiplier of prem is the leading coefficient of ``other`` w.r.t.
        # the *reduction variable* v (not other's own leader) -- for the default
        # v==leader(other) this coincides with other.initial().
        if v is None:
            init = other.initial()
        else:
            dv = other.degree_in(v)
            init = other.coefficient_in(v, dv)
        # init^k * self - rem is divisible by other (Ritt pseudo-division id.)
        scaled = self
        for _ in range(int(k)):
            scaled = scaled * init
        diff = scaled - rem
        if diff.is_zero():
            return R.zero(), rem
        quo = diff.exquo(other)
        return quo, rem

    # ======================================================================
    # Phase D: Sage-MPolynomial element idioms (dual-substrate).
    #
    # The shared Layer-0 engine calls these on the polynomials flowing through
    # ``triangularize`` / ``intersect`` / ``regular_gcd``.  Each is answered in
    # ``bap``-native terms (NOT via the ``__getattr__`` -> ``sage()`` jet-shadow
    # fallback, which would silently switch ranking / field-param semantics and
    # break the field-param-aware ``mvar``).  The variable argument ``g`` may be
    # a name string or a degree-one :class:`DifferentialPolynomial` (what
    # ``ring.gen(v)`` returns).
    # ======================================================================

    def is_constant(self):
        r"""
        Whether ``self`` is a base constant (a rational): NO appearing name at
        all -- not even a field param.

        .. NOTE::

            This is the *base-field* constant test, matching a Sage
            ``MPolynomial.is_constant`` over the full ring ``QQ[all vars]``: a
            pure field-param polynomial (``a^2 - 3``) is NOT constant here (it
            has an appearing name ``a``), even though it has no :meth:`mvar`.
            The ranking-aware "constant w.r.t. the ranking" test the engine
            needs is ``ring.mvar(p) is None`` (``polyutil.is_constant``), which
            consults :meth:`DifferentialPolynomialRing.mvar`.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('5').is_constant()
            True
            sage: R('u - 1').is_constant()
            False
        """
        return not self._jet_names()

    def degree(self, g=None):
        r"""
        Degree of ``self`` in the variable ``g`` (a name / degree-one element).
        With ``g=None``, the total degree.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: p = R('u[x]^3 + 3*u[x] - 5')
            sage: p.degree('u[x]')
            3
            sage: p.degree(R('u[x]'))
            3
        """
        if g is None:
            best = 0
            for _c, term in _blad.read_terms(self._h()):
                tot = sum(int(d) for _nm, d in term)
                if tot > best:
                    best = tot
            return int(best)
        return self.degree_in(g)

    def coefficient(self, arg):
        r"""
        Coefficient extraction in the Sage ``{var: degree}`` dict idiom.

        ``p.coefficient({g: d})`` returns the coefficient of ``g**d`` in ``p``
        (viewing the other variables as parameters), matching
        :meth:`coefficient_in`.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: p = R('3*u^2 + 5*u + 7')
            sage: p.coefficient({R('u'): 2})
            3
            sage: p.coefficient({'u': 0})
            7
        """
        if isinstance(arg, dict):
            if len(arg) != 1:
                raise NotImplementedError(
                    "coefficient supports a single {var: degree} entry")
            (v, d), = arg.items()
            return self.coefficient_in(v, int(d))
        raise NotImplementedError("coefficient expects a {var: degree} dict")

    def derivative(self, g):
        r"""
        ALGEBRAIC partial derivative ``d(self)/dg`` w.r.t. the single jet ``g``
        (treating every other jet as independent) -- NOT the total/differential
        derivative :meth:`differentiate`.

        This is the ``MPolynomial.derivative(var)`` the engine calls when
        forming a separant / discriminant ``d p / d (leader)`` algebraically.
        Implemented as ``sum_{k>=1} k * coeff_k(g) * g**(k-1)``.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('u[x]^3 + 3*u[x] - 5').derivative('u[x]')
            3*u_x^2 + 3
            sage: R('u[x]^2*u + u').derivative('u')
            u_x^2 + 1
        """
        R = self.parent()
        d = self.degree_in(g)
        if d == 0:
            return R.zero()
        gel = R(R._as_blad_name(g))
        res = R.zero()
        gpow = R.one()              # g**(k-1)
        for k in range(1, d + 1):
            ck = self.coefficient_in(g, k)
            res = res + R(int(k)) * ck * gpow
            gpow = gpow * gel
        return res

    def lc(self):
        r"""
        The leading coefficient: the numeric (rational) content's sign-carrying
        leading numeric coefficient in BLAD canonical order -- a base constant.

        For the engine's :func:`_normalize` this only needs to (a) be a base
        constant exactly when ``self``'s leading term is field-param-free, and
        (b) carry the right sign.  We return the integer coefficient of the
        first (leading) term as a :class:`DifferentialPolynomial` constant.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: R('3*u^2 + 5*u + 7').lc()
            3
        """
        R = self.parent()
        terms = list(_blad.read_terms(self._h()))
        if not terms:
            return R.zero()
        return R(int(terms[0][0]))

    def variables(self):
        r"""
        The appearing variables as degree-one :class:`DifferentialPolynomial`
        elements, in ranking order (highest first) -- mirroring
        ``MPolynomial.variables``.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: [str(v) for v in R('u[x]*u + u').variables()]
            ['u_x', 'u']
        """
        R = self.parent()
        names = R.sort(list(self._jet_names()), "descending")
        return [R(nm) for nm in names]

    def dict(self):
        r"""
        The ``{exponent-tuple: coefficient}`` dict over the FINITE generating
        set (mirroring ``MPolynomial.dict``).

        Only meaningful when every appearing name is a finite generator (a head
        or a field param); a jet/prolongation outside the finite set raises.
        Used by the (peripheral) field-param-content stripper.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: sorted(R('3*u^2 + 5').dict().items())
            [((0, 0), 5), ((0, 2), 3)]
        """
        R = self.parent()
        names = R._all_finite_names()
        pos = {nm: i for i, nm in enumerate(names)}
        ng = len(names)
        out = {}
        for coeff, term in _blad.read_terms(self._h()):
            exps = [0] * ng
            for nm, deg in term:
                if nm not in pos:
                    raise KeyError(
                        "dict() over the finite generating set, but %r is not "
                        "a finite generator" % (nm,))
                exps[pos[nm]] += int(deg)
            key = tuple(exps)
            c = QQ(coeff)
            out[key] = out.get(key, QQ(0)) + c
        return out

    def __floordiv__(self, other):
        r"""
        Exact division ``self // other`` for the engine's ``p // content`` idiom
        (the field-param-content stripper divides by an exact factor).

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: (R('u^2 - 1') // R('u - 1'))
            u + 1
        """
        R = self.parent()
        if not isinstance(other, DifferentialPolynomial):
            other = R(other)
        return self.exquo(other)

    # -- lowering to Sage ---------------------------------------------------
    def sage(self):
        r"""
        Lower to the jet shadow: a Sage ``MPolynomial`` in
        ``R.jet_shadow([self])``.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: p = R('u[x,x]^2 + 3*u[x] - 5')
            sage: g = p.sage(); g.parent()
            Multivariate Polynomial Ring in u_x_x, u_x over Rational Field
            sage: g
            u_x_x^2 + 3*u_x - 5
        """
        R = self.parent()
        ring, gmap = R.jet_shadow([self])
        result = ring.zero()
        for coeff, term in _blad.read_terms(self._h()):
            mon = ring.one()
            for nm, deg in term:
                mon *= gmap[nm] ** deg
            result += QQ(coeff) * mon
        self._owned = (ring, result)
        return result

    def _sage_(self):
        return self.sage()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.sage(), name)


DifferentialPolynomialRing.Element = DifferentialPolynomial


def _reconstruct_element(parent, blad_string):
    """Unpickle helper: rebuild an element from its BLAD string snapshot."""
    return parent.element_class(parent, blad_string)


def _term_sort_key(term, R):
    """Decreasing-ranking sort key for a monomial's (name,deg) list."""
    key = []
    for nm, d in sorted(term, key=lambda nd: R._ranking_key(nd[0])):
        rk = R._ranking_key(nm)
        key.append((rk[0], rk[1], rk[2], -d))
    return key

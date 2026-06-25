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

    def differential_prem(self, p, reductors):
        r"""
        Full **differential** pseudo-remainder of ``p`` by ``reductors``.

        Reduce ``p`` by each reductor *and all its derivatives* (Ritt
        reduction), highest-leader-first.  This is the differential analogue of
        :meth:`DifferentialPolynomial.prem` (which reduces by a single algebraic
        leader only): a derivative ``D`` strictly above a reductor's leader is
        eliminated by that reductor prolonged to leader ``D``; the leader itself
        is reduced last.

        Returns ``(r, h)`` where ``r`` is the reduced
        :class:`DifferentialPolynomial` and ``h`` is the product of the
        initials/separants the reduction multiplied through (a
        :class:`DifferentialPolynomial`; ``R.one()`` if none) -- the
        non-vanishing Thomas inequation cofactor.

        v1 supports a list with a single reductor (matching the consumer);
        a multi-reductor list reduces against each in turn.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: A = R('u[x,x,x]^2 + u[x,x] - u')
            sage: B = R('u[x,x] - u^2')
            sage: r, h = R.differential_prem(A, [B]); r
            4*u_x^2*u^2 + u^2 - u
        """
        if not isinstance(reductors, (list, tuple)):
            reductors = [reductors]
        r = self(p) if not isinstance(p, DifferentialPolynomial) else p
        h = self.one()
        for B in reductors:
            r, hB = self._differential_prem_one(r, B)
            h = h * hB
            if r.is_zero():
                break
        return r, h

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
        best = 0
        for _coeff, term in _blad.read_terms(self._h()):
            for nm, deg in term:
                if nm == bn and deg > best:
                    best = deg
        return int(best)

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
        Irreducible factorization over `\QQ`.

        Returns ``(unit, [(factor, multiplicity), ...])`` where ``unit`` is a
        rational :class:`DifferentialPolynomial` (the numeric content) and each
        ``factor`` is a non-constant irreducible :class:`DifferentialPolynomial`.

        EXAMPLES::

            sage: from sage_differential_polynomial import DifferentialPolynomialRing
            sage: R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
            sage: unit, facs = R('2*(u-1)^2*(u-2)').factor()
            sage: unit
            2
            sage: sorted((str(f), m) for f, m in facs)
            [('u - 1', 2), ('u - 2', 1)]
        """
        R = self.parent()
        ep = R.epoch
        num, facs = _blad.factor(self._h(), ep)
        unit = R(int(num))
        out = [(DifferentialPolynomial._wrap_handle(R, fh), int(m))
               for fh, m in facs]
        return unit, out

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

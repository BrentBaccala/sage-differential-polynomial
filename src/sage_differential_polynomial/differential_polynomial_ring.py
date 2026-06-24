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

    def _ranking_key(self, blad_name):
        """Sort key approximating the BLAD ranking (higher rank -> earlier)."""
        if "[" in blad_name:
            head, rest = blad_name.split("[", 1)
            order = len(rest.rstrip("]").split(","))
        else:
            head, order = blad_name, 0
        try:
            hpos = self._indeterminates.index(head)
        except ValueError:
            hpos = 1000
        return (-order, hpos, blad_name)


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
        p._blad_string = handle.to_string()
        return p

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
        return hash(self._h().to_string())

    def __reduce__(self):
        # serialize via the durable BLAD string snapshot (never the C handle)
        return (_reconstruct_element, (self.parent(), self._h().to_string()))

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
        """
        return _blad.leader_name(self._h())

    def leading_derivative(self):
        """Alias for :meth:`leader`."""
        return self.leader()

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

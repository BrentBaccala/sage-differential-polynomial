# cython: language_level=3
# distutils: language = c
r"""
Low-level term-walk binding to BLAD (libblad), statically linked with symbols
hidden.

This module is the *thin* C layer.  It exposes BLAD's differential-polynomial
engine to Python as a handful of operations on opaque ``bap_polynom_mpz``
handles, plus a term-walk marshalling pair (bap -> Python tuples, Python tuples
-> bap) that never crosses a string boundary on the polynomial body.

The high-level Sage Parent/Element types live in pure Python in
``differential_polynomial_ring.py`` and call into here.

Memory model: a single ``bas_restart`` at ring install; per top-level operation
the high-level layer does nothing special -- BLAD's own stacks accumulate, and
results we wish to keep are immediately read out to Python (GMP limb-copied
Integers + variable-name tuples), so the bap arena can be reset.  Every BLAD
call is wrapped in BA0_TRY/CATCH inside the C helper functions below and
converted to a Python exception.

Variables are identified across the boundary by their BLAD *name string*
(e.g. ``u``, ``u[x]``, ``u[x,x]``) -- stable and unique -- which the Python
layer maps to Sage jet-shadow generators.  ``index_in_vars`` is the stable
integer handle BLAD uses internally; we expose the name (printed via ``%v``)
because it round-trips through ``ba0_sscanf2(name, "%v", &v)`` to recreate /
look up the very same variable, including lazily creating new derivatives.
"""

from cpython.bytes cimport PyBytes_FromString

cimport sage_differential_polynomial.cblad as cb

# GMP types/functions, declared locally (not cimported from Sage) so this
# extension does not depend on Sage's exact Cython ABI for sage.rings.integer.
cdef extern from "gmp.h":
    ctypedef struct __mpz_struct:
        pass
    ctypedef __mpz_struct mpz_t[1]
    ctypedef __mpz_struct *mpz_ptr
    ctypedef const __mpz_struct *mpz_srcptr
    void mpz_init(mpz_t)
    void mpz_clear(mpz_t)
    void mpz_set(mpz_t, mpz_srcptr)
    char *mpz_get_str(char *, int, mpz_srcptr)
    int mpz_set_str(mpz_t, const char *, int)
    size_t mpz_sizeinbase(mpz_srcptr, int)

from libc.stdlib cimport malloc as _malloc, free as _free


class BladError(RuntimeError):
    """A BLAD exception, translated to Python."""


# ---------------------------------------------------------------------------
# C helper layer: each BLAD-touching primitive wrapped in BA0_TRY/CATCH.
#
# A helper returns 0 on success, 1 on BLAD exception; on exception it copies
# the BLAD message into a caller-supplied buffer.  This keeps the setjmp frame
# entirely in C (Cython cannot host BA0_TRY/CATCH directly).
# ---------------------------------------------------------------------------
cdef extern from *:
    """
    #include "blad.h"
    #include <string.h>
    #include <gmp.h>

    /* one-time init guard */
    static int sdp_initialized = 0;

    /* A no-op replacement for GMP's mp_set_memory_functions.  By handing this
       to ba0_set_settings_gmp before bas_restart, we stop BLAD from hijacking
       the *global* GMP allocator: it leaves whatever allocators are already
       installed (Sage's) in place and uses them for its own mpz too.  This is
       what makes in-process coexistence with Sage Integers safe -- both sides
       share one GMP allocator (Sage's), so no integer is freed by the wrong
       allocator.  (Equivalent to the BMI Maple path that pushes/pops the host
       allocator; here the host stays the allocator throughout.) */
    static void sdp_noop_set_memory_functions(
        void *(*alloc)(size_t),
        void *(*realloc)(void *, size_t, size_t),
        void (*freefn)(void *, size_t))
    {
        (void)alloc; (void)realloc; (void)freefn;  /* deliberately do nothing */
    }

    static void sdp_copymsg(char *buf, int n) {
        const char *m = ba0_global.exception.raised;
        if (!m) m = "BLAD exception (no message)";
        strncpy(buf, m, n - 1);
        buf[n-1] = 0;
    }

    /* bas_restart once.  NOTE: bas_restart bootstraps the ba0 exception
       machinery itself, so it must NOT be wrapped in BA0_TRY (the TRY macro
       pushes onto ba0_global.exception.stack, which bas_restart is what
       allocates).  bav_set_settings_ordering after it is safe to wrap. */
    static int sdp_init(char *err, int n) {
        if (sdp_initialized) return 0;
        /* Keep Sage's GMP allocators: install a no-op setter so bas_restart
           does not call mp_set_memory_functions. */
        /* Integer_PFE = 0  -> print integers plainly ("3", not "Integer(3)"),
           so to_string() output re-parses and reads cleanly in _repr_. */
        ba0_set_settings_gmp(&sdp_noop_set_memory_functions, (char *)0);
        bas_restart(0, 0);
        BA0_TRY {
            bav_set_settings_ordering("ranking");
            sdp_initialized = 1;
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }

    /* Install a ranking from a fully-formatted "ranking(...)" string. */
    static int sdp_install_ranking(const char *rankstr, char *err, int n) {
        bav_Iordering r;
        BA0_TRY {
            ba0_scanf_printf("%ordering", (char *)rankstr, &r);
            if (bav_R_ambiguous_symbols()) {
                strncpy(err, "ambiguous symbols in ranking", n-1); err[n-1]=0;
                return 1;
            }
            bav_push_ordering(r);
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }

    /* Allocate a fresh persistent bap_polynom_mpz on BLAD's stack. */
    static struct bap_polynom_mpz *sdp_new_poly(char *err, int n) {
        struct bap_polynom_mpz *p = (struct bap_polynom_mpz *)0;
        BA0_TRY {
            p = bap_new_polynom_mpz();
        } BA0_CATCH { sdp_copymsg(err, n); return (struct bap_polynom_mpz*)0; } BA0_ENDTRY;
        return p;
    }

    /* Parse a polynomial from a string (used only for small inputs / tests). */
    static int sdp_parse(struct bap_polynom_mpz *p, const char *s, char *err, int n) {
        BA0_TRY {
            ba0_sscanf2((char *)s, "%Az", p);
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }

    /* Print a polynomial to a freshly-allocated string. */
    static char *sdp_print(struct bap_polynom_mpz *p, char *err, int n) {
        char *s = (char*)0;
        BA0_TRY {
            s = ba0_new_printf("%Az", p);
        } BA0_CATCH { sdp_copymsg(err, n); return (char*)0; } BA0_ENDTRY;
        return s;
    }

    /* number of monomials */
    static long sdp_nbmon(struct bap_polynom_mpz *p) {
        return (long)bap_nbmon_polynom_mpz(p);
    }
    static int sdp_iszero(struct bap_polynom_mpz *p) {
        return bap_is_zero_polynom_mpz(p) ? 1 : 0;
    }
    static int sdp_equal(struct bap_polynom_mpz *a, struct bap_polynom_mpz *b) {
        return bap_equal_polynom_mpz(a,b) ? 1 : 0;
    }

    /* leader name; returns NULL if constant (no leader). */
    static char *sdp_leader(struct bap_polynom_mpz *p, char *err, int n) {
        char *s = (char*)0;
        BA0_TRY {
            if (bap_is_zero_polynom_mpz(p)) { return (char*)0; }
            struct bav_variable *v = bap_leader_polynom_mpz(p);
            if (v == (struct bav_variable*)0) return (char*)0;
            /* a numeric polynomial has no real leader */
            if (v->root->type == bav_independent_symbol &&
                bap_nbmon_polynom_mpz(p) == 1) { /* fallthrough, still print */ }
            s = ba0_new_printf("%v", v);
        } BA0_CATCH { sdp_copymsg(err, n); return (char*)0; } BA0_ENDTRY;
        return s;
    }

    /* differentiate w.r.t. a single derivation named `der`. */
    static int sdp_diff1(struct bap_polynom_mpz *out, struct bap_polynom_mpz *in,
                         const char *der, char *err, int n) {
        BA0_TRY {
            struct bav_symbol *s = bav_R_string_to_existing_derivation((char*)der);
            if (s == (struct bav_symbol*)0) {
                strncpy(err, "unknown derivation", n-1); err[n-1]=0; return 1;
            }
            bap_diff_polynom_mpz(out, in, s);
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }

    /* separant w.r.t. the leader. */
    static int sdp_separant(struct bap_polynom_mpz *out, struct bap_polynom_mpz *in,
                            char *err, int n) {
        BA0_TRY { bap_separant_polynom_mpz(out, in); }
        BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }
    /* separant2 w.r.t. a named variable v (formal jet-partial). */
    static int sdp_separant2(struct bap_polynom_mpz *out, struct bap_polynom_mpz *in,
                             const char *var, char *err, int n) {
        BA0_TRY {
            struct bav_variable *v = (struct bav_variable*)0;
            ba0_sscanf2((char*)var, "%v", &v);
            bap_separant2_polynom_mpz(out, in, v);
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }

    static int sdp_initial(struct bap_polynom_mpz *out, struct bap_polynom_mpz *in,
                           char *err, int n) {
        BA0_TRY { bap_initial_polynom_mpz(out, in); }
        BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }

    /* Ritt pseudo-remainder of A by B w.r.t. leader(B). out h = power of init. */
    static int sdp_prem(struct bap_polynom_mpz *out, long *h,
                        struct bap_polynom_mpz *A, struct bap_polynom_mpz *B,
                        char *err, int n) {
        BA0_TRY {
            struct bav_variable *v = bap_leader_polynom_mpz(B);
            bav_Idegree hd = 0;
            bap_prem_polynom_mpz(out, &hd, A, B, v);
            *h = (long)hd;
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }
    /* prem of A by B w.r.t. a *named* leader variable (explicit reduction var). */
    static int sdp_prem_var(struct bap_polynom_mpz *out, long *h,
                            struct bap_polynom_mpz *A, struct bap_polynom_mpz *B,
                            const char *var, char *err, int n) {
        BA0_TRY {
            struct bav_variable *v = (struct bav_variable*)0;
            ba0_sscanf2((char*)var, "%v", &v);
            bav_Idegree hd = 0;
            bap_prem_polynom_mpz(out, &hd, A, B, v);
            *h = (long)hd;
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }
    """
    int sdp_init(char *, int)
    int sdp_install_ranking(const char *, char *, int)
    cb.bap_polynom_mpz *sdp_new_poly(char *, int)
    int sdp_parse(cb.bap_polynom_mpz *, const char *, char *, int)
    char *sdp_print(cb.bap_polynom_mpz *, char *, int)
    long sdp_nbmon(cb.bap_polynom_mpz *)
    int sdp_iszero(cb.bap_polynom_mpz *)
    int sdp_equal(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *)
    char *sdp_leader(cb.bap_polynom_mpz *, char *, int)
    int sdp_diff1(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, const char *, char *, int)
    int sdp_separant(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, char *, int)
    int sdp_separant2(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, const char *, char *, int)
    int sdp_initial(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, char *, int)
    int sdp_prem(cb.bap_polynom_mpz *, long *, cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, char *, int)
    int sdp_prem_var(cb.bap_polynom_mpz *, long *, cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, const char *, char *, int)


DEF ERRBUF = 1024


cdef inline _check(int rc, char *err):
    if rc != 0:
        raise BladError(err.decode("utf-8", "replace"))


# ---------------------------------------------------------------------------
# Module init / ring install
# ---------------------------------------------------------------------------
def blad_init():
    cdef char err[ERRBUF]
    if sdp_init(err, ERRBUF) != 0:
        raise BladError(err.decode("utf-8", "replace"))


def install_ranking(str rankstr):
    """Install a ranking from a fully-formatted ``ranking(...)`` string."""
    cdef char err[ERRBUF]
    cdef bytes b = rankstr.encode("utf-8")
    if sdp_install_ranking(b, err, ERRBUF) != 0:
        raise BladError(err.decode("utf-8", "replace"))


# ---------------------------------------------------------------------------
# Poly handle: a Python wrapper holding a bap_polynom_mpz* on BLAD's stack.
# The pointer is valid only while the current ring is installed (epoch).  The
# high-level layer owns the canonical Sage form; this handle is a cache.
# ---------------------------------------------------------------------------
cdef class PolyHandle:
    cdef cb.bap_polynom_mpz *ptr
    cdef readonly long epoch

    def __cinit__(self):
        self.ptr = NULL
        self.epoch = -1

    @staticmethod
    cdef PolyHandle _wrap(cb.bap_polynom_mpz *p, long epoch):
        cdef PolyHandle h = PolyHandle.__new__(PolyHandle)
        h.ptr = p
        h.epoch = epoch
        return h

    def nbmon(self):
        return sdp_nbmon(self.ptr)

    def is_zero(self):
        return bool(sdp_iszero(self.ptr))

    def __richcmp__(PolyHandle self, PolyHandle other, int op):
        cdef int eq = sdp_equal(self.ptr, other.ptr)
        if op == 2:   # ==
            return bool(eq)
        if op == 3:   # !=
            return not bool(eq)
        return NotImplemented

    def to_string(self):
        cdef char err[ERRBUF]
        cdef char *s = sdp_print(self.ptr, err, ERRBUF)
        if s == NULL:
            raise BladError(err.decode("utf-8", "replace"))
        return PyBytes_FromString(s).decode("utf-8")


def new_handle(long epoch):
    cdef char err[ERRBUF]
    cdef cb.bap_polynom_mpz *p = sdp_new_poly(err, ERRBUF)
    if p == NULL:
        raise BladError(err.decode("utf-8", "replace"))
    return PolyHandle._wrap(p, epoch)


def parse_poly(str s, long epoch):
    """Parse a polynomial from a BLAD string into a fresh handle (small inputs)."""
    cdef char err[ERRBUF]
    cdef cb.bap_polynom_mpz *p = sdp_new_poly(err, ERRBUF)
    if p == NULL:
        raise BladError(err.decode("utf-8", "replace"))
    cdef bytes b = s.encode("utf-8")
    if sdp_parse(p, b, err, ERRBUF) != 0:
        raise BladError(err.decode("utf-8", "replace"))
    return PolyHandle._wrap(p, epoch)


# ---------------------------------------------------------------------------
# Term-walk marshalling.
#
# read_terms(handle)  -> list of (Integer coeff, tuple of (varname, deg))
#   limb-copies each mpz coefficient into a Sage Integer; reads each variable's
#   BLAD name via %v.  No string round-trip on the polynomial body.
#
# build_terms(terms, epoch) -> PolyHandle
#   builds a bap polynomial from such a list via the creator API.  Each term is
#   (Integer coeff, tuple of (varname, deg)); terms must be in BLAD's current
#   ranking order (the high-level layer sorts before calling).
# ---------------------------------------------------------------------------
cdef extern from *:
    """
    #include "blad.h"
    #include <gmp.h>

    /* iterate monomials, calling back into Cython via the provided context.
       We can't easily call Python per-term from inside a TRY, so instead the
       read-out is split: a counted "describe" that fills caller arrays.
       Simpler: expose primitives to walk one polynomial. */

    /* Opaque iterator allocated by the caller. */
    typedef struct sdp_iter { struct bap_itermon_mpz it; struct bav_term T; int started; } sdp_iter;

    static int sdp_iter_begin(struct sdp_iter *s, struct bap_polynom_mpz *p,
                              char *err, int n) {
        BA0_TRY {
            bav_init_term(&s->T);
            bap_begin_itermon_mpz(&s->it, p);
            s->started = 1;
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }
    static int sdp_iter_outof(struct sdp_iter *s) {
        return bap_outof_itermon_mpz(&s->it) ? 1 : 0;
    }
    static void sdp_iter_next(struct sdp_iter *s) {
        bap_next_itermon_mpz(&s->it);
    }
    /* current coefficient pointer (do not free) */
    static mpz_ptr sdp_iter_coeff(struct sdp_iter *s) {
        ba0_mpz_t *c = bap_coeff_itermon_mpz(&s->it);
        return (mpz_ptr)(*c);
    }
    /* number of distinct variables in current term */
    static long sdp_iter_termsize(struct sdp_iter *s) {
        bap_term_itermon_mpz(&s->T, &s->it);
        return (long)s->T.size;
    }
    /* name of j-th variable in current term (must call termsize first) */
    static char *sdp_iter_varname(struct sdp_iter *s, long j, char *err, int n) {
        char *r = (char*)0;
        BA0_TRY { r = ba0_new_printf("%v", s->T.rg[j].var); }
        BA0_CATCH { sdp_copymsg(err, n); return (char*)0; } BA0_ENDTRY;
        return r;
    }
    static long sdp_iter_vardeg(struct sdp_iter *s, long j) {
        return (long)s->T.rg[j].deg;
    }

    /* ---- creator (build) ---- */
    typedef struct sdp_builder {
        struct bap_creator_mpz crea;
        struct bav_term tot;
        struct bav_term cur;
        int open;
    } sdp_builder;
    /* Begin building into poly p with nmon monomials and a generous total rank.
       We assemble the total rank by unioning each term; simplest correct option
       is to pass an exact total rank we accumulate first.  To avoid two passes,
       we build the total-rank term incrementally is not allowed by the creator
       (needs it up front), so the Python layer supplies the total-rank variable
       multiset as a flat (name,deg) list. */
    static int sdp_build_begin(struct sdp_builder *b, struct bap_polynom_mpz *p,
                               const char **tot_names, long *tot_degs, long tot_size,
                               long nmon, char *err, int n) {
        BA0_TRY {
            bav_init_term(&b->tot);
            bav_init_term(&b->cur);
            long i;
            for (i = 0; i < tot_size; i++) {
                struct bav_variable *v = (struct bav_variable*)0;
                ba0_sscanf2((char*)tot_names[i], "%v", &v);
                /* set total_rank: multiply variable to deg into b->tot */
                struct bav_term one;
                bav_init_term(&one);
                bav_set_term_variable(&one, v, (bav_Idegree)tot_degs[i]);
                bav_mul_term(&b->tot, &b->tot, &one);
            }
            bap_begin_creator_mpz(&b->crea, p, &b->tot, bap_exact_total_rank, (ba0_int_p)nmon);
            b->open = 1;
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }
    /* write one monomial: coeff (mpz) and a (name,deg) list for its term. */
    static int sdp_build_write(struct sdp_builder *b, mpz_srcptr coeff,
                               const char **names, long *degs, long size,
                               char *err, int n) {
        BA0_TRY {
            bav_set_term_one(&b->cur);
            long i;
            for (i = 0; i < size; i++) {
                struct bav_variable *v = (struct bav_variable*)0;
                ba0_sscanf2((char*)names[i], "%v", &v);
                struct bav_term one;
                bav_init_term(&one);
                bav_set_term_variable(&one, v, (bav_Idegree)degs[i]);
                bav_mul_term(&b->cur, &b->cur, &one);
            }
            ba0_mpz_t c;
            ba0_mpz_init(c);
            ba0_mpz_set(c, (mpz_ptr)coeff);
            bap_write_creator_mpz(&b->crea, &b->cur, c);
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }
    static int sdp_build_close(struct sdp_builder *b, char *err, int n) {
        BA0_TRY { bap_close_creator_mpz(&b->crea); b->open = 0; }
        BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }
    """
    ctypedef struct sdp_iter:
        pass
    int sdp_iter_begin(sdp_iter *, cb.bap_polynom_mpz *, char *, int)
    int sdp_iter_outof(sdp_iter *)
    void sdp_iter_next(sdp_iter *)
    mpz_ptr sdp_iter_coeff(sdp_iter *)
    long sdp_iter_termsize(sdp_iter *)
    char *sdp_iter_varname(sdp_iter *, long, char *, int)
    long sdp_iter_vardeg(sdp_iter *, long)

    ctypedef struct sdp_builder:
        pass
    int sdp_build_begin(sdp_builder *, cb.bap_polynom_mpz *, const char **, long *, long, long, char *, int)
    int sdp_build_write(sdp_builder *, mpz_srcptr, const char **, long *, long, char *, int)
    int sdp_build_close(sdp_builder *, char *, int)


_B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _pyint_base62(n):
    """Render a Python int / Sage Integer in GMP's base-62 alphabet."""
    n = int(n)
    if n == 0:
        return "0"
    neg = n < 0
    if neg:
        n = -n
    out = []
    while n:
        n, r = divmod(n, 62)
        out.append(_B62[r])
    if neg:
        out.append("-")
    return "".join(reversed(out))


_B62_VAL = {c: i for i, c in enumerate(_B62)}


cdef object _mpz_to_pyint(mpz_srcptr z):
    """Convert a GMP integer to a Python int via base-62 string (ABI-free)."""
    cdef size_t n = mpz_sizeinbase(z, 62) + 2
    cdef char *buf = <char *>_malloc(n)
    if buf == NULL:
        raise MemoryError()
    try:
        mpz_get_str(buf, 62, z)
        s = PyBytes_FromString(buf).decode("ascii")
    finally:
        _free(buf)
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    val = 0
    for ch in s:
        val = val * 62 + _B62_VAL[ch]
    return -val if neg else val


def read_terms(PolyHandle h):
    r"""
    Term-walk read-out: return a list of ``(int, term)`` where ``term`` is a
    tuple of ``(varname, degree)`` pairs.  The Python layer maps these onto
    Sage Integers / jet-shadow generators; this layer stays ABI-free.
    """
    cdef char err[ERRBUF]
    cdef sdp_iter it
    cdef long sz, j, deg
    cdef char *nm
    cdef list out = []
    if sdp_iter_begin(&it, h.ptr, err, ERRBUF) != 0:
        raise BladError(err.decode("utf-8", "replace"))
    while sdp_iter_outof(&it) == 0:
        coeff = _mpz_to_pyint(sdp_iter_coeff(&it))
        sz = sdp_iter_termsize(&it)
        term = []
        for j in range(sz):
            nm = sdp_iter_varname(&it, j, err, ERRBUF)
            if nm == NULL:
                raise BladError(err.decode("utf-8", "replace"))
            deg = sdp_iter_vardeg(&it, j)
            term.append((PyBytes_FromString(nm).decode("utf-8"), int(deg)))
        out.append((coeff, tuple(term)))
        sdp_iter_next(&it)
    return out


def build_terms(list terms, list total_rank, long epoch):
    r"""
    Term-walk build: construct a bap polynomial from ``terms`` (a list of
    ``(int, ((varname,deg),...))`` in *decreasing* ranking order) with the
    given ``total_rank`` (a list of ``(varname, deg)`` covering the max degree of
    each appearing variable).  Returns a :class:`PolyHandle`.
    """
    cdef char err[ERRBUF]
    cdef cb.bap_polynom_mpz *p = sdp_new_poly(err, ERRBUF)
    if p == NULL:
        raise BladError(err.decode("utf-8", "replace"))

    cdef sdp_builder b
    cdef long tot_size = len(total_rank)
    cdef const char **tot_names = <const char **>_malloc(sizeof(char*) * (tot_size if tot_size else 1))
    cdef long *tot_degs = <long *>_malloc(sizeof(long) * (tot_size if tot_size else 1))
    cdef list keepalive = []
    cdef bytes nb
    cdef long i
    try:
        for i in range(tot_size):
            nb = total_rank[i][0].encode("utf-8")
            keepalive.append(nb)
            tot_names[i] = nb
            tot_degs[i] = total_rank[i][1]
        if sdp_build_begin(&b, p, tot_names, tot_degs, tot_size,
                           len(terms), err, ERRBUF) != 0:
            raise BladError(err.decode("utf-8", "replace"))
    finally:
        _free(tot_names)
        _free(tot_degs)

    cdef mpz_t cz
    cdef long sz
    cdef const char **names
    cdef long *degs
    cdef bytes cb_str
    mpz_init(cz)
    try:
        for coeff_obj, term in terms:
            # coeff_obj is a Python int or Sage Integer; route via base-62 string
            cb_str = _pyint_base62(coeff_obj).encode("ascii")
            mpz_set_str(cz, cb_str, 62)
            sz = len(term)
            names = <const char **>_malloc(sizeof(char*) * (sz if sz else 1))
            degs = <long *>_malloc(sizeof(long) * (sz if sz else 1))
            keep2 = []
            try:
                for i in range(sz):
                    nb = term[i][0].encode("utf-8")
                    keep2.append(nb)
                    names[i] = nb
                    degs[i] = term[i][1]
                if sdp_build_write(&b, cz, names, degs, sz, err, ERRBUF) != 0:
                    raise BladError(err.decode("utf-8", "replace"))
            finally:
                _free(names)
                _free(degs)

        if sdp_build_close(&b, err, ERRBUF) != 0:
            raise BladError(err.decode("utf-8", "replace"))
    finally:
        mpz_clear(cz)
    return PolyHandle._wrap(p, epoch)


# ---------------------------------------------------------------------------
# Differential operations (handle -> handle)
# ---------------------------------------------------------------------------
def leader_name(PolyHandle h):
    cdef char err[ERRBUF]
    cdef char *s = sdp_leader(h.ptr, err, ERRBUF)
    if s == NULL:
        # could be a genuine error or "no leader" (constant). Distinguish:
        if err[0] != 0:
            raise BladError(err.decode("utf-8", "replace"))
        return None
    return PyBytes_FromString(s).decode("utf-8")


def differentiate(PolyHandle h, str derivation, long epoch):
    cdef char err[ERRBUF]
    err[0] = 0
    cdef cb.bap_polynom_mpz *out = sdp_new_poly(err, ERRBUF)
    if out == NULL:
        raise BladError(err.decode("utf-8", "replace"))
    cdef bytes d = derivation.encode("utf-8")
    if sdp_diff1(out, h.ptr, d, err, ERRBUF) != 0:
        raise BladError(err.decode("utf-8", "replace"))
    return PolyHandle._wrap(out, epoch)


def separant(PolyHandle h, long epoch, var=None):
    cdef char err[ERRBUF]
    err[0] = 0
    cdef cb.bap_polynom_mpz *out = sdp_new_poly(err, ERRBUF)
    if out == NULL:
        raise BladError(err.decode("utf-8", "replace"))
    cdef bytes vb
    if var is None:
        if sdp_separant(out, h.ptr, err, ERRBUF) != 0:
            raise BladError(err.decode("utf-8", "replace"))
    else:
        vb = var.encode("utf-8")
        if sdp_separant2(out, h.ptr, vb, err, ERRBUF) != 0:
            raise BladError(err.decode("utf-8", "replace"))
    return PolyHandle._wrap(out, epoch)


def initial(PolyHandle h, long epoch):
    cdef char err[ERRBUF]
    err[0] = 0
    cdef cb.bap_polynom_mpz *out = sdp_new_poly(err, ERRBUF)
    if out == NULL:
        raise BladError(err.decode("utf-8", "replace"))
    if sdp_initial(out, h.ptr, err, ERRBUF) != 0:
        raise BladError(err.decode("utf-8", "replace"))
    return PolyHandle._wrap(out, epoch)


def prem(PolyHandle a, PolyHandle b, long epoch, var=None):
    r"""
    Ritt pseudo-remainder of ``a`` by ``b``.  If ``var`` is given, reduce w.r.t.
    that named leader; otherwise w.r.t. ``leader(b)``.  Returns ``(remainder, h)``
    where ``h`` is the power the initial of ``b`` was raised to.
    """
    cdef char err[ERRBUF]
    err[0] = 0
    cdef cb.bap_polynom_mpz *out = sdp_new_poly(err, ERRBUF)
    if out == NULL:
        raise BladError(err.decode("utf-8", "replace"))
    cdef long h = 0
    cdef bytes vb
    if var is None:
        if sdp_prem(out, &h, a.ptr, b.ptr, err, ERRBUF) != 0:
            raise BladError(err.decode("utf-8", "replace"))
    else:
        vb = var.encode("utf-8")
        if sdp_prem_var(out, &h, a.ptr, b.ptr, vb, err, ERRBUF) != 0:
            raise BladError(err.decode("utf-8", "replace"))
    return PolyHandle._wrap(out, epoch), int(h)

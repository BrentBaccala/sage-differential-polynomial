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

    /* DIAGNOSTIC: current depth of BLAD's exception-handler stack. */
    static long sdp_exc_stack_size(void) {
        return (long)ba0_global.exception.stack.size;
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
        int ambiguous = 0;
        BA0_TRY {
            ba0_scanf_printf("%ordering", (char *)rankstr, &r);
            if (bav_R_ambiguous_symbols()) {
                strncpy(err, "ambiguous symbols in ranking", n-1); err[n-1]=0;
                ambiguous = 1;      /* do NOT return inside TRY (frame leak) */
            } else {
                bav_push_ordering(r);
            }
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return ambiguous ? 1 : 0;
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
        /* NOTE: the "no leader" cases (zero / constant polynomial) must NOT
           `return` from inside the BA0_TRY body -- an early return skips
           BA0_ENDTRY and therefore leaks the pushed exception frame on
           ba0_global.exception.stack (overflowing at BA0_SIZE_EXCEPTION_STACK
           after ~100 such calls).  Fall through to the single ENDTRY instead;
           `s` stays NULL, which the caller reads as "no leader". */
        BA0_TRY {
            if (!bap_is_zero_polynom_mpz(p)) {
                struct bav_variable *v = bap_leader_polynom_mpz(p);
                if (v != (struct bav_variable*)0) {
                    s = ba0_new_printf("%v", v);
                }
            }
        } BA0_CATCH { sdp_copymsg(err, n); return (char*)0; } BA0_ENDTRY;
        return s;
    }

    /* differentiate w.r.t. a single derivation named `der`. */
    static int sdp_diff1(struct bap_polynom_mpz *out, struct bap_polynom_mpz *in,
                         const char *der, char *err, int n) {
        int unknown = 0;
        BA0_TRY {
            struct bav_symbol *s = bav_R_string_to_existing_derivation((char*)der);
            if (s == (struct bav_symbol*)0) {
                strncpy(err, "unknown derivation", n-1); err[n-1]=0;
                unknown = 1;        /* do NOT return inside TRY (frame leak) */
            } else {
                bap_diff_polynom_mpz(out, in, s);
            }
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return unknown ? 1 : 0;
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

    /* C-native ring arithmetic (no materialization to Sage). */
    static int sdp_add(struct bap_polynom_mpz *o, struct bap_polynom_mpz *a,
                       struct bap_polynom_mpz *b, char *err, int n) {
        BA0_TRY { bap_add_polynom_mpz(o, a, b); }
        BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }
    static int sdp_sub(struct bap_polynom_mpz *o, struct bap_polynom_mpz *a,
                       struct bap_polynom_mpz *b, char *err, int n) {
        BA0_TRY { bap_sub_polynom_mpz(o, a, b); }
        BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }
    static int sdp_mul(struct bap_polynom_mpz *o, struct bap_polynom_mpz *a,
                       struct bap_polynom_mpz *b, char *err, int n) {
        BA0_TRY { bap_mul_polynom_mpz(o, a, b); }
        BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }
    static int sdp_neg(struct bap_polynom_mpz *o, struct bap_polynom_mpz *a,
                       char *err, int n) {
        BA0_TRY { bap_neg_polynom_mpz(o, a); }
        BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }

    /* ---- algebraic primitives (Phase A) -------------------------------- */

    /* Resolve a named jet variable "u[x,x]" to a bav_variable*.  A NULL var
       string is returned as NULL (meaning "use leader").  On parse failure the
       returned pointer may be NULL; callers treat that as an error. */
    static struct bav_variable *sdp_var(const char *name) {
        struct bav_variable *v = (struct bav_variable*)0;
        if (name == (const char*)0) return (struct bav_variable*)0;
        ba0_sscanf2((char*)name, "%v", &v);
        return v;
    }

    /* degree of A in the named variable v. */
    static int sdp_degree(struct bap_polynom_mpz *A, const char *var,
                          long *out, char *err, int n) {
        BA0_TRY {
            struct bav_variable *v = sdp_var(var);
            *out = (long)bap_degree_polynom_mpz(A, v);
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }

    /* coefficient of A by (variable v, degree d): out = coeff(A, v^d). */
    static int sdp_coeff(struct bap_polynom_mpz *out, struct bap_polynom_mpz *A,
                         const char *var, long d, char *err, int n) {
        BA0_TRY {
            struct bav_variable *v = sdp_var(var);
            bap_coeff_polynom_mpz(out, A, v, (bav_Idegree)d);
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }

    /* exact quotient out = A / B (B must divide A). */
    static int sdp_exquo(struct bap_polynom_mpz *out, struct bap_polynom_mpz *A,
                         struct bap_polynom_mpz *B, char *err, int n) {
        BA0_TRY { bap_exquo_polynom_mpz(out, A, B); }
        BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }

    /* is B a factor of A?  out_q (may be 0) receives the cofactor. */
    static int sdp_is_factor(struct bap_polynom_mpz *A, struct bap_polynom_mpz *B,
                             struct bap_polynom_mpz *out_q, int *res,
                             char *err, int n) {
        BA0_TRY { *res = bap_is_factor_polynom_mpz(A, B, out_q) ? 1 : 0; }
        BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }

    /* gcd of A and B (G); cofA, cofB may be 0 to skip cofactors. */
    static int sdp_gcd(struct bap_polynom_mpz *G,
                       struct bap_polynom_mpz *cofA, struct bap_polynom_mpz *cofB,
                       struct bap_polynom_mpz *A, struct bap_polynom_mpz *B,
                       char *err, int n) {
        BA0_TRY { baz_gcd_polynom_mpz(G, cofA, cofB, A, B); }
        BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }

    /* content / primpart of A w.r.t. named variable v (or leader if NULL). */
    static int sdp_content(struct bap_polynom_mpz *out, struct bap_polynom_mpz *A,
                           const char *var, char *err, int n) {
        BA0_TRY {
            struct bav_variable *v = sdp_var(var);
            baz_content_polynom_mpz(out, A, v);
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }
    static int sdp_primpart(struct bap_polynom_mpz *out, struct bap_polynom_mpz *A,
                            const char *var, char *err, int n) {
        BA0_TRY {
            struct bav_variable *v = sdp_var(var);
            baz_primpart_polynom_mpz(out, A, v);
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }

    /* resultant of P, Q w.r.t. named variable v: out = expand(resultant).
       Ducos requires positive degree in v for both operands.  When either has
       degree 0 in v we apply the classical degenerate identities, matching
       Sage's Sylvester resultant:
         deg_v(Q)=0, Q nonzero:  res = Q^deg_v(P)
         deg_v(P)=0, P nonzero:  res = P^deg_v(Q)
         either is zero:         res = 0
       (when both are degree 0 the first identity gives Q^0 = 1.) */
    static int sdp_resultant(struct bap_polynom_mpz *out,
                             struct bap_polynom_mpz *P, struct bap_polynom_mpz *Q,
                             const char *var, char *err, int n) {
        BA0_TRY {
            struct bav_variable *v = sdp_var(var);
            if (bap_is_zero_polynom_mpz(P) || bap_is_zero_polynom_mpz(Q)) {
                bap_set_polynom_zero_mpz(out);
            } else {
                bav_Idegree dP = bap_degree_polynom_mpz(P, v);
                bav_Idegree dQ = bap_degree_polynom_mpz(Q, v);
                if (dQ == 0) {
                    bap_pow_polynom_mpz(out, Q, (bav_Idegree)dP);
                } else if (dP == 0) {
                    bap_pow_polynom_mpz(out, P, (bav_Idegree)dQ);
                } else {
                    struct bap_product_mpz *prod = bap_new_product_mpz();
                    bap_resultant2_Ducos_polynom_mpz(prod, P, Q, v);
                    bap_expand_product_mpz(out, prod);
                }
            }
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }

    /* gcd-controlled pseudo-remainder (swell control): R = gcd_prem(A, B, v).
       H (the multiplier product) is discarded here; *hexp returns the total
       degree of H (sum of exponents) as a convenience power signal. */
    static int sdp_gcd_prem(struct bap_polynom_mpz *R, struct bap_polynom_mpz *A,
                            struct bap_polynom_mpz *B, const char *var,
                            long *hexp, char *err, int n) {
        BA0_TRY {
            struct bav_variable *v = sdp_var(var);
            if (v == (struct bav_variable*)0) v = bap_leader_polynom_mpz(B);
            struct bap_product_mpz *H = bap_new_product_mpz();
            baz_gcd_prem_polynom_mpz(R, H, A, B, v);
            long e = 0; long i;
            for (i = 0; i < H->size; i++) e += (long)H->tab[i].exponent;
            *hexp = e;
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return 0;
    }

    /* ---- subresultant polynomial chain --------------------------------- */
    /* Run Lionel Ducos' subresultant PRS (the loop of bap_resultant2_Ducos /
       algo_new) and collect EVERY intermediate subresultant produced -- the
       running B (= S_{d-1}) and Z (= S_e) polynomials -- into the caller's
       array, paired with each one's degree in v.  out[i] is a fresh
       bap_polynom_mpz* (BLAD-stack), deg[i] its degree in v.  Returns the count
       (<= cap) via *count, or -1 (with err set) on a BLAD exception or overflow.

       v must be the highest-ranked variable of both operands (Ducos
       constraint), exactly as for sdp_resultant.  The resultant S_0 is included
       as a degree-0 entry; degenerate degrees (deg_v(P)==0 or deg_v(Q)==0) are
       handled with the classical identities so the chain is never empty.

       The caller post-processes into a {degree -> poly} dict, keeping the
       lowest-index entry per degree (the subresultant of that degree). */
    static int sdp_subres_chain(struct bap_polynom_mpz **out, long *deg,
                                long cap, long *count,
                                struct bap_polynom_mpz *P0,
                                struct bap_polynom_mpz *Q0,
                                const char *var, char *err, int n) {
        BA0_TRY {
            struct bav_variable *v = sdp_var(var);
            long k = 0;
            struct bap_polynom_mpz *P = P0, *Q = Q0;
            if (bap_is_zero_polynom_mpz(P) || bap_is_zero_polynom_mpz(Q)) {
                *count = 0;
            } else {
                bav_Idegree dP = bap_degree_polynom_mpz(P, v);
                bav_Idegree dQ = bap_degree_polynom_mpz(Q, v);
                /* keep P the higher-degree operand (mirror algo_new's swap) */
                if (dP < dQ) {
                    struct bap_polynom_mpz *t = P; P = Q; Q = t;
                    bav_Idegree td = dP; dP = dQ; dQ = td;
                }
                if (dQ == 0) {
                    /* res = Q^dP (degenerate); only S_0 exists. */
                    struct bap_polynom_mpz *r = bap_new_polynom_mpz();
                    bap_pow_polynom_mpz(r, Q, (bav_Idegree)dP);
                    out[k] = r; deg[k] = 0; k++;
                } else {
                    /* The subresultant PRS, mirroring algo_new_COEFF.
                       A = S_{c-1}, B = S_{d-1}, s = lc(S_d). */
                    struct bap_polynom_mpz coeff, s, Z;
                    struct bap_polynom_mpz *A, *B;
                    bav_Idegree delta;
                    bap_init_readonly_polynom_mpz(&coeff);
                    bap_init_polynom_mpz(&s);
                    bap_init_polynom_mpz(&Z);
                    bap_initial2_polynom_mpz(&coeff, Q, v);
                    delta = bap_leading_degree_polynom_mpz(P)
                            - bap_degree_polynom_mpz(Q, v);
                    bap_pow_polynom_mpz(&s, &coeff, delta);
                    A = bap_new_polynom_mpz();
                    B = bap_new_polynom_mpz();
                    bap_set_polynom_mpz(A, Q);
                    {
                        bav_Idegree hd;
                        bap_prem_polynom_mpz(B, &hd, P, Q, v);
                    }
                    bap_neg_polynom_mpz(B, B);
                    /* record the higher-degree starting subresultants:
                       Q (= A, the lower-degree input) is the top of its degree */
                    {
                        struct bap_polynom_mpz *qc = bap_new_polynom_mpz();
                        bap_set_polynom_mpz(qc, Q);
                        out[k] = qc; deg[k] = (long)bap_degree_polynom_mpz(Q, v);
                        k++;
                    }
                    for (;;) {
                        if (k + 2 > cap) { *count = -1;
                            strncpy(err, "subres chain overflow", n-1);
                            err[n-1]=0; break; }
                        if (bap_is_zero_polynom_mpz(B)) break;
                        /* B is the current subresultant S_{d-1}; record it. */
                        {
                            struct bap_polynom_mpz *bc = bap_new_polynom_mpz();
                            bap_set_polynom_mpz(bc, B);
                            out[k] = bc;
                            deg[k] = (long)bap_degree_polynom_mpz(B, v);
                            k++;
                        }
                        delta = bap_leading_degree_polynom_mpz(A)
                                - bap_degree_polynom_mpz(B, v);
                        bap_initial2_polynom_mpz(&coeff, B, v);
                        bap_muldiv2_Lazard_polynom_mpz(&Z, B, &coeff, &s, delta);
                        if (!bap_depend_polynom_mpz(&Z, v)) {
                            /* Z is the (degree-0) resultant S_0 */
                            struct bap_polynom_mpz *zc = bap_new_polynom_mpz();
                            bap_set_polynom_mpz(zc, &Z);
                            out[k] = zc; deg[k] = 0; k++;
                            break;
                        }
                        /* Z = S_e: record it too (its degree may differ from B) */
                        {
                            struct bap_polynom_mpz *zc = bap_new_polynom_mpz();
                            bap_set_polynom_mpz(zc, &Z);
                            out[k] = zc;
                            deg[k] = (long)bap_degree_polynom_mpz(&Z, v);
                            k++;
                        }
                        bap_nsr2_Ducos_polynom_mpz(A, A, B, &Z, &s, v);
                        { struct bap_polynom_mpz *t = A; A = B; B = t; }
                        bap_lcoeff_polynom_mpz(&s, &Z, v);
                    }
                }
                if (*count != -1) *count = k;
            }
        } BA0_CATCH { sdp_copymsg(err, n); return 1; } BA0_ENDTRY;
        return (*count == -1) ? 1 : 0;
    }

    /* ---- product walks (factor / squarefree) --------------------------- */
    /* Compute the squarefree / irreducible factorization product into a
       caller-owned product, then expose its size, numeric factor, and per-factor
       (poly, exponent) so Cython can read them out without holding the product
       struct in .pyx. */
    static struct bap_product_mpz *sdp_factor(struct bap_polynom_mpz *A,
                                              int squarefree_only,
                                              char *err, int n) {
        struct bap_product_mpz *prod = (struct bap_product_mpz*)0;
        BA0_TRY {
            prod = bap_new_product_mpz();
            if (squarefree_only)
                baz_squarefree_polynom_mpz(prod, A);
            else
                baz_factor_polynom_mpz(prod, A);
        } BA0_CATCH { sdp_copymsg(err, n); return (struct bap_product_mpz*)0; } BA0_ENDTRY;
        return prod;
    }
    static long sdp_product_size(struct bap_product_mpz *p) {
        return (long)p->size;
    }
    static mpz_ptr sdp_product_numfactor(struct bap_product_mpz *p) {
        return (mpz_ptr)(p->num_factor);
    }
    static struct bap_polynom_mpz *sdp_product_factor(struct bap_product_mpz *p,
                                                      long i) {
        return &p->tab[i].factor;
    }
    static long sdp_product_exponent(struct bap_product_mpz *p, long i) {
        return (long)p->tab[i].exponent;
    }
    """
    int sdp_init(char *, int)
    long sdp_exc_stack_size()
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
    int sdp_add(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, char *, int)
    int sdp_sub(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, char *, int)
    int sdp_mul(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, char *, int)
    int sdp_neg(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, char *, int)
    # Phase A algebraic primitives
    int sdp_degree(cb.bap_polynom_mpz *, const char *, long *, char *, int)
    int sdp_coeff(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, const char *, long, char *, int)
    int sdp_exquo(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, char *, int)
    int sdp_is_factor(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, int *, char *, int)
    int sdp_gcd(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, char *, int)
    int sdp_content(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, const char *, char *, int)
    int sdp_primpart(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, const char *, char *, int)
    int sdp_resultant(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, const char *, char *, int)
    int sdp_gcd_prem(cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, const char *, long *, char *, int)
    int sdp_subres_chain(cb.bap_polynom_mpz **, long *, long, long *, cb.bap_polynom_mpz *, cb.bap_polynom_mpz *, const char *, char *, int)
    cb.bap_product_mpz *sdp_factor(cb.bap_polynom_mpz *, int, char *, int)
    long sdp_product_size(cb.bap_product_mpz *)
    mpz_ptr sdp_product_numfactor(cb.bap_product_mpz *)
    cb.bap_polynom_mpz *sdp_product_factor(cb.bap_product_mpz *, long)
    long sdp_product_exponent(cb.bap_product_mpz *, long)


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


def exception_stack_size():
    """DIAGNOSTIC: current depth of BLAD's exception-handler stack
    (``ba0_global.exception.stack.size``).  A monotonic climb across
    operations indicates a leaked ``BA0_TRY`` frame."""
    return sdp_exc_stack_size()


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
    err[0] = 0                      # sdp_leader leaves err untouched on the
                                    # "no leader" path; without this the
                                    # uninitialised buffer is read as a bogus
                                    # BladError instead of a clean None.
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


cdef PolyHandle _binop(PolyHandle a, PolyHandle b, int which, long epoch):
    cdef char err[ERRBUF]
    err[0] = 0
    cdef cb.bap_polynom_mpz *out = sdp_new_poly(err, ERRBUF)
    if out == NULL:
        raise BladError(err.decode("utf-8", "replace"))
    cdef int rc
    if which == 0:
        rc = sdp_add(out, a.ptr, b.ptr, err, ERRBUF)
    elif which == 1:
        rc = sdp_sub(out, a.ptr, b.ptr, err, ERRBUF)
    else:
        rc = sdp_mul(out, a.ptr, b.ptr, err, ERRBUF)
    if rc != 0:
        raise BladError(err.decode("utf-8", "replace"))
    return PolyHandle._wrap(out, epoch)


def add(PolyHandle a, PolyHandle b, long epoch):
    return _binop(a, b, 0, epoch)


def sub(PolyHandle a, PolyHandle b, long epoch):
    return _binop(a, b, 1, epoch)


def mul(PolyHandle a, PolyHandle b, long epoch):
    return _binop(a, b, 2, epoch)


def neg(PolyHandle a, long epoch):
    cdef char err[ERRBUF]
    err[0] = 0
    cdef cb.bap_polynom_mpz *out = sdp_new_poly(err, ERRBUF)
    if out == NULL:
        raise BladError(err.decode("utf-8", "replace"))
    if sdp_neg(out, a.ptr, err, ERRBUF) != 0:
        raise BladError(err.decode("utf-8", "replace"))
    return PolyHandle._wrap(out, epoch)


# ---------------------------------------------------------------------------
# Algebraic primitives (Phase A): gcd, factor, squarefree, content/primpart,
# resultant, exact division, degree/coefficient by variable, gcd-prem.
#
# A "var" argument is a BLAD jet *name string* (e.g. "u[x,x]") or None (meaning
# the leader of the relevant polynomial); the high-level layer resolves a Sage
# name / degree-1 element to that string before calling here.
# ---------------------------------------------------------------------------

cdef inline cb.bap_polynom_mpz *_fresh(char *err) except NULL:
    cdef cb.bap_polynom_mpz *p = sdp_new_poly(err, ERRBUF)
    if p == NULL:
        raise BladError(err.decode("utf-8", "replace"))
    return p


def degree_in(PolyHandle a, var):
    """Degree of ``a`` in the named jet variable ``var`` (a BLAD name string)."""
    cdef char err[ERRBUF]
    err[0] = 0
    cdef bytes vb
    cdef long out = 0
    if var is None:
        raise ValueError("degree_in needs a variable name")
    vb = var.encode("utf-8")
    if sdp_degree(a.ptr, vb, &out, err, ERRBUF) != 0:
        raise BladError(err.decode("utf-8", "replace"))
    return int(out)


def coefficient_in(PolyHandle a, var, long d, long epoch):
    """Coefficient of ``a`` viewed in ``var`` at degree ``d`` (BLAD name)."""
    cdef char err[ERRBUF]
    err[0] = 0
    cdef cb.bap_polynom_mpz *out = _fresh(err)
    cdef bytes vb
    if var is None:
        raise ValueError("coefficient_in needs a variable name")
    vb = var.encode("utf-8")
    if sdp_coeff(out, a.ptr, vb, d, err, ERRBUF) != 0:
        raise BladError(err.decode("utf-8", "replace"))
    return PolyHandle._wrap(out, epoch)


def exquo(PolyHandle a, PolyHandle b, long epoch):
    """Exact quotient ``a / b`` (``b`` must divide ``a``)."""
    cdef char err[ERRBUF]
    err[0] = 0
    cdef cb.bap_polynom_mpz *out = _fresh(err)
    if sdp_exquo(out, a.ptr, b.ptr, err, ERRBUF) != 0:
        raise BladError(err.decode("utf-8", "replace"))
    return PolyHandle._wrap(out, epoch)


def is_factor(PolyHandle a, PolyHandle b, long epoch):
    """Return ``(divides, cofactor)``: True iff ``b`` divides ``a``, with the
    exact cofactor ``a/b`` when it does (the cofactor handle is meaningful only
    when ``divides`` is True)."""
    cdef char err[ERRBUF]
    err[0] = 0
    cdef cb.bap_polynom_mpz *q = _fresh(err)
    cdef int res = 0
    if sdp_is_factor(a.ptr, b.ptr, q, &res, err, ERRBUF) != 0:
        raise BladError(err.decode("utf-8", "replace"))
    return bool(res), PolyHandle._wrap(q, epoch)


def gcd(PolyHandle a, PolyHandle b, long epoch):
    """Greatest common divisor of ``a`` and ``b``."""
    cdef char err[ERRBUF]
    err[0] = 0
    cdef cb.bap_polynom_mpz *g = _fresh(err)
    if sdp_gcd(g, NULL, NULL, a.ptr, b.ptr, err, ERRBUF) != 0:
        raise BladError(err.decode("utf-8", "replace"))
    return PolyHandle._wrap(g, epoch)


def content(PolyHandle a, var, long epoch):
    """Content of ``a`` w.r.t. ``var`` (BLAD name) or its leader if ``var`` is
    None."""
    cdef char err[ERRBUF]
    err[0] = 0
    cdef cb.bap_polynom_mpz *out = _fresh(err)
    cdef bytes vb
    if var is None:
        if sdp_content(out, a.ptr, NULL, err, ERRBUF) != 0:
            raise BladError(err.decode("utf-8", "replace"))
    else:
        vb = var.encode("utf-8")
        if sdp_content(out, a.ptr, vb, err, ERRBUF) != 0:
            raise BladError(err.decode("utf-8", "replace"))
    return PolyHandle._wrap(out, epoch)


def primpart(PolyHandle a, var, long epoch):
    """Primitive part of ``a`` w.r.t. ``var`` (BLAD name) or its leader."""
    cdef char err[ERRBUF]
    err[0] = 0
    cdef cb.bap_polynom_mpz *out = _fresh(err)
    cdef bytes vb
    if var is None:
        if sdp_primpart(out, a.ptr, NULL, err, ERRBUF) != 0:
            raise BladError(err.decode("utf-8", "replace"))
    else:
        vb = var.encode("utf-8")
        if sdp_primpart(out, a.ptr, vb, err, ERRBUF) != 0:
            raise BladError(err.decode("utf-8", "replace"))
    return PolyHandle._wrap(out, epoch)


def resultant(PolyHandle a, PolyHandle b, var, long epoch):
    """Resultant of ``a`` and ``b`` w.r.t. ``var`` (a BLAD name string)."""
    cdef char err[ERRBUF]
    err[0] = 0
    cdef cb.bap_polynom_mpz *out = _fresh(err)
    cdef bytes vb
    if var is None:
        raise ValueError("resultant needs a variable name")
    vb = var.encode("utf-8")
    if sdp_resultant(out, a.ptr, b.ptr, vb, err, ERRBUF) != 0:
        raise BladError(err.decode("utf-8", "replace"))
    return PolyHandle._wrap(out, epoch)


def gcd_prem(PolyHandle a, PolyHandle b, var, long epoch):
    """Swell-controlled pseudo-remainder of ``a`` by ``b`` w.r.t. ``var`` (BLAD
    name) or ``leader(b)``.  Returns ``(remainder, hexp)`` where ``hexp`` is the
    total exponent of the multiplier product (a coarse power signal)."""
    cdef char err[ERRBUF]
    err[0] = 0
    cdef cb.bap_polynom_mpz *out = _fresh(err)
    cdef long hexp = 0
    cdef bytes vb
    if var is None:
        if sdp_gcd_prem(out, a.ptr, b.ptr, NULL, &hexp, err, ERRBUF) != 0:
            raise BladError(err.decode("utf-8", "replace"))
    else:
        vb = var.encode("utf-8")
        if sdp_gcd_prem(out, a.ptr, b.ptr, vb, &hexp, err, ERRBUF) != 0:
            raise BladError(err.decode("utf-8", "replace"))
    return PolyHandle._wrap(out, epoch), int(hexp)


DEF SUBRES_CAP = 512

def subresultant_chain(PolyHandle a, PolyHandle b, var, long epoch):
    """Full Ducos subresultant polynomial chain of ``a`` and ``b`` w.r.t. ``var``
    (a BLAD name string, the highest-ranked variable of both).

    Returns a list ``[(degree, PolyHandle), ...]`` of every subresultant
    produced by the PRS (including the resultant ``S_0`` at degree 0 and the
    lower-degree input).  The caller folds this into a ``{degree -> poly}`` dict.
    """
    cdef char err[ERRBUF]
    err[0] = 0
    cdef cb.bap_polynom_mpz *outarr[SUBRES_CAP]
    cdef long degarr[SUBRES_CAP]
    cdef long count = 0
    cdef bytes vb
    if var is None:
        raise ValueError("subresultant_chain needs a variable name")
    vb = var.encode("utf-8")
    if sdp_subres_chain(outarr, degarr, SUBRES_CAP, &count,
                        a.ptr, b.ptr, vb, err, ERRBUF) != 0:
        raise BladError(err.decode("utf-8", "replace"))
    out = []
    cdef long i
    for i in range(count):
        out.append((int(degarr[i]), PolyHandle._wrap(outarr[i], epoch)))
    return out


def _factor_walk(PolyHandle a, int squarefree_only, long epoch):
    """Walk a factorization/squarefree product, returning
    ``(num_factor:int, [(PolyHandle, exponent), ...])``.  Constant / numeric-only
    factors are excluded (the product's numeric part is returned separately)."""
    cdef char err[ERRBUF]
    err[0] = 0
    cdef cb.bap_product_mpz *prod = sdp_factor(a.ptr, squarefree_only, err, ERRBUF)
    if prod == NULL:
        raise BladError(err.decode("utf-8", "replace"))
    cdef long sz = sdp_product_size(prod)
    cdef long i, e
    cdef cb.bap_polynom_mpz *fac
    num = _mpz_to_pyint(<mpz_srcptr> sdp_product_numfactor(prod))
    out = []
    for i in range(sz):
        fac = sdp_product_factor(prod, i)
        e = sdp_product_exponent(prod, i)
        # wrap a fresh copy-free handle onto the factor pointer (it lives on the
        # BLAD stack for the current epoch, like every other handle here)
        out.append((PolyHandle._wrap(fac, epoch), int(e)))
    return int(num), out


def factor(PolyHandle a, long epoch):
    """Irreducible factorization: ``(num_factor:int, [(PolyHandle, mult), ...])``."""
    return _factor_walk(a, 0, epoch)


def squarefree(PolyHandle a, long epoch):
    """Yun squarefree decomposition: ``(num_factor:int, [(PolyHandle, mult), ...])``."""
    return _factor_walk(a, 1, epoch)

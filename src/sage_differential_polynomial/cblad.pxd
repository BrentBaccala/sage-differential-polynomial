# cython: language_level=3
#
# Cython declarations for the subset of BLAD (libblad) used by the
# term-walk differential-polynomial binding.  These mirror the public
# header ~/DifferentialAlgebra/blad-install-c99/include/blad.h.
#
# Only the symbols the binding actually calls are declared.  ba0_int_p is
# ``long int`` on this 64-bit build; bav_I* and bav_zipterm are all aliases
# of it.  Coefficients are GMP mpz_t / mpq_t (we cimport the GMP pxd from
# Sage for the limb-level types and reuse the same mpz_t here).

from sage.libs.gmp.types cimport mpz_t, mpq_t

cdef extern from "blad.h":

    # -- integer typedefs --------------------------------------------------
    ctypedef long ba0_int_p
    ctypedef long bav_Idegree
    ctypedef long bav_Iorder
    ctypedef long bav_Iordering
    ctypedef long bav_Inumber

    # ba0_mpz_t is a macro alias for the GMP mpz_t; BLAD function prototypes
    # use ba0_mpz_t.  We just pass mpz_t (same underlying type).
    ctypedef mpz_t ba0_mpz_t

    # -- ba0 restart levels ------------------------------------------------
    cdef enum ba0_restart_level:
        ba0_init_level
        ba0_reset_level
        ba0_done_level

    # -- ba0 memory mark ---------------------------------------------------
    cdef struct ba0_mark:
        pass

    void ba0_record(ba0_mark *)
    void ba0_restore(ba0_mark *)

    void *ba0_alloc(unsigned long)

    # -- exception machinery ----------------------------------------------
    # ba0_global is a large struct; we read exception.raised via the inline
    # C helper sdp_exception_message() declared at the bottom of this file.
    char *ba0_get_context_analex()

    # -- bas init/teardown -------------------------------------------------
    void bas_restart(ba0_int_p, ba0_int_p)
    void bas_terminate(ba0_restart_level)
    void bas_reset_all_settings()

    # -- string IO (ring build + debug) ------------------------------------
    void ba0_sscanf2(char *, char *, ...)
    void ba0_scanf_printf(char *, char *, ...)
    char *ba0_new_printf(char *, ...)

    # -- bav symbol / variable / term --------------------------------------
    cdef enum bav_typeof_symbol:
        bav_independent_symbol
        bav_dependent_symbol
        bav_operator_symbol
        bav_temporary_symbol

    cdef struct bav_symbol:
        char *ident
        bav_typeof_symbol type
        ba0_int_p index_in_syms
        ba0_int_p derivation_index
        ba0_int_p index_in_pars

    cdef struct bav_variable:
        bav_symbol *root
        ba0_int_p index_in_vars
        # number / order / derivative tables omitted (not poked directly)

    cdef struct bav_rank:
        bav_variable *var
        bav_Idegree deg

    cdef struct bav_term:
        ba0_int_p alloc
        ba0_int_p size
        bav_rank *rg

    void bav_init_term(bav_term *)

    # -- bav ring / ranking ------------------------------------------------
    void bav_set_settings_ordering(char *)
    void bav_push_ordering(bav_Iordering)
    void bav_pull_ordering()
    bint bav_R_ambiguous_symbols()

    bav_symbol *bav_R_string_to_existing_symbol(char *)
    bav_symbol *bav_R_string_to_existing_derivation(char *)
    bav_variable *bav_R_string_to_existing_variable(char *)
    bav_variable *bav_symbol_to_variable(bav_symbol *)

    bav_variable *bav_diff_variable(bav_variable *, bav_symbol *)

    # -- bap polynomial (mpz) ----------------------------------------------
    cdef struct bap_clot_mpz:
        pass

    cdef struct bap_polynom_mpz:
        bav_term total_rank

    void bap_init_polynom_mpz(bap_polynom_mpz *)
    bap_polynom_mpz *bap_new_polynom_mpz()
    void bap_set_polynom_zero_mpz(bap_polynom_mpz *)
    bint bap_is_zero_polynom_mpz(bap_polynom_mpz *)
    bint bap_equal_polynom_mpz(bap_polynom_mpz *, bap_polynom_mpz *)
    ba0_int_p bap_nbmon_polynom_mpz(bap_polynom_mpz *)

    bav_variable *bap_leader_polynom_mpz(bap_polynom_mpz *)
    bav_rank bap_rank_polynom_mpz(bap_polynom_mpz *)

    # differential / algebraic ops
    void bap_diff_polynom_mpz(bap_polynom_mpz *, bap_polynom_mpz *, bav_symbol *)
    void bap_diff2_polynom_mpz(bap_polynom_mpz *, bap_polynom_mpz *, bav_term *)
    void bap_separant_polynom_mpz(bap_polynom_mpz *, bap_polynom_mpz *)
    void bap_separant2_polynom_mpz(bap_polynom_mpz *, bap_polynom_mpz *, bav_variable *)
    void bap_initial_polynom_mpz(bap_polynom_mpz *, bap_polynom_mpz *)
    void bap_prem_polynom_mpz(bap_polynom_mpz *, bav_Idegree *,
                              bap_polynom_mpz *, bap_polynom_mpz *,
                              bav_variable *)

    # -- algebraic primitives (Phase A) ------------------------------------
    bav_Idegree bap_degree_polynom_mpz(bap_polynom_mpz *, bav_variable *)
    void bap_coeff_polynom_mpz(bap_polynom_mpz *, bap_polynom_mpz *,
                               bav_variable *, bav_Idegree)
    bint bap_is_factor_polynom_mpz(bap_polynom_mpz *, bap_polynom_mpz *,
                                   bap_polynom_mpz *)
    void bap_exquo_polynom_mpz(bap_polynom_mpz *, bap_polynom_mpz *,
                               bap_polynom_mpz *)
    void bap_resultant2_Ducos_polynom_mpz(bap_product_mpz *,
                                          bap_polynom_mpz *, bap_polynom_mpz *,
                                          bav_variable *)
    void bap_nsr2_Ducos_polynom_mpz(bap_polynom_mpz *, bap_polynom_mpz *,
                                    bap_polynom_mpz *, bap_polynom_mpz *,
                                    bap_polynom_mpz *, bav_variable *)

    # -- bap_product (factorization / resultant output) --------------------
    cdef struct bap_power_mpz:
        bap_polynom_mpz factor
        bav_Idegree exponent
    cdef struct bap_product_mpz:
        mpz_t num_factor
        ba0_int_p alloc
        ba0_int_p size
        bap_power_mpz *tab
    void bap_init_product_mpz(bap_product_mpz *)
    bap_product_mpz *bap_new_product_mpz()
    void bap_expand_product_mpz(bap_polynom_mpz *, bap_product_mpz *)

    # -- baz higher algebra ------------------------------------------------
    void baz_gcd_polynom_mpz(bap_polynom_mpz *, bap_polynom_mpz *,
                             bap_polynom_mpz *, bap_polynom_mpz *,
                             bap_polynom_mpz *)
    void baz_content_polynom_mpz(bap_polynom_mpz *, bap_polynom_mpz *,
                                 bav_variable *)
    void baz_primpart_polynom_mpz(bap_polynom_mpz *, bap_polynom_mpz *,
                                  bav_variable *)
    void baz_squarefree_polynom_mpz(bap_product_mpz *, bap_polynom_mpz *)
    void baz_factor_polynom_mpz(bap_product_mpz *, bap_polynom_mpz *)
    void baz_gcd_prem_polynom_mpz(bap_polynom_mpz *, bap_product_mpz *,
                                  bap_polynom_mpz *, bap_polynom_mpz *,
                                  bav_variable *)

    # (polynomials are parsed via ba0_sscanf2(str, "%Az", &poly) and printed
    #  via ba0_new_printf("%Az", &poly); no separate bap_scanf/printf needed)

    # -- monomial iterator (read-out term-walk) ----------------------------
    cdef struct bap_itermon_mpz:
        pass
    void bap_begin_itermon_mpz(bap_itermon_mpz *, bap_polynom_mpz *)
    bint bap_outof_itermon_mpz(bap_itermon_mpz *)
    void bap_next_itermon_mpz(bap_itermon_mpz *)
    mpz_t *bap_coeff_itermon_mpz(bap_itermon_mpz *)
    void bap_term_itermon_mpz(bav_term *, bap_itermon_mpz *)

    # -- creator (write-in term-walk) --------------------------------------
    cdef enum bap_typeof_total_rank:
        bap_exact_total_rank
        bap_approx_total_rank

    cdef struct bap_creator_mpz:
        pass
    void bap_begin_creator_mpz(bap_creator_mpz *, bap_polynom_mpz *,
                               bav_term *, bap_typeof_total_rank, ba0_int_p)
    void bap_write_creator_mpz(bap_creator_mpz *, bav_term *, mpz_t)
    void bap_close_creator_mpz(bap_creator_mpz *)


# A tiny C helper compiled inline to read the exception message without
# transcribing the whole ba0_global struct into Cython.
cdef extern from *:
    """
    #include "blad.h"
    static const char *sdp_exception_message(void) {
        return ba0_global.exception.raised;
    }
    """
    const char *sdp_exception_message()

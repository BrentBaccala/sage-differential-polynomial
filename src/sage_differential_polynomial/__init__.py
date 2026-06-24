r"""
sage_differential_polynomial -- a native Sage differential-polynomial ring
backed by BLAD (libblad) via a term-walk Cython binding.

Public API::

    from sage_differential_polynomial import DifferentialPolynomialRing

See :mod:`sage_differential_polynomial.differential_polynomial_ring`.
"""

__version__ = "0.1.0"


class BladNotAvailable(ImportError):
    """Raised when the BLAD C extension could not be imported."""


def _blad_available():
    try:
        from . import _blad  # noqa: F401
        return True
    except Exception:
        return False


# Re-export the high-level types if (and only if) the C extension built.
try:
    from .differential_polynomial_ring import (
        DifferentialPolynomialRing,
        DifferentialPolynomial,
    )
except Exception as _exc:  # optional-dependency gating
    _IMPORT_ERROR = _exc

    def DifferentialPolynomialRing(*args, **kwargs):
        raise BladNotAvailable(
            "the BLAD-backed C extension is not available: %r" % (_IMPORT_ERROR,)
        )

    DifferentialPolynomial = None

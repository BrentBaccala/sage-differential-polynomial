#!/usr/bin/env python
"""
Build script for sage-differential-polynomial.

Links BLAD (libblad.a) STATICALLY with every symbol hidden except the module
init.  Rationale (see ~/project/docs/blad-native-binding-design.md): the pip
``DifferentialAlgebra`` package re-exports all ~2184 BLAD symbols at GLOBAL
DEFAULT visibility (including ``bav_global`` and ``ba0_global``).  Hiding our
copy means the two BLAD engines cannot be interposed onto one another's global
ring, so both packages can be imported in one process safely.
"""
import os
import sys

from setuptools import setup, Extension

# ---------------------------------------------------------------------------
# Locate BLAD.  Override with SAGE_DIFFPOLY_BLAD_PREFIX if installed elsewhere.
# ---------------------------------------------------------------------------
BLAD_PREFIX = os.environ.get(
    "SAGE_DIFFPOLY_BLAD_PREFIX",
    # The PIC build (configured with -fPIC and --enable-gmp=yes against the
    # conda env's GMP, so ba0_mpz_t == Sage's mpz_t and limb-copy is exact).
    os.path.expanduser("~/DifferentialAlgebra/blad-install-pic"),
)
BLAD_INCLUDE = os.path.join(BLAD_PREFIX, "include")
BLAD_STATIC = os.path.join(BLAD_PREFIX, "lib", "libblad.a")

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT_MAP = os.path.join(HERE, "export.map")

# Optional-dependency gating: if BLAD is absent we still want a package that
# imports and reports a clean error.  We detect that here and (when missing)
# build a pure-Python stub module path -- but in this environment BLAD is
# present, so the normal path runs.
blad_present = os.path.exists(BLAD_STATIC) and os.path.exists(
    os.path.join(BLAD_INCLUDE, "blad.h")
)

include_dirs = [BLAD_INCLUDE]
try:
    import sage.env

    include_dirs = sage.env.sage_include_directories() + include_dirs
except Exception:
    # Building without Sage on the path: the .pyx that cimports Sage will fail
    # to compile, which is the intended "needs Sage" signal.  Still provide the
    # conda include for gmp.h / Python.h.
    pass

# conda env include (gmp.h)
conda_inc = os.path.join(sys.prefix, "include")
if os.path.isdir(conda_inc):
    include_dirs.append(conda_inc)

ext_modules = []
if blad_present:
    from Cython.Build import cythonize

    common_compile = ["-fvisibility=hidden", "-O2", "-w"]
    common_link = [
        "-fvisibility=hidden",
        "-Wl,--exclude-libs,ALL",
        "-Wl,--version-script,%s" % EXPORT_MAP,
    ]

    ext = Extension(
        "sage_differential_polynomial._blad",
        sources=["src/sage_differential_polynomial/_blad.pyx"],
        include_dirs=include_dirs,
        libraries=["gmp", "m"],
        extra_objects=[BLAD_STATIC],          # static link, NOT -lblad
        extra_compile_args=common_compile,
        extra_link_args=common_link,
        language="c",
    )
    ext_modules = cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",
            "embedsignature": True,
        },
    )
else:
    sys.stderr.write(
        "WARNING: libblad.a not found at %s; building without the C extension. "
        "The package will import but raise BladNotAvailable on use.\n" % BLAD_STATIC
    )

setup(
    ext_modules=ext_modules,
)

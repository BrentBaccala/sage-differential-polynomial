# sage-differential-polynomial

A native Sage differential-polynomial ring (`DifferentialPolynomialRing` /
`DifferentialPolynomial`) backed by **BLAD** (`libblad`, the C engine behind
François Boulier's `DifferentialAlgebra`) via a **term-walk** Cython binding.

Unlike the existing `bmi/sage` interface — a command facade that shuttles Sage
`SR` expressions across a *string* boundary into BLAD — this is a native Sage
*type*: Sage gains a differential-polynomial ring whose elements are genuine
Parent/Element objects, marshalled to/from BLAD by a GMP-limb-level term-walk
that never crosses a string boundary on the polynomial body.

## Install

```bash
# needs a PIC + system-GMP BLAD build (see docs):
~/miniforge3/envs/sage/bin/pip install -e .
```

BLAD is linked statically with all symbols hidden, so this package coexists in
one process with the pip `DifferentialAlgebra` (each holds its own private BLAD
copy). The GMP allocator is shared with Sage (a no-op `ba0_set_settings_gmp`
keeps Sage's allocators), so Sage `Integer`s and BLAD `mpz` round-trip exactly.

## Use

```python
from sage_differential_polynomial import DifferentialPolynomialRing
R = DifferentialPolynomialRing(QQ, ['u'], ['x'])
p = R('u[x,x]^2 + 3*u[x] - 5')
q = R('u[x,x] - u')
p.prem(q)              # 3*u_x + u^2 - 5
p.differentiate('x')   # total derivative
p.separant(); p.initial(); p.leader()
p.sage()               # lower to an MPolynomialRing jet shadow
```

## Test

```bash
~/miniforge3/envs/sage/bin/sage -t --optional=sage,DifferentialAlgebra src/sage_differential_polynomial/
```

## Status

v1 (Phases 0–5): `differentiate`/`diff2`, `separant`/`separant2`, `initial`,
`leader`, algebraic `prem`, C-native ring arithmetic, jet shadow, SR/sympy
bridge + out-of-process oracle. Single live ring at a time (BLAD's one global
ring). v2 (RosenfeldGroebner, RegularDifferentialChain, change_ranking) is a
declared roadmap, not yet built.

See `~/project/docs/blad-native-binding-design.md` for the full design.

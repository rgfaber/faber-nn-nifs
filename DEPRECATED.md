# DEPRECATED

This package was absorbed into **faber_tweann** in faber_tweann v2.0.0
(2026-07-20). It is no longer developed or released here.

## What to do

Replace any dependency on `faber_nn_nifs` with `faber_tweann`:

```erlang
{deps, [{faber_tweann, "~> 2.0"}]}.
```

The NIFs ship with faber_tweann and are built from source during compilation.
A Rust toolchain is required. See `guides/native-nifs.md` in faber-tweann.

## Why

The split served no purpose. This repository was private and the package was
described as an "enterprise" edition, but there was no second edition: only a
private repository. Because faber-tweann never declared this package as a
dependency, every hex install of faber_tweann silently ran the pure Erlang
fallback, and the native path was never exercised by its own test suite.

The two implementations had drifted apart as a result. `weight_distance_l1/2`
computed Manhattan distance in Erlang and mean absolute deviation in Rust;
`weight_distance_batch/3` had different argument and return types on each
side; `random_weights_batch/1` silently discarded the requested mean and
standard deviation. Each repository asserted its own behaviour, so both test
suites passed.

All four contract bugs are fixed in faber_tweann v2.0.0, and
`test/unit/nif_fallback_conformance_tests.erl` now runs both implementations
over the same inputs and asserts they agree.

## Code location

`native/` moved to `faber-tweann/native/faber_nn_nifs/`
`src/faber_nn_nifs.erl` moved to `faber-tweann/src/`
`test/unit/faber_nn_nifs_tests.erl` moved to `faber-tweann/test/unit/`

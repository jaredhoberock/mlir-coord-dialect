// RUN: opt %s -split-input-file -verify-diagnostics

func.func @from_scalar_bad_shape(%arg0 : i64) {
  // expected-error@+1 {{invalid shape encoding}}
  %0 = coord.from_scalar %arg0 : !coord.coord<7>
  return
}

// -----

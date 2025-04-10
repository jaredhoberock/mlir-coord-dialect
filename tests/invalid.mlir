// RUN: opt %s -split-input-file -verify-diagnostics

func.func @make_scalar_bad_shape(%arg0 : i64) {
  // expected-error@+1 {{expected shape encoding 2 for scalar}}
  %0 = coord.make %arg0 : i64 to !coord.coord<7>
  return
}

// -----

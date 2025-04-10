// RUN: opt %s | FileCheck %s

// ---- Test 1: scalar i64 ----

// CHECK-LABEL: func @make_scalar
// CHECK: %{{.*}} = coord.make %arg0 : i64 to !coord.coord<2>
func.func @make_scalar(%arg0: i64) -> !coord.coord<2> {
  %c = coord.make %arg0 : i64 to !coord.coord<2>
  return %c : !coord.coord<2>
}

// ---- Test 2: Empty coordinate () ----

// CHECK-LABEL: func @make_empty
// CHECK: %{{.+}} = coord.make to !coord.coord<0>
func.func @make_empty() -> !coord.coord<0> {
  %c = coord.make to !coord.coord<0>
  return %c : !coord.coord<0>
}

// ---- Test 3: Tuple with a single i64 element (i64,) ----

// CHECK-LABEL: func @make_single_element_tuple
// CHECK: %{{.+}} = coord.make %arg0 : !coord.coord<2> to !coord.coord<27>
func.func @make_single_element_tuple(%arg0: !coord.coord<2>) -> !coord.coord<27> {
  %c = coord.make %arg0 : !coord.coord<2> to !coord.coord<27>
  return %c : !coord.coord<27>
}

// ---- Test 4: Pair of i64 (i64, i64) ----

// CHECK-LABEL: func @make_pair_of_i64
// CHECK: %{{.+}} = coord.make %arg0, %arg1 : i64, i64 to !coord.coord<107>
func.func @make_pair_of_i64(%arg0: i64, %arg1: i64) -> !coord.coord<107> {
  %c = coord.make %arg0, %arg1 : i64, i64 to !coord.coord<107>
  return %c : !coord.coord<107>
}

// ---- Test 5: Nested Tuple (i64, (i64, i64)) ----

// CHECK-LABEL: func @make_nested
// CHECK: %{{.+}} = coord.make %arg0, %arg1 : i64, !coord.coord<107> to !coord.coord<6575>
func.func @make_nested(%arg0: i64, %arg1: !coord.coord<107>) -> !coord.coord<6575> {
  %c = coord.make %arg0, %arg1 : i64, !coord.coord<107> to !coord.coord<6575>
  return %c : !coord.coord<6575>
}

// ---- Test 6: Tuple with nested pair in middle (i64, (i64, i64), i64) ----

// CHECK-LABEL: func @make_nested_middle
// CHECK: %{{.+}} = coord.make %arg0, %arg1, %arg2 : i64, !coord.coord<107>, i64 to !coord.coord<26299>
func.func @make_nested_middle(%arg0: i64, %arg1: !coord.coord<107>, %arg2: i64) -> !coord.coord<26299> {
  %c = coord.make %arg0, %arg1, %arg2 : i64, !coord.coord<107>, i64 to !coord.coord<26299>
  return %c : !coord.coord<26299>
}

// ---- Test 7: Deeply nested tuple ((i64,i64), (i64,)) ----

// CHECK-LABEL: func @make_deeply_nested
// CHECK: %{{.+}} = coord.make %arg0, %arg1 : !coord.coord<107>, !coord.coord<27> to !coord.coord<93039>
func.func @make_deeply_nested(%arg0: !coord.coord<107>, %arg1: !coord.coord<27>) -> !coord.coord<93039> {
  %c = coord.make %arg0, %arg1 : !coord.coord<107>, !coord.coord<27> to !coord.coord<93039>
  return %c : !coord.coord<93039>
}

// ---- Test 8: Tuple with empty, scalar, and nested scalar ((), i64, (i64,)) ----

// CHECK-LABEL: func @make_mixed_tuple
// CHECK: %{{.+}} = coord.make %arg0, %arg1, %arg2 : !coord.coord<0>, !coord.coord<2>, !coord.coord<27> to !coord.coord<4719>
func.func @make_mixed_tuple(
  %arg0: !coord.coord<0>,
  %arg1: !coord.coord<2>,
  %arg2: !coord.coord<27>
) -> !coord.coord<4719> {
  %c = coord.make %arg0, %arg1, %arg2 : !coord.coord<0>, !coord.coord<2>, !coord.coord<27> to !coord.coord<4719>
  return %c : !coord.coord<4719>
}

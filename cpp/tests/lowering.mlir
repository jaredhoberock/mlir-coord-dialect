// RUN: opt --convert-to-llvm %s | FileCheck %s

// -----

// CHECK-LABEL: llvm.func @make_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_empty() -> tuple<> {
  %c = coord.make_tuple : tuple<>
  return %c : tuple<>
}

// -----

// CHECK-LABEL: llvm.func @make_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_i64(%arg0: i64) -> tuple<i64> {
  %c = coord.make_tuple(%arg0 : i64) : tuple<i64>
  return %c : tuple<i64>
}

// -----

// CHECK-LABEL: func @make_i64_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_i64_i64(%a: i64, %b: i64) -> tuple<i64,i64> {
  %c = coord.make_tuple(%a, %b : i64, i64) : tuple<i64,i64>
  return %c : tuple<i64,i64>
}

// -----

// CHECK-LABEL: func @make_nested
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_nested(%a: i64, %b: tuple<i64,i64>) -> tuple<i64,tuple<i64,i64>> {
  %c = coord.make_tuple(%a, %b : i64, tuple<i64,i64>) : tuple<i64,tuple<i64,i64>>
  return %c : tuple<i64,tuple<i64,i64>>
}

// -----

// CHECK-LABEL: func @make_nested_middle
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_nested_middle(%arg0: i64, %arg1: tuple<i64,i64>, %arg2: i64) -> tuple<i64,tuple<i64,i64>,i64> {
  %c = coord.make_tuple(%arg0, %arg1, %arg2 : i64, tuple<i64,i64>, i64) : tuple<i64,tuple<i64,i64>,i64>
  return %c : tuple<i64,tuple<i64,i64>,i64>
}

// -----

// CHECK-LABEL: func @make_deeply_nested
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_deeply_nested(%arg0: tuple<i64,i64>, %arg1: tuple<i64>) -> tuple<tuple<i64,i64>,tuple<i64>> {
  %c = coord.make_tuple(%arg0, %arg1 : tuple<i64,i64>, tuple<i64>) : tuple<tuple<i64,i64>,tuple<i64>>
  return %c : tuple<tuple<i64,i64>,tuple<i64>>
}

// -----

// CHECK-LABEL: func @make_mixed_tuple
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_mixed_tuple(
  %arg0: tuple<>,
  %arg1: i64,
  %arg2: tuple<i64>
) -> tuple<tuple<>, i64, tuple<i64>> {
  %c = coord.make_tuple(%arg0, %arg1, %arg2 : tuple<>, i64, tuple<i64>) : tuple<tuple<>, i64, tuple<i64>>
  return %c : tuple<tuple<>, i64, tuple<i64>>
}

// -----

// CHECK-LABEL: func @sum_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.add
// CHECK: llvm.return
// CHECK: i1
func.func @sum_empty(%a: tuple<>, %b: tuple<>) -> tuple<> {
  %0 = coord.sum %a, %b : tuple<>
  return %0 : tuple<>
}

// -----

// CHECK-LABEL: func @sum_scalar
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.add
// CHECK: llvm.return
// CHECK: i64
func.func @sum_scalar(%a: i64, %b: i64) -> i64 {
  %0 = coord.sum %a, %b : i64
  return %0 : i64
}

// -----

// CHECK-LABEL: func @sum_pair
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.add
// CHECK: vector<2xi64>
// CHECK: llvm.return
func.func @sum_pair(%a: tuple<i64,i64>, %b: tuple<i64,i64>) -> tuple<i64,i64> {
  %0 = coord.sum %a, %b : tuple<i64,i64>
  return %0 : tuple<i64,i64>
}

// -----

// CHECK-LABEL: func @sum_nested
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.add
// CHECK: vector<3xi64>
// CHECK: llvm.return
func.func @sum_nested(%a: tuple<i64,tuple<i64,i64>>, %b: tuple<i64,tuple<i64,i64>>) -> tuple<i64,tuple<i64,i64>> {
  %c = coord.sum %a, %b : tuple<i64,tuple<i64,i64>>
  return %c : tuple<i64,tuple<i64,i64>>
}

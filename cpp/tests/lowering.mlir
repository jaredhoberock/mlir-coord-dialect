// RUN: mlir-opt --convert-to-llvm %s | FileCheck %s

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

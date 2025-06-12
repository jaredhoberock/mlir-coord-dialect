// RUN: mlir-opt %s | FileCheck %s

// ----

// CHECK-LABEL: func @concrete_operands
// CHECK: %[[RES:.*]] = coord.weak_product %arg0, %arg1
// CHECK: return %[[RES]] : i64
func.func @concrete_operands(%a: tuple<i64,i64>, %b: tuple<i64,i64>) -> i64 {
  %res = coord.weak_product %a, %b : tuple<i64,i64>, tuple<i64,i64> -> i64
  return %res : i64
}

// ----

// CHECK-LABEL: func @poly_operands
// CHECK: %[[RES:.*]] = coord.weak_product %arg0, %arg1
// CHECK: return %[[RES]] : !coord.poly<2>
func.func @poly_operands(%a: !coord.poly<0>, %b: !coord.poly<1>) -> !coord.poly<2> {
  %res = coord.weak_product %a, %b : !coord.poly<0>, !coord.poly<1> -> !coord.poly<2>
  return %res : !coord.poly<2>
}

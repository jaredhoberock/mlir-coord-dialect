// RUN: mlir-opt %s | FileCheck %s

// ---- i64

// CHECK-LABEL: func @zero_i64
// CHECK: %[[RES:.*]] = coord.zero
// CHECK: return %[[RES]] : i64

func.func @zero_i64() -> i64 {
  %res = coord.zero : i64
  return %res : i64
}

// ---- empty tuple

// CHECK-LABEL: func @zero_empty
// CHECK: %[[RES:.*]] = coord.zero
// CHECK: return %[[RES]] : tuple<>

func.func @zero_empty() -> tuple<> {
  %res = coord.zero : tuple<>
  return %res : tuple<>
}

// ---- nested tuple

// CHECK-LABEL: func @zero_nested
// CHECK: %[[RES:.*]] = coord.zero
// CHECK: return %[[RES]] : tuple

func.func @zero_nested() -> tuple<i64, tuple<>, tuple<i64, i64, i64>> {
  %res = coord.zero : tuple<i64, tuple<>, tuple<i64, i64, i64>>
  return %res : tuple<i64, tuple<>, tuple<i64, i64, i64>>
}

// ---- poly

// CHECK-LABEL: func @zero_poly
// CHECK: %[[RES:.*]] = coord.zero
// CHECK: return %[[RES]] : !coord.poly

func.func @zero_poly() -> !coord.poly<0> {
  %res = coord.zero : !coord.poly<0>
  return %res : !coord.poly<0>
}


// RUN: opt %s | FileCheck %s

// ---- Test 1: i64 operands

// CHECK-LABEL: func @sum_scalar
// CHECK: %[[SUM:.*]] = coord.sum %arg0, %arg1 : i64
// CHECK: return %[[SUM]] : i64

func.func @sum_scalar(%arg0: i64, %arg1: i64) -> i64 {
  %sum = coord.sum %arg0, %arg1 : i64
  return %sum : i64
}

// ---- Test 2: empty tuple operands

// CHECK-LABEL: func @sum_empty
// CHECK: %[[SUM:.*]] = coord.sum %arg0, %arg1 : tuple<>
// CHECK: return %[[SUM]] : tuple<>

func.func @sum_empty(%arg0: tuple<>, %arg1: tuple<>) -> tuple<> {
  %sum = coord.sum %arg0, %arg1 : tuple<>
  return %sum : tuple<>
}

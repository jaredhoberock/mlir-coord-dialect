// RUN: opt %s | FileCheck %s

// ---- Test 1: scalar coords

// CHECK-LABEL: func @old_sum_scalar
// CHECK: %[[SUM:.*]] = coord.sum %arg0, %arg1 : !coord.coord<2>
// CHECK: return %[[SUM]] : !coord.coord<2>

func.func @old_sum_scalar(%arg0: !coord.coord<2>, %arg1: !coord.coord<2>) -> !coord.coord<2> {
  %sum = coord.sum %arg0, %arg1 : !coord.coord<2>
  return %sum : !coord.coord<2>
}

// ---- Test 2: empty coords

// CHECK-LABEL: func @old_sum_empty
// CHECK: %[[SUM:.*]] = coord.sum %arg0, %arg1 : !coord.coord<0>
// CHECK: return %[[SUM]] : !coord.coord<0>

func.func @old_sum_empty(%arg0: !coord.coord<0>, %arg1: !coord.coord<0>) -> !coord.coord<0> {
  %sum = coord.sum %arg0, %arg1 : !coord.coord<0>
  return %sum : !coord.coord<0>
}

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

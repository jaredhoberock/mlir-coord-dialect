// RUN: opt %s | FileCheck %s

// ---- Test 1: scalar coords

// CHECK-LABEL: func @sum_scalar
// CHECK: %[[SUM:.*]] = coord.sum %arg0, %arg1 : !coord.coord<2>
// CHECK: return %[[SUM]] : !coord.coord<2>

func.func @sum_scalar(%arg0: !coord.coord<2>, %arg1: !coord.coord<2>) -> !coord.coord<2> {
  %sum = coord.sum %arg0, %arg1 : !coord.coord<2>
  return %sum : !coord.coord<2>
}

// ---- Test 2: empty coords

// CHECK-LABEL: func @sum_empty
// CHECK: %[[SUM:.*]] = coord.sum %arg0, %arg1 : !coord.coord<0>
// CHECK: return %[[SUM]] : !coord.coord<0>

func.func @sum_empty(%arg0: !coord.coord<0>, %arg1: !coord.coord<0>) -> !coord.coord<0> {
  %sum = coord.sum %arg0, %arg1 : !coord.coord<0>
  return %sum : !coord.coord<0>
}

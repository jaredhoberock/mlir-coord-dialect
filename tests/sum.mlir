// RUN: opt %s | FileCheck %s

// CHECK-LABEL: func @test_sum
// CHECK: %[[SUM:.*]] = coord.sum %arg0, %arg1 : !coord.coord<2>
// CHECK: return %[[SUM]] : !coord.coord<2>

func.func @test_sum(%arg0: !coord.coord<2>, %arg1: !coord.coord<2>) -> !coord.coord<2> {
  %sum = coord.sum %arg0, %arg1 : !coord.coord<2>
  return %sum : !coord.coord<2>
}

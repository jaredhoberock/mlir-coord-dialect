// RUN: opt %s | FileCheck %s

// CHECK-LABEL: func @test_scalar
// CHECK: %{{.+}} = coord.from_scalar %arg0
func.func @test_scalar(%arg0 : i64) -> !coord.coord<2> {
  %0 = coord.from_scalar %arg0 : !coord.coord<2>
  return %0 : !coord.coord<2>
}

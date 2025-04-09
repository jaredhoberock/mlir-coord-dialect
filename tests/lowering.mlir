// RUN: opt --convert-to-llvm %s | FileCheck %s

// CHECK-LABEL: llvm.func @from_scalar
// CHECK: llvm.return
func.func @from_scalar(%arg0: i64) -> !coord.coord<2> {
  %0 = coord.from_scalar %arg0 : !coord.coord<2>
  return %0 : !coord.coord<2>
}

// -----

// CHECK-LABEL: func @sum_scalar
// CHECK: llvm.add
// CHECK: llvm.return
// CHECK: vector<1xi64>
func.func @sum_scalar(%a: !coord.coord<2>, %b: !coord.coord<2>) -> !coord.coord<2> {
  %0 = coord.sum %a, %b : !coord.coord<2>
  return %0 : !coord.coord<2>
}

// -----
// CHECK-LABEL: func @sum_pair
// CHECK: llvm.add
// CHECK: vector<2xi64>
// CHECK: llvm.return
func.func @sum_pair(%a: !coord.coord<107>, %b: !coord.coord<107>) -> !coord.coord<107> {
  %0 = coord.sum %a, %b : !coord.coord<107>
  return %0 : !coord.coord<107>
}

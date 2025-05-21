// RUN: opt -pass-pipeline='builtin.module(monomorphize-trait,convert-to-llvm)' %s | FileCheck %s

trait.trait @Coord {
  func.func private @sum(!trait.self, !trait.self) -> !trait.self 
}

trait.impl @Coord for !coord.coord {
  func.func private @sum(%a: !coord.coord, %b: !coord.coord) -> !coord.coord {
    %res = coord.sum %a, %b : !coord.coord
    return %res : !coord.coord
  }
}

// -----

// CHECK-LABEL: func @sum_empty_tuple
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
// CHECK: i1
func.func @sum_empty_tuple(%a: tuple<>, %b: tuple<>) -> tuple<> {
  %0 = trait.method.call @Coord::@sum<tuple<>>(%a, %b) : (!trait.self, !trait.self) -> !trait.self to (tuple<>,tuple<>) -> tuple<>
  return %0 : tuple<>
}

// -----

// CHECK-LABEL: func @sum_scalar
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call @__trait_Coord_impl_i64_sum
// CHECK: llvm.return
// CHECK: i64
func.func @sum_scalar(%a: i64, %b: i64) -> i64 {
  %0 = trait.method.call @Coord::@sum<i64>(%a, %b) : (!trait.self, !trait.self) -> !trait.self to (i64,i64) -> i64
  return %0 : i64
}

// -----

// CHECK-LABEL: func @sum_pair
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call @"__trait_Coord_impl_tuple
// CHECK: vector<2xi64>
// CHECK: llvm.return
func.func @sum_pair(%a: tuple<i64,i64>, %b: tuple<i64,i64>) -> tuple<i64,i64> {
  %0 = trait.method.call @Coord::@sum<tuple<i64,i64>>(%a, %b) : (!trait.self, !trait.self) -> !trait.self to (tuple<i64,i64>,tuple<i64,i64>) -> tuple<i64,i64>
  return %0 : tuple<i64,i64>
}

// -----

// CHECK-LABEL: func @sum_nested
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call @"__trait_Coord_impl_tuple
// CHECK: vector<3xi64>
// CHECK: llvm.return
!T = tuple<i64,tuple<i64,i64>>
func.func @sum_nested(%a: !T, %b: !T) -> !T {
  %res = trait.method.call @Coord::@sum<!T>(%a, %b) : (!trait.self,!trait.self) -> !trait.self to (!T,!T) -> !T
  return %res : !T
}

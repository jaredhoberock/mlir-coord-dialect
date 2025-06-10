// RUN: mlir-opt -pass-pipeline='builtin.module(monomorphize-trait,convert-to-llvm)' %s | FileCheck %s

trait.trait @Coord {
  func.func private @add(!trait.self, !trait.self) -> !trait.self 
  func.func private @sub(!trait.self, !trait.self) -> !trait.self
  func.func private @inner_product(!trait.self, !trait.self) -> i64
}

trait.impl @Coord for !coord.coord {
  func.func private @add(%a: !coord.coord, %b: !coord.coord) -> !coord.coord {
    %res = coord.add %a, %b : !coord.coord
    return %res : !coord.coord
  }

  func.func private @sub(%a: !coord.coord, %b: !coord.coord) -> !coord.coord {
    %res = coord.sub %a, %b : !coord.coord
    return %res : !coord.coord
  }

  func.func private @inner_product(%a: !coord.coord, %b: !coord.coord) -> i64 {
    %res = coord.inner_product %a, %b : !coord.coord
    return %res : i64
  }
}

// -----

// CHECK-LABEL: func @add_empty_tuple
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
// CHECK: !llvm.struct<()>
func.func @add_empty_tuple(%a: tuple<>, %b: tuple<>) -> tuple<> {
  %0 = trait.method.call @Coord::@add<tuple<>>(%a, %b) : (!trait.self, !trait.self) -> !trait.self to (tuple<>,tuple<>) -> tuple<>
  return %0 : tuple<>
}

// -----

// CHECK-LABEL: func @add_scalar
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call @__trait_Coord_impl_i64_add
// CHECK: llvm.return
// CHECK: i64
func.func @add_scalar(%a: i64, %b: i64) -> i64 {
  %0 = trait.method.call @Coord::@add<i64>(%a, %b) : (!trait.self, !trait.self) -> !trait.self to (i64,i64) -> i64
  return %0 : i64
}

// -----

// CHECK-LABEL: func @add_pair
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call @"__trait_Coord_impl_tuple
// CHECK: !llvm.struct<(i64, i64)>
// CHECK: llvm.return
func.func @add_pair(%a: tuple<i64,i64>, %b: tuple<i64,i64>) -> tuple<i64,i64> {
  %0 = trait.method.call @Coord::@add<tuple<i64,i64>>(%a, %b) : (!trait.self, !trait.self) -> !trait.self to (tuple<i64,i64>,tuple<i64,i64>) -> tuple<i64,i64>
  return %0 : tuple<i64,i64>
}

// -----

// CHECK-LABEL: func @add_nested
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call @"__trait_Coord_impl_tuple
// CHECK: !llvm.struct<(i64, struct<(i64, i64)>)>
// CHECK: llvm.return
!T = tuple<i64,tuple<i64,i64>>
func.func @add_nested(%a: !T, %b: !T) -> !T {
  %res = trait.method.call @Coord::@add<!T>(%a, %b) : (!trait.self,!trait.self) -> !trait.self to (!T,!T) -> !T
  return %res : !T
}


// -----

// CHECK-LABEL: func @sub_empty_tuple
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
// CHECK: !llvm.struct<()>
func.func @sub_empty_tuple(%a: tuple<>, %b: tuple<>) -> tuple<> {
  %0 = trait.method.call @Coord::@sub<tuple<>>(%a, %b) : (!trait.self, !trait.self) -> !trait.self to (tuple<>,tuple<>) -> tuple<>
  return %0 : tuple<>
}

// -----

// CHECK-LABEL: func @sub_scalar
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call @__trait_Coord_impl_i64_sub
// CHECK: llvm.return
// CHECK: i64
func.func @sub_scalar(%a: i64, %b: i64) -> i64 {
  %0 = trait.method.call @Coord::@sub<i64>(%a, %b) : (!trait.self, !trait.self) -> !trait.self to (i64,i64) -> i64
  return %0 : i64
}

// -----

// CHECK-LABEL: func @sub_pair
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call @"__trait_Coord_impl_tuple
// CHECK: !llvm.struct<(i64, i64)>
// CHECK: llvm.return
func.func @sub_pair(%a: tuple<i64,i64>, %b: tuple<i64,i64>) -> tuple<i64,i64> {
  %0 = trait.method.call @Coord::@sub<tuple<i64,i64>>(%a, %b) : (!trait.self, !trait.self) -> !trait.self to (tuple<i64,i64>,tuple<i64,i64>) -> tuple<i64,i64>
  return %0 : tuple<i64,i64>
}

// -----

// CHECK-LABEL: func @sub_nested
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call @"__trait_Coord_impl_tuple
// CHECK: !llvm.struct<(i64, struct<(i64, i64)>)>
// CHECK: llvm.return
func.func @sub_nested(%a: !T, %b: !T) -> !T {
  %res = trait.method.call @Coord::@sub<!T>(%a, %b) : (!trait.self,!trait.self) -> !trait.self to (!T,!T) -> !T
  return %res : !T
}

// -----

// CHECK-LABEL: func @inner_product_empty_tuple
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
// CHECK: i64
func.func @inner_product_empty_tuple(%a: tuple<>, %b: tuple<>) -> i64 {
  %0 = trait.method.call @Coord::@inner_product<tuple<>>(%a, %b) : (!trait.self, !trait.self) -> i64 to (tuple<>,tuple<>) -> i64
  return %0 : i64
}

// -----

// CHECK-LABEL: func @inner_product_scalar
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call @__trait_Coord_impl_i64_inner_product
// CHECK: llvm.return
// CHECK: i64
func.func @inner_product_scalar(%a: i64, %b: i64) -> i64 {
  %0 = trait.method.call @Coord::@inner_product<i64>(%a, %b) : (!trait.self, !trait.self) -> i64 to (i64,i64) -> i64
  return %0 : i64
}

// -----

// CHECK-LABEL: func @inner_product_pair
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call @"__trait_Coord_impl_tuple
// CHECK: !llvm.struct<(i64, i64)>
// CHECK: llvm.return
func.func @inner_product_pair(%a: tuple<i64,i64>, %b: tuple<i64,i64>) -> i64 {
  %0 = trait.method.call @Coord::@inner_product<tuple<i64,i64>>(%a, %b) : (!trait.self, !trait.self) -> i64 to (tuple<i64,i64>,tuple<i64,i64>) -> i64
  return %0 : i64
}

// -----

// CHECK-LABEL: func @inner_product_nested
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call @"__trait_Coord_impl_tuple
// CHECK: !llvm.struct<(i64, struct<(i64, i64)>)>
// CHECK: llvm.return
func.func @inner_product_nested(%a: !T, %b: !T) -> i64 {
  %res = trait.method.call @Coord::@inner_product<!T>(%a, %b) : (!trait.self,!trait.self) -> i64 to (!T,!T) -> i64
  return %res : i64
}

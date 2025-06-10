// RUN: mlir-opt --pass-pipeline="builtin.module(monomorphize-trait,convert-to-llvm)" %s | FileCheck %s

// -----

// CHECK-LABEL: func @add_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
// CHECK: !llvm.struct<()>
func.func @add_empty(%a: tuple<>, %b: tuple<>) -> tuple<> {
  %0 = coord.add %a, %b : tuple<>
  return %0 : tuple<>
}

// -----

// CHECK-LABEL: func @add_scalar
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.add
// CHECK: llvm.return
// CHECK: i64
func.func @add_scalar(%a: i64, %b: i64) -> i64 {
  %0 = coord.add %a, %b : i64
  return %0 : i64
}

// -----

// CHECK-LABEL: llvm.func @add_pair
// CHECK-SAME: !llvm.struct<(i64, i64)>
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK-COUNT-2: llvm.add {{.*}} : i64
// CHECK: llvm.return {{.*}} : !llvm.struct<(i64, i64)>
func.func @add_pair(%a: tuple<i64,i64>, %b: tuple<i64,i64>) -> tuple<i64,i64> {
  %0 = coord.add %a, %b : tuple<i64,i64>
  return %0 : tuple<i64,i64>
}

// -----

// CHECK-LABEL: llvm.func @add_nested
// CHECK-SAME: !llvm.struct<(i64, struct<(i64, i64)>)>
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK-COUNT-3: llvm.add {{.*}} : i64
// CHECK: llvm.return {{.*}} : !llvm.struct<(i64, struct<(i64, i64)>)>
func.func @add_nested(%a: tuple<i64,tuple<i64,i64>>, %b: tuple<i64,tuple<i64,i64>>) -> tuple<i64,tuple<i64,i64>> {
  %c = coord.add %a, %b : tuple<i64,tuple<i64,i64>>
  return %c : tuple<i64,tuple<i64,i64>>
}

// -----

// CHECK-LABEL: func @sub_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
// CHECK: !llvm.struct<()>
func.func @sub_empty(%a: tuple<>, %b: tuple<>) -> tuple<> {
  %0 = coord.sub %a, %b : tuple<>
  return %0 : tuple<>
}

// -----

// CHECK-LABEL: func @sub_scalar
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.sub
// CHECK: llvm.return
// CHECK: i64
func.func @sub_scalar(%a: i64, %b: i64) -> i64 {
  %0 = coord.sub %a, %b : i64
  return %0 : i64
}

// -----

// CHECK-LABEL: llvm.func @sub_pair
// CHECK-SAME: !llvm.struct<(i64, i64)>
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK-COUNT-2: llvm.sub {{.*}} : i64
// CHECK: llvm.return {{.*}} : !llvm.struct<(i64, i64)>
func.func @sub_pair(%a: tuple<i64,i64>, %b: tuple<i64,i64>) -> tuple<i64,i64> {
  %0 = coord.sub %a, %b : tuple<i64,i64>
  return %0 : tuple<i64,i64>
}

// -----

// CHECK-LABEL: llvm.func @sub_nested
// CHECK-SAME: !llvm.struct<(i64, struct<(i64, i64)>)>
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK-COUNT-3: llvm.sub {{.*}} : i64
// CHECK: llvm.return {{.*}} : !llvm.struct<(i64, struct<(i64, i64)>)>
func.func @sub_nested(%a: tuple<i64,tuple<i64,i64>>, %b: tuple<i64,tuple<i64,i64>>) -> tuple<i64,tuple<i64,i64>> {
  %c = coord.sub %a, %b : tuple<i64,tuple<i64,i64>>
  return %c : tuple<i64,tuple<i64,i64>>
}

// -----

// CHECK-LABEL: func @inner_product_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: !llvm.struct<()>
// CHECK: llvm.mlir.constant(0 : i64)
// CHECK: llvm.return {{.*}} : i64
func.func @inner_product_empty(%a: tuple<>, %b: tuple<>) -> i64 {
  %0 = coord.inner_product %a, %b : tuple<>
  return %0 : i64
}

// -----

// CHECK-LABEL: func @inner_product_scalar
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.mul
// CHECK: llvm.return {{.*}} : i64
func.func @inner_product_scalar(%a: i64, %b: i64) -> i64 {
  %0 = coord.inner_product %a, %b : i64
  return %0 : i64
}

// -----

// CHECK-LABEL: llvm.func @inner_product_pair
// CHECK-SAME: !llvm.struct<(i64, i64)>
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK-COUNT-2: llvm.mul {{.*}} : i64
// CHECK: llvm.return {{.*}} : i64
func.func @inner_product_pair(%a: tuple<i64,i64>, %b: tuple<i64,i64>) -> i64 {
  %0 = coord.inner_product %a, %b : tuple<i64,i64>
  return %0 : i64
}

// -----

// CHECK-LABEL: llvm.func @inner_product_nested
// CHECK-SAME: !llvm.struct<(i64, struct<(i64, i64)>)>
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK-COUNT-3: llvm.mul {{.*}} : i64
// CHECK: llvm.return {{.*}} : i64
func.func @inner_product_nested(%a: tuple<i64,tuple<i64,i64>>, %b: tuple<i64,tuple<i64,i64>>) -> i64 {
  %0 = coord.inner_product %a, %b : tuple<i64,tuple<i64,i64>>
  return %0 : i64
}

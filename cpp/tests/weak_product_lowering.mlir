// RUN: mlir-opt --pass-pipeline="builtin.module(monomorphize-trait,convert-to-llvm)" %s | FileCheck %s

// -----
// CHECK-LABEL: func @scalar
// CHECK-COUNT-1: llvm.mul
// CHECK-NOT: llvm.mul
func.func @scalar(
  %a: i64,
  %b: i64
) -> i64 {
  %res = coord.weak_product %a, %b : i64, i64 -> i64
  return %res : i64
}

// -----
// CHECK-LABEL: func @congruent_flat_tuples
// CHECK-COUNT-3: llvm.mul
// CHECK-NOT: llvm.mul
func.func @congruent_flat_tuples(
  %a: tuple<i64,i64,i64>,
  %b: tuple<i64,i64,i64>
) -> i64 {
  %res = coord.weak_product %a, %b : tuple<i64,i64,i64>, tuple<i64,i64,i64> -> i64
  return %res : i64
}

// -----
// CHECK-LABEL: func @congruent_nested_tuples
// CHECK-COUNT-3: llvm.mul
// CHECK-NOT: llvm.mul
func.func @congruent_nested_tuples(
  %a: tuple<i64, tuple<i64, i64>>,
  %b: tuple<i64, tuple<i64, i64>>
) -> i64 {
  %res = coord.weak_product %a, %b : tuple<i64, tuple<i64,i64>>, tuple<i64,tuple<i64,i64>> -> i64
  return %res : i64
}

// -----
// CHECK-LABEL: func @empty_tuples
// CHECK-COUNT-1: llvm.mlir.constant(0 : i64)
// CHECK-NOT: llvm.mlir.constant
func.func @empty_tuples(
  %a: tuple<>,
  %b: tuple<>
) -> i64 {
  %res = coord.weak_product %a, %b : tuple<>, tuple<> -> i64
  return %res : i64
}

// -----
// CHECK-LABEL: func @scalar_deep_tuple
// CHECK-COUNT-3: llvm.mul
// CHECK-NOT: llvm.mul
func.func @scalar_deep_tuple(
  %a: i64,
  %b: tuple<i64, tuple<i64, i64>>
) -> tuple<i64, tuple<i64, i64>> {
  %res = coord.weak_product %a, %b : i64, tuple<i64, tuple<i64, i64>> -> tuple<i64, tuple<i64, i64>>
  return %res : tuple<i64, tuple<i64, i64>>
}

// -----
// CHECK-LABEL: func @deep_congruent_tuples
// CHECK-COUNT-4: llvm.mul
// CHECK-NOT: llvm.mul
func.func @deep_congruent_tuples(
  %a: tuple<i64, tuple<i64, tuple<i64, i64>>>,
  %b: tuple<i64, tuple<i64, tuple<i64, i64>>>
) -> i64 {
  %res = coord.weak_product %a, %b :
    tuple<i64, tuple<i64, tuple<i64, i64>>>,
    tuple<i64, tuple<i64, tuple<i64, i64>>>
  -> i64
  return %res : i64
}

// -----
// CHECK-LABEL: func @partially_congruent_tuples
// CHECK-COUNT-8: llvm.mul
// CHECK-NOT: llvm.mul
func.func @partially_congruent_tuples(
  %a: tuple<tuple<i64, i64>, tuple<i64, i64>>,
  %b: tuple<tuple<tuple<i64, i64>, tuple<i64, i64>>, tuple<tuple<i64, i64>, tuple<i64, i64>>>
) -> tuple<i64, i64> {
  %res = coord.weak_product %a, %b :
    tuple<tuple<i64, i64>, tuple<i64, i64>>,
    tuple<tuple<tuple<i64, i64>, tuple<i64, i64>>, tuple<tuple<i64, i64>, tuple<i64, i64>>>
  -> tuple<i64, i64>
  return %res : tuple<i64, i64>
}

// -----
// CHECK-LABEL: func @tuple_with_empty_tuple_element
// CHECK-COUNT-1: llvm.mul
// CHECK-NOT: llvm.mul
func.func @tuple_with_empty_tuple_element(
  %a: tuple<tuple<>, i64>,
  %b: tuple<tuple<>, i64>
) -> i64 {
  %res = coord.weak_product %a, %b :
    tuple<tuple<>, i64>,
    tuple<tuple<>, i64>
  -> i64
  return %res : i64
}

// -----
// CHECK-LABEL: func @deeply_nested_with_empty_tuple_branches
// CHECK-COUNT-2: llvm.mul
// CHECK-NOT: llvm.mul
func.func @deeply_nested_with_empty_tuple_branches(
  %a: tuple<tuple<>, tuple<i64, tuple<tuple<>, i64>>>,
  %b: tuple<tuple<>, tuple<i64, tuple<tuple<>, i64>>>
) -> i64 {
  %res = coord.weak_product %a, %b :
    tuple<tuple<>, tuple<i64, tuple<tuple<>, i64>>>,
    tuple<tuple<>, tuple<i64, tuple<tuple<>, i64>>>
  -> i64
  return %res : i64
}

// -----
// CHECK-LABEL: func @structurally_irregular_nested_tuples
// CHECK-COUNT-6: llvm.mul
// CHECK-NOT: llvm.mul
func.func @structurally_irregular_nested_tuples(
  %a: tuple<i64, tuple<i64, i64>, i64, i64>,
  %b: tuple<i64, tuple<i64, tuple<i64, i64>>, i64, i64>
) -> tuple<i64, tuple<i64, tuple<i64, i64>>, i64, i64> {
  %res = coord.weak_product %a, %b :
    tuple<i64, tuple<i64, i64>, i64, i64>,
    tuple<i64, tuple<i64, tuple<i64, i64>>, i64, i64>
  -> tuple<i64, tuple<i64, tuple<i64, i64>>, i64, i64>
  return %res : tuple<i64, tuple<i64, tuple<i64, i64>>, i64, i64>
}

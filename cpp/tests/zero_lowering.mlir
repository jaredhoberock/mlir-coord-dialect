// RUN: mlir-opt --pass-pipeline="builtin.module(monomorphize-trait,convert-to-llvm)" %s | FileCheck %s

// ---- i64

// CHECK-LABEL: func @zero_i64
// CHECK: llvm.mlir.constant(0 : i64)
func.func @zero_i64() -> i64 {
  %res = coord.zero : i64
  return %res : i64
}

// ---- empty tuple

// CHECK-LABEL: func @zero_empty
// CHECK: %[[RES:.*]] = llvm.mlir.undef
// CHECK: return %[[RES]] : !llvm.struct<()>
func.func @zero_empty() -> tuple<> {
  %res = coord.zero : tuple<>
  return %res : tuple<>
}

// ---- nested tuple

// CHECK-LABEL: func @zero_nested
// CHECK-3: llvm.mlir.constant(0 : i64)
// CHECK: !llvm.struct<(i64, struct<()>, struct<(i64, i64)>)>
func.func @zero_nested() -> tuple<i64, tuple<>, tuple<i64,i64>> {
  %res = coord.zero : tuple<i64, tuple<>, tuple<i64,i64>>
  return %res : tuple<i64, tuple<>, tuple<i64,i64>>
}

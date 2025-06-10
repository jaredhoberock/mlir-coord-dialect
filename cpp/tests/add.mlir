// RUN: mlir-opt %s | FileCheck %s

// ---- Test 1: i64 operands

// CHECK-LABEL: func @add_scalar
// CHECK: %[[RES:.*]] = coord.add %arg0, %arg1 : i64
// CHECK: return %[[RES]] : i64

func.func @add_scalar(%arg0: i64, %arg1: i64) -> i64 {
  %res = coord.add %arg0, %arg1 : i64
  return %res : i64
}

// ---- Test 2: empty tuple operands

// CHECK-LABEL: func @add_empty
// CHECK: %[[RES:.*]] = coord.add %arg0, %arg1 : tuple<>
// CHECK: return %[[RES]] : tuple<>

func.func @add_empty(%arg0: tuple<>, %arg1: tuple<>) -> tuple<> {
  %res = coord.add %arg0, %arg1 : tuple<>
  return %res : tuple<>
}

// ---- Test 3: generic coord operands

// CHECK-LABEL: func @add_generic
// CHECK: %[[RES:.*]] = coord.add %arg0, %arg1 : !coord.coord
// CHECK: return %[[RES]] : !coord.coord

func.func @add_generic(%arg0: !coord.coord, %arg1: !coord.coord) -> !coord.coord {
  %res = coord.add %arg0, %arg1 : !coord.coord
  return %res : !coord.coord
}

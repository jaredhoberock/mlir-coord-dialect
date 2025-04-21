// RUN: opt %s | FileCheck %s

func.func private @foo(%a: !coord.coord, %b: !coord.coord) -> !coord.coord

// ---- Test 1: mono_call with tuple<>

// CHECK-LABEL: func @call_foo_empty
// CHECK: %[[R:.*]] = coord.mono_call @foo(%arg0, %arg1) : (tuple<>, tuple<>) -> tuple<>
// CHECK: return %[[R]] : tuple<>

func.func @call_foo_empty(%arg0: tuple<>, %arg1: tuple<>) -> tuple<> {
  %r = coord.mono_call @foo(%arg0, %arg1) : (tuple<>, tuple<>) -> tuple<>
  return %r : tuple<>
}

// ---- Test 2: basic mono_call with i64

// CHECK-LABEL: func @call_foo_scalar
// CHECK: %[[R:.*]] = coord.mono_call @foo(%arg0, %arg1) : (i64, i64) -> i64
// CHECK: return %[[R]] : i64

func.func @call_foo_scalar(%arg0: i64, %arg1: i64) -> i64 {
  %r = coord.mono_call @foo(%arg0, %arg1) : (i64, i64) -> i64
  return %r : i64
}

// ---- Test 3: mono_call with tuple<i64, i64> arguments

// CHECK-LABEL: func @call_foo_tuple
// CHECK: %[[R:.*]] = coord.mono_call @foo(%arg0, %arg1) : (tuple<i64, i64>, tuple<i64, i64>) -> tuple<i64, i64>
// CHECK: return %[[R]] : tuple<i64, i64>

func.func @call_foo_tuple(%arg0: tuple<i64, i64>, %arg1: tuple<i64, i64>) -> tuple<i64, i64> {
  %r = coord.mono_call @foo(%arg0, %arg1) : (tuple<i64, i64>, tuple<i64, i64>) -> tuple<i64, i64>
  return %r : tuple<i64, i64>
}

// ---- Test 4: mono_call with nested tuple operands

// CHECK-LABEL: func @call_foo_nested
// CHECK: %[[R:.*]] = coord.mono_call @foo(%arg0, %arg1) : (tuple<i64, tuple<i64>>, tuple<i64, tuple<i64>>) -> tuple<i64, tuple<i64>>
// CHECK: return %[[R]] : tuple<i64, tuple<i64>>

func.func @call_foo_nested(%arg0: tuple<i64, tuple<i64>>, %arg1: tuple<i64, tuple<i64>>) -> tuple<i64, tuple<i64>> {
  %r = coord.mono_call @foo(%arg0, %arg1) : (tuple<i64, tuple<i64>>, tuple<i64, tuple<i64>>) -> tuple<i64, tuple<i64>>
  return %r : tuple<i64, tuple<i64>>
}

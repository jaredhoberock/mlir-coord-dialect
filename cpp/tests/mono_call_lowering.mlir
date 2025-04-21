// RUN: opt --convert-to-llvm %s | FileCheck %s

func.func @foo(%a: !coord.coord, %b: !coord.coord) -> !coord.coord {
  %r = coord.sum %a, %b : !coord.coord
  return %r : !coord.coord
}

// ---- Test 1: mono_call with i64

// CHECK-LABEL: llvm.func @call_foo_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call
// CHECK: llvm.return
func.func @call_foo_i64(%x: i64, %y: i64) -> i64 {
  %r = coord.mono_call @foo(%x, %y) : (i64, i64) -> i64
  return %r : i64
}

// ---- Test 2: mono_call with empty tuple

// CHECK-LABEL: llvm.func @call_foo_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call
// CHECK: llvm.return
func.func @call_foo_empty(%x: tuple<>, %y: tuple<>) -> tuple<> {
  %r = coord.mono_call @foo(%x, %y) : (tuple<>, tuple<>) -> tuple<>
  return %r : tuple<>
}

// ---- Test 3: mono_call with tuple<i64, i64>

// CHECK-LABEL: llvm.func @call_foo_tuple
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call
// CHECK: llvm.return
func.func @call_foo_tuple(%x: tuple<i64,i64>, %y: tuple<i64,i64>) -> tuple<i64,i64> {
  %r = coord.mono_call @foo(%x, %y) : (tuple<i64,i64>, tuple<i64,i64>) -> tuple<i64,i64>
  return %r : tuple<i64,i64>
}

// ---- Test 4: mono_call with nested tuple operands

// CHECK-LABEL: llvm.func @call_foo_nested
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.call
// CHECK: llvm.return
func.func @call_foo_nested(%arg0: tuple<i64, tuple<i64>>, %arg1: tuple<i64, tuple<i64>>) -> tuple<i64, tuple<i64>> {
  %r = coord.mono_call @foo(%arg0, %arg1) : (tuple<i64, tuple<i64>>, tuple<i64, tuple<i64>>) -> tuple<i64, tuple<i64>>
  return %r : tuple<i64, tuple<i64>>
}

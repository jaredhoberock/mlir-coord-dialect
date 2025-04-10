// RUN: opt --convert-to-llvm %s | FileCheck %s

// -----

// CHECK-LABEL: llvm.func @make_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_empty() -> !coord.coord<0> {
  %c = coord.make to !coord.coord<0>
  return %c : !coord.coord<0>
}

// -----

// CHECK-LABEL: llvm.func @make_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_i64(%arg0: i64) -> !coord.coord<2> {
  %c = coord.make %arg0 : i64 to !coord.coord<2>
  return %c : !coord.coord<2>
}

// -----

// CHECK-LABEL: func @make_i64_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_i64_i64(%a: i64, %b: i64) -> !coord.coord<107> {
  %c = coord.make %a, %b : i64, i64 to !coord.coord<107>
  return %c : !coord.coord<107>
}

// -----

// CHECK-LABEL: func @make_nested
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_nested(%a: i64, %b: !coord.coord<107>) -> !coord.coord<6575> {
  %c = coord.make %a, %b : i64, !coord.coord<107> to !coord.coord<6575>
  return %c : !coord.coord<6575>
}

// -----

// CHECK-LABEL: func @make_nested_middle
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_nested_middle(%arg0: i64, %arg1: !coord.coord<107>, %arg2: i64) -> !coord.coord<26299> {
  %c = coord.make %arg0, %arg1, %arg2 : i64, !coord.coord<107>, i64 to !coord.coord<26299>
  return %c : !coord.coord<26299>
}

// -----

// CHECK-LABEL: func @make_deeply_nested
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_deeply_nested(%arg0: !coord.coord<107>, %arg1: !coord.coord<27>) -> !coord.coord<93039> {
  %c = coord.make %arg0, %arg1 : !coord.coord<107>, !coord.coord<27> to !coord.coord<93039>
  return %c : !coord.coord<93039>
}

// -----

// CHECK-LABEL: func @make_mixed_tuple
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_mixed_tuple(
  %arg0: !coord.coord<0>,
  %arg1: !coord.coord<2>,
  %arg2: !coord.coord<27>
) -> !coord.coord<4719> {
  %c = coord.make %arg0, %arg1, %arg2 : !coord.coord<0>, !coord.coord<2>, !coord.coord<27> to !coord.coord<4719>
  return %c : !coord.coord<4719>
}

// -----

// CHECK-LABEL: func @sum_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.add
// CHECK: llvm.return
// CHECK: i1
func.func @sum_empty(%a: !coord.coord<0>, %b: !coord.coord<0>) -> !coord.coord<0> {
  %0 = coord.sum %a, %b : !coord.coord<0>
  return %0 : !coord.coord<0>
}

// -----

// CHECK-LABEL: func @sum_scalar
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.add
// CHECK: llvm.return
// CHECK: vector<1xi64>
func.func @sum_scalar(%a: !coord.coord<2>, %b: !coord.coord<2>) -> !coord.coord<2> {
  %0 = coord.sum %a, %b : !coord.coord<2>
  return %0 : !coord.coord<2>
}

// -----

// CHECK-LABEL: func @sum_pair
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.add
// CHECK: vector<2xi64>
// CHECK: llvm.return
func.func @sum_pair(%a: !coord.coord<107>, %b: !coord.coord<107>) -> !coord.coord<107> {
  %0 = coord.sum %a, %b : !coord.coord<107>
  return %0 : !coord.coord<107>
}

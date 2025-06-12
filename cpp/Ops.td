#ifndef COORD_OPS_TD
#define COORD_OPS_TD

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "Dialect.td"
include "OpInterfaces.td"

def CoordLike : TypeConstraint<
  CPred<"mlir::coord::isCoordLike($_self)">,
  "coordinate-compatible type (generic !coord.coord, !coord.poly, integer, or tuple of CoordLike elements)">;

def AddOp : CoordOp<"add", [
  Pure,
  SameOperandsAndResultType
]> {
  let summary = "Sum two `!coord.coord` values";
  let description = [{
    Sums two coord-like values. Both operands must be the same type.
  }];

  let arguments = (ins CoordLike:$lhs, CoordLike:$rhs);
  let results = (outs CoordLike:$result);

  let assemblyFormat = "$lhs `,` $rhs `:` type($result) attr-dict";
}

def WeakProductOp : CoordOp<"weak_product", [
  Pure,
  WeaklyCongruentOpInterface
]> {
  let summary = "Applies a weak product across two coord-like values.";
  let description = [{
    Applies a weak product across two coord-like values.
    If the operands are concrete types, the `lhs` must be weakly congruent to `rhs`.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($result) attr-dict";

  let hasVerifier = 1;
}

// XXX instead of AllTypesMatch, I think we want to add some interface that says lhs must be congruent to rhs
// XXX we might need to relax the result type from I64
def InnerProductOp : CoordOp<"inner_product", [
  AllTypesMatch<["lhs", "rhs"]>,
  Pure
]> {
  let summary = "Applies an inner product across two coord-like values";
  let description = [{
    Applies an inner product across two coord-like values.
    Both operands must have the same shape. The result is `i64`.
  }];

  let arguments = (ins CoordLike:$lhs, CoordLike:$rhs);
  let results = (outs I64:$result);

  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) attr-dict";
}

def SubOp : CoordOp<"sub", [
  Pure,
  SameOperandsAndResultType
]> {
  let summary = "Subtracts one `!coord.coord` value from another";
  let description = [{
    Subtracts one `!coord.coord` value from another.
    Both operands must have the same shape. The result has the same shape.
  }];

  let arguments = (ins CoordLike:$lhs, CoordLike:$rhs);
  let results = (outs CoordLike:$result);

  let assemblyFormat = "$lhs `,` $rhs `:` type($result) attr-dict";
}

def ZeroOp : CoordOp<"zero", [Pure]> {
  let summary = "Create a coordinate initialized to zero";
  let description = [{
    The `coord.zero` operation creates a coordinate of the specified type
    with all elements initialized to zero.

    Example:
    ```mlir
    %0 = coord.zero : i64
    %1 = coord.zero : tuple<i64, tuple<>, tuple<i64, i64, i64>>
    %2 = coord.zero : !coord.poly<0>
    ```
  }];

  let results = (outs CoordLike:$result);

  let assemblyFormat = "`:` type($result) attr-dict";
}

#endif // COORD_OPS_TD

#ifndef COORD_OPS_TD
#define COORD_OPS_TD

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "Dialect.td"

def CoordLike : TypeConstraint<
  CPred<"mlir::coord::isCoordLike($_self)">,
  "coordinate-compatible type (generic !coord.coord, integer, or tuple of CoordLike elements)">;

def AddOp : CoordOp<"add", [
  Pure,
  SameOperandsAndResultType
]> {
  let summary = "Sum two `!coord.coord` values";
  let description = [{
    Sums two `!coord.coord` values.
    Both operands must have the same shape. The result has the same shape.
  }];

  let arguments = (ins CoordLike:$lhs, CoordLike:$rhs);
  let results = (outs CoordLike:$result);

  let assemblyFormat = "$lhs `,` $rhs `:` type($result) attr-dict";
}

def InnerProductOp : CoordOp<"inner_product", [
  Pure,
  AllTypesMatch<["lhs", "rhs"]>
]> {
  let summary = "Applies an inner product across two `!coord.coord` values";
  let description = [{
    Applies an inner product across two `!coord.coord` values.
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

#endif // COORD_OPS_TD

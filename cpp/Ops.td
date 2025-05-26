#ifndef COORD_OPS_TD
#define COORD_OPS_TD

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "Dialect.td"

def CoordLike : TypeConstraint<
  CPred<"mlir::coord::isCoordLike($_self)">,
  "coordinate-compatible type (generic !coord.coord, i64, or tuple of CoordLike elements)">;

def SumOp : CoordOp<"sum", [
  Pure,
  AllTypesMatch<["lhs", "rhs", "result"]>
]> {
  let summary = "Sum two coord.coord values";
  let description = [{
    Sums two `coord.coord<shape>` values.
    Both operands must have the same shape. The result has the same shape.
  }];

  let arguments = (ins CoordLike:$lhs, CoordLike:$rhs);
  let results = (outs CoordLike:$result);

  let assemblyFormat = "$lhs `,` $rhs `:` type($result) attr-dict";
}

#endif // COORD_OPS_TD

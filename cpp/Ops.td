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

def MakeTupleOp : CoordOp<"make_tuple", [Pure]> {
  let summary = "Construct a coord‑like tuple value";
  let description = [{
    Packs operands into an MLIR `tuple<...>`.  Each operand and every element of
    the result type must satisfy the `CoordLike` constraint.  Arity and per‑
    element type matching are enforced by the custom verifier.
  }];

  // accept any type here; the verifier enforces coord-like structure
  let arguments = (ins Variadic<AnyType>:$elements);

  // result is also any type (must be a tuple of coord-like types)
  let results = (outs AnyType:$result);

  // optional parens around operands, then a single `: <result-type>`
  let assemblyFormat = "attr-dict (`(` $elements^ `:` type($elements) `)`)? `:` type($result)";

  let hasVerifier = 1;
}

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

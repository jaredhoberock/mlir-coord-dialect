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

def MonoCallOp : CoordOp<"mono_call", [
  CallOpInterface, 
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Monomorphizable call to a generic func.func";
  let description = [{
    The `coord.mono_call` operation represents a call to a `func.func`
    whose signature includes at least one generic `!coord.coord` type. The types
    of the operands and results determine the concrete instantiation.

    Example:

    ```mlir
    %0 = coord.mono_call @coordinate_sum(%a, %b)
        : (tuple<i64, i64>, tuple<i64, i64>) -> tuple<i64, i64>
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
               CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
               CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state,
            SymbolRefAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>
  ];

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return getOperands();
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

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

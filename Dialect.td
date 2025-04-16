#ifndef COORD_DIALECT_TD
#define COORD_DIALECT_TD

include "mlir/IR/OpBase.td"

def CoordDialect : Dialect {
  let name = "coord";
  let summary = "A dialect for tensor coordinate types";
  let cppNamespace = "::mlir::coord";

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::vector::VectorDialect"
  ];
}

class CoordOp<string mnemonic, list<Trait> traits = []> :
  Op<CoordDialect, mnemonic, traits>;

#endif // COORD_DIALECT_TD

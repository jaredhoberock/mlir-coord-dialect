#ifndef COORD_DIALECT_TD
#define COORD_DIALECT_TD

include "mlir/IR/OpBase.td"

def CoordDialect : Dialect {
  let name = "coord";
  let summary = "A dialect for tensor coordinate types";
  let cppNamespace = "::mlir::coord";

  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let dependentDialects = [
    "mlir::tuple::TupleDialect"
  ];
}

class CoordOp<string mnemonic, list<Trait> traits = []> :
  Op<CoordDialect, mnemonic, traits>;

#endif // COORD_DIALECT_TD

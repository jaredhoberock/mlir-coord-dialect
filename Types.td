#ifndef COORD_TYPES_TD
#define COORD_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Dialect.td"

def CoordType : TypeDef<CoordDialect, "Coord"> {
  let mnemonic = "coord";
  let summary = "Coordinate type with shape encoding";
  let parameters = (ins
      "int64_t":$shape
  );

  let assemblyFormat = "`<` $shape `>`";
  let genVerifyDecl = 1;
}

#endif // COORD_TYPES_TD

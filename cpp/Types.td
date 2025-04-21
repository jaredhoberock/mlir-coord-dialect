#ifndef COORD_TYPES_TD
#define COORD_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Dialect.td"

def CoordType : TypeDef<CoordDialect, "Coord"> {
  let summary = "Placeholder type representing any type satisfying the Coord trait";
  let description = [{
    The `!coord.coord` type is a placeholder used in generic functions to indicate a value
    of some unknown type that implements the Coord trait. It is used for generic
    operations and function arguments, and must be replaced during monomorphization.

    For example:
      func.func @coordinate_sum(%a: !coord.coord, %b: !coord.coord) -> !coord.coord {
        %sum = coord.sum %a, %b : !coord.coord
        return %sum : !coord.coord
      }

    After monomorphization, this might become:
      func.func @coordinate_sum_i64(%a: i64, %b: i64) -> i64 {
        %sum = arith.addi %a, %b : i64
        return %sum : i64
      }
  }];

  let mnemonic = "coord";
}

#endif // COORD_TYPES_TD

#ifndef COORD_TYPES_TD
#define COORD_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir-trait-dialect/cpp/TypeInterfaces.td"
include "Dialect.td"

def CoordType : TypeDef<CoordDialect, "Coord", [SymbolicTypeUnificationInterface]> {
  let summary = "Placeholder type representing any type satisfying the Coord trait";
  let description = [{
    The `!coord.coord` type is a symbolic type used to provide a generic `trait.impl`s
    for any coord-like type. It provides the method `bool matches(Type ty)` which will
    return `true` for any coord-like type.

    For example:
      trait.trait @Coord {
        func.func private sum(!trait.self, !trait.self) -> !trait.self
      }

      trait.impl @Coord !coord.coord {
        func.func private @sum(%a: !coord.coord, %b: !coord.coord) -> !coord.coord {
          %res = coord.sum %a, %b : !coord.coord
          return $res : !coord.coord
        }
      }

      !C = trait.poly<0, [@Coord]>
      func.func @coordinate_sum(%a: !C, %b: !C) -> !C {
        %res = trait.method.call @Coord::@sum<!C>(%a, %b) : (!trait.self,!trait.self) -> !trait.self to (!C,!C) -> !C
        return %res : !C
      }

    After monomorphization and inlining, this might become:
      func.func @coordinate_sum_i64(%a: i64, %b: i64) -> i64 {
        %sum = arith.addi %a, %b : i64
        return %sum : i64
      }
  }];

  let extraClassDeclaration = [{
    // for SymbolicTypeUnificationInterface
    LogicalResult unifyWith(Type ty, ModuleOp module, llvm::function_ref<InFlightDiagnostic()> emitError) const;
  }];

  let mnemonic = "coord";
}

def PolyType : TypeDef<CoordDialect, "Poly", [SymbolicTypeUnificationInterface]> {
  let mnemonic = "poly";

  let summary = "Polymorphic symbolic type variable representing a coord-like type";
  let description = [{
    The `!coord.poly` type represents a universally quantified type parameter that is
    constrained to be coord-like. It is used to model generic functions for coordinate types.

    The `unique_id` serves as a unique identifier for this type within the module.
  }];

  let extraClassDeclaration = [{
    // for SymbolicTypeUnificationInterface
    LogicalResult unifyWith(Type ty, ModuleOp module, llvm::function_ref<InFlightDiagnostic()> emitError) const;
  }];

  let parameters = (ins "int":$unique_id);

  let assemblyFormat = "`<` $unique_id `>`";
}

#endif // COORD_TYPES_TD

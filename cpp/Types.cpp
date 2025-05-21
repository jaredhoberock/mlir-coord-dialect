#include "Dialect.hpp"
#include "Types.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::coord;

#define GET_TYPEDEF_CLASSES
#include "Types.cpp.inc"

bool CoordType::matches(Type ty, mlir::trait::TraitOp&) const {
  // we assume that the trait doing the query is @Coord
  // isCoordLike depends only on the structure of ty,
  // so we can ignore the trait
  return isCoordLike(ty);
}

void CoordDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Types.cpp.inc"
  >();
}

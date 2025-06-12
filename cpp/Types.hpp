#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <mlir-trait-dialect/cpp/Types.hpp>
#include "Dialect.hpp"

#define GET_TYPEDEF_CLASSES
#include "Types.hpp.inc"

namespace mlir::coord {

inline bool isCoordLike(Type ty) {
  if (isa<CoordType>(ty)) {
    return true;
  }

  if (isa<PolyType>(ty)) {
    return true;
  }

  if (ty.isInteger()) {
    return true;
  }

  if (auto tup = dyn_cast<TupleType>(ty)) {
    return llvm::all_of(tup.getTypes(), isCoordLike);
  }

  return false;
}

} // end mlir::coord

#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "Dialect.hpp.inc"

namespace mlir::coord {

inline bool isCoordLike(Type ty) {
  if (ty.isInteger(64)) {
    return true;
  }

  if (auto tup = dyn_cast<TupleType>(ty)) {
    return llvm::all_of(tup.getTypes(), isCoordLike);
  }

  return false;
}

}

#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <mlir-trait-dialect/cpp/Types.hpp>
#include <optional>
#include "Dialect.hpp"

#define GET_TYPEDEF_CLASSES
#include "Types.hpp.inc"

namespace mlir::coord {

bool isConcreteCoordLike(Type ty);

bool isCoordLike(Type ty);

bool isCongruentTo(Type a, Type b);

bool areCongruent(TypeRange types);

bool isWeaklyCongruentTo(Type a, Type b);

LogicalResult verifyIsWeaklyCongruentTo(std::optional<Location> loc, Type a, Type b);

Type inferInnerProductReturnType(std::optional<Location> loc, Type a, Type b);

Type inferWeakProductReturnType(std::optional<Location> loc, Type a, Type b);

} // end mlir::coord

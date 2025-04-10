#include "Dialect.hpp"
#include "Ops.hpp"
#include "ShapeEncoding.hpp"
#include <mlir/IR/Builders.h>
#include <iostream>

#define GET_OP_CLASSES
#include "Ops.cpp.inc"

using namespace mlir;
using namespace mlir::coord;

/// Recursively validates that a type is a tuple of valid coordinate elements.
/// A valid element is:
/// - an i64 (representing a scalar coordinate)
/// - a coord.coord<shape>
/// - another recursively valid tuple
LogicalResult validateTupleStructure(Type type) {
  auto tupleTy = dyn_cast<TupleType>(type);
  if (!tupleTy)
    return failure(); // Top-level type must be a tuple

  for (Type elem : tupleTy.getTypes()) {
    // Accept i64 scalars
    if (elem.isInteger(64)) {
      continue;
    }

    // Accept nested coord.coord types
    if (isa<CoordType>(elem)) {
      continue;
    }

    // Recursively validate nested tuples
    if (isa<TupleType>(elem)) {
      if (failed(validateTupleStructure(elem))) {
        return failure();
      }

      continue;
    }

    // Reject all other types
    return failure();
  }

  return success();
}

LogicalResult MakeOp::verify() {
  auto coordTy = dyn_cast<CoordType>(getResult().getType());
  if (!coordTy)
    return emitOpError("result must be a coord.coord");

  auto elements = getElements();

  // a single i64 is a special case
  // we don't create a tuple for this case
  if (elements.size() == 1 && elements[0].getType().isInteger(64)) {
    // Single scalar → scalar coordinate
    if (coordTy.getShape() != 0b10)
      return emitOpError() << "expected shape encoding 2 for scalar, but got " << coordTy.getShape();
    return success();
  }

  // Build a TupleType from the operand types
  SmallVector<Type> operandTypes;
  for (Value v : elements)
    operandTypes.push_back(v.getType());
  auto tupleTy = TupleType::get(getContext(), operandTypes);

  // Validate tuple structure
  if (failed(validateTupleStructure(tupleTy)))
    return emitOpError("operand types must be i64 or coord.coord");

  // Compute shape from structure
  int64_t expectedShape = computeShapeOfCoord(tupleTy);
  int64_t actualShape = coordTy.getShape();

  if (expectedShape != actualShape)
    return emitOpError()
      << "expected shape encoding " << expectedShape
      << " from operands, but got " << actualShape;

  return success();
}

#include "Dialect.hpp"
#include "Ops.hpp"
#include "ShapeEncoding.hpp"
#include <mlir/IR/Builders.h>

#define GET_OP_CLASSES
#include "Ops.cpp.inc"

using namespace mlir;
using namespace mlir::coord;

LogicalResult FromScalarOp::verify() {
  auto coordTy = dyn_cast<CoordType>(getResult().getType());
  if (!coordTy) {
    return emitOpError("result must have coord.coord type");
  }

  if (coordTy.getShape() != ScalarShapeEncoding) {
    return emitOpError()
      << "result must have scalar shape encoding (0b10 = "
      << ScalarShapeEncoding << ")";
  }

  return success();
}

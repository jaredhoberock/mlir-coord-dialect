#include "Dialect.hpp"
#include "ShapeEncoding.hpp"
#include "Types.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::coord;

#define GET_TYPEDEF_CLASSES
#include "Types.cpp.inc"

void CoordDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Types.cpp.inc"
  >();
}

LogicalResult CoordType::verify(function_ref<InFlightDiagnostic()> emitError,
                                int64_t shape) {
  if (!isValidShapeEncoding(shape))
    return emitError() << "invalid shape encoding for coord.coord: " << shape;

  return success();
}

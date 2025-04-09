#include "Dialect.hpp"
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
  // XXX TODO port the python code to verify the encoding
  auto isValid = [](int64_t shape) -> bool {
    // Very simple scalar-only check for now
    return shape == 2;
  };

  if (!isValid(shape))
    return emitError() << "invalid shape encoding for coord.coord: " << shape;

  return success();
}

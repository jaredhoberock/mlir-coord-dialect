#include "Dialect.hpp"
#include "Types.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::coord;

#define GET_TYPEDEF_CLASSES
#include "Types.cpp.inc"

LogicalResult CoordType::unifyWith(Type ty, ModuleOp, llvm::function_ref<InFlightDiagnostic()> emitError) const {
  if (!isCoordLike(ty)) {
    return emitError() << "'" << ty << "' is not coord-like";
  }
  return success();
}

void CoordDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Types.cpp.inc"
  >();
}

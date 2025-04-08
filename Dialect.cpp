#include "Dialect.hpp"
#include "Ops.hpp"
#include "Lowering.hpp"
#include "Types.hpp"
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace mlir::coord;

#include "Dialect.cpp.inc"

void CoordDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
  >();

  registerTypes();
}

#include "coord_c.h"
#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/IR/Builders.h>

using namespace mlir;
using namespace mlir::coord;

extern "C" {

void coordRegisterDialect(MlirContext context) {
  unwrap(context)->loadDialect<CoordDialect>();
}

MlirOperation coordSumOpCreate(MlirLocation loc, MlirValue lhs, MlirValue rhs,
                               MlirType resultType) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  auto op = builder.create<SumOp>(unwrap(loc),
                                  unwrap(resultType),
                                  unwrap(lhs), unwrap(rhs));
  return wrap(op.getOperation());
}

MlirType coordCoordTypeGet(MlirContext ctx) {
  return wrap(CoordType::get(unwrap(ctx)));
}

bool coordTypeIsCoord(MlirType type) {
  return isa<CoordType>(unwrap(type));
}

} // end extern "C"

#include "coord_c.h"
#include "Dialect.hpp"
#include "Ops.hpp"
#include "Types.hpp"
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/IR/Builders.h>

using namespace mlir;
using namespace mlir::coord;

extern "C" {

void coordRegisterDialect(MlirContext context) {
  unwrap(context)->loadDialect<CoordDialect>();
}

MlirType coordCoordTypeGet(MlirContext ctx, int64_t shape) {
  return wrap(CoordType::get(unwrap(ctx), shape));
}

bool coordTypeIsCoord(MlirType type) {
  return isa<CoordType>(unwrap(type));
}

int64_t coordCoordTypeGetShape(MlirType type) {
  return cast<CoordType>(unwrap(type)).getShape();
}

MlirOperation coordMakeOpCreate(MlirLocation loc, MlirType resultType,
                                MlirValue* elements, intptr_t nElements) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  SmallVector<Value, 4> elementVals;
  for (intptr_t i = 0; i < nElements; ++i)
    elementVals.push_back(unwrap(elements[i]));

  auto op = builder.create<MakeOp>(unwrap(loc),
                                   cast<CoordType>(unwrap(resultType)),
                                   elementVals);
  return wrap(op.getOperation());
}

MlirOperation coordMakeTupleOpCreate(MlirLocation loc, MlirType resultType,
                                     MlirValue* elements, intptr_t nElements) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  SmallVector<Value, 4> elementVals;
  for (intptr_t i = 0; i < nElements; ++i)
    elementVals.push_back(unwrap(elements[i]));

  auto op = builder.create<MakeTupleOp>(unwrap(loc),
                                   unwrap(resultType),
                                   elementVals);
  return wrap(op.getOperation());
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

} // end extern "C"

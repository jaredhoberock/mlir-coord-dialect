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

MlirOperation coordMonoCallOpCreate(MlirLocation loc, MlirStringRef callee,
                                    MlirValue* arguments, intptr_t nArguments,
                                    MlirType* resultTypes, intptr_t nResults) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  SmallVector<Value> args;
  for (intptr_t i = 0; i < nArguments; ++i)
    args.push_back(unwrap(arguments[i]));

  SmallVector<Type> results;
  for (intptr_t i = 0; i < nResults; ++i)
    results.push_back(unwrap(resultTypes[i]));

  auto calleeAttr = SymbolRefAttr::get(ctx, StringRef(callee.data, callee.length));

  auto op = builder.create<MonoCallOp>(unwrap(loc), calleeAttr, results, args);

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

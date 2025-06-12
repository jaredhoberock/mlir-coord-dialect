#include "Dialect.hpp"
#include "Ops.hpp"
#include "Types.hpp"
#include <mlir/IR/Builders.h>
#include <iostream>

#define GET_OP_CLASSES
#include "Ops.cpp.inc"

namespace mlir::coord {

LogicalResult WeakProductOp::verify() {
  Type lhsTy = getLhs().getType();
  Type rhsTy = getRhs().getType();
  Type resultTy = getResult().getType();
  
  if (!trait::containsSymbolicType(lhsTy) and !trait::containsSymbolicType(rhsTy)) {
    // in when operands are concrete types, verify the expected result type

    Type expectedTy = inferWeakProductReturnType(getLoc(), lhsTy, rhsTy);
    if (expectedTy != resultTy)
      return emitError() << "expected result type " << expectedTy << ", got "
                         << resultTy;

    return success();
  }

  // when operands are symbolic types, the expected result type must be !coord.poly
  if (!isa<PolyType>(resultTy))
    return emitError() << "expected 'coord.poly' for type of result, got "
                       << resultTy;

  return success();
}

}

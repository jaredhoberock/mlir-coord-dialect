#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/IR/Builders.h>
#include <iostream>

#define GET_OP_CLASSES
#include "Ops.cpp.inc"

using namespace mlir;
using namespace mlir::coord;

LogicalResult MakeTupleOp::verify() {
  auto tupleTy = dyn_cast<TupleType>(getResult().getType());
  if (!tupleTy)
    return emitOpError("result must be a tuple type");
  if (tupleTy.size() != getNumOperands())
    return emitOpError("operand/result arity mismatch");

  // Check each element type and operand type match and are coord‑like.
  for (unsigned i = 0, e = tupleTy.size(); i != e; ++i) {
    Type eltTy = tupleTy.getType(i);
    if (eltTy != getOperand(i).getType())
      return emitOpError() << "operand " << i << " type mismatch";
    if (!isCoordLike(eltTy))
      return emitOpError() << "element type " << eltTy
                           << " is not coord‑like (i64 or nested tuple)";
  }
  return success();
}

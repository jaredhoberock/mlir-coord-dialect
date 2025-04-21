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

LogicalResult MonoCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // 1. Verify callee attribute exists.
  auto calleeAttr = getCalleeAttr();
  if (!calleeAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");

  // 2. Look up the callee func.func
  auto func = symbolTable.lookupNearestSymbolFrom<mlir::func::FuncOp>(*this, calleeAttr);
  if (!func)
    return emitOpError() << "'" << calleeAttr.getValue()
                         << "' does not refer to a valid func.func";

  auto funcType = func.getFunctionType();
  auto operands = getOperands();
  auto results = getResultTypes();

  // 3. Check operand count
  if (operands.size() != funcType.getNumInputs())
    return emitOpError("operand count does not match callee function type");

  // 4. Check operand types
  for (auto [i, expected] : llvm::enumerate(funcType.getInputs())) {
    auto actual = operands[i].getType();
  
    if (isa<coord::CoordType>(expected)) {
      // The actual argument must *not* be CoordType, but must be CoordLike.
      if (isa<coord::CoordType>(actual)) {
        return emitOpError("operand ")
               << i << " must be a concrete CoordLike type, not !coord.coord";
      }
      if (!coord::isCoordLike(actual)) {
        return emitOpError("operand ")
               << i << " must be CoordLike, but got " << actual;
      }
      continue;
    }
  
    if (actual != expected) {
      return emitOpError("operand type mismatch at index ")
             << i << ": expected " << expected << ", got " << actual;
    }
  }

  // 5. Check result count
  if (results.size() != funcType.getNumResults())
    return emitOpError("result count does not match callee function type");

  // 6. Check result types
  for (auto [i, expected] : llvm::enumerate(funcType.getResults())) {
    auto actual = results[i];
  
    if (isa<coord::CoordType>(expected)) {
      // Result must *not* be generic anymore
      if (isa<coord::CoordType>(actual)) {
        return emitOpError("result ")
               << i << " must be a concrete CoordLike type, not !coord.coord";
      }
      if (!coord::isCoordLike(actual)) {
        return emitOpError("result ")
               << i << " must be CoordLike, but got " << actual;
      }
      continue;
    }
  
    if (actual != expected) {
      return emitOpError("result type mismatch at index ")
             << i << ": expected " << expected << ", got " << actual;
    }
  }

  return success();
}

//mlir::LogicalResult MonoCallOp::verify() {
//  auto calleeAttr = getCalleeAttr();
//  auto callee = getOperation()->getParentOfType<mlir::ModuleOp>()
//                    .lookupSymbol<mlir::func::FuncOp>(calleeAttr.getAttr());
//
//  if (!callee) {
//    return emitOpError("callee symbol '")
//           << calleeAttr.getAttr() << "' does not refer to a valid func.func";
//  }
//
//  auto funcType = callee.getFunctionType();
//  auto operands = getOperands();
//
//  if (funcType.getNumInputs() != operands.size()) {
//    return emitOpError("expected ")
//           << funcType.getNumInputs() << " operands but got " << operands.size();
//  }
//
//  auto results = getResultTypes();
//  if (funcType.getNumResults() != results.size()) {
//    return emitOpError("expected ")
//           << funcType.getNumResults() << " results but got " << results.size();
//  }
//
//  bool hasCoordPlaceholder = false;
//
//  // Check operand types.
//  for (auto [i, expected] : llvm::enumerate(funcType.getInputs())) {
//    auto actual = operands[i].getType();
//
//    if (isa<CoordType>(expected)) {
//      hasCoordPlaceholder = true;
//
//      if (isa<CoordType>(actual)) {
//        return emitOpError("operand at index ")
//               << i << " is also generic (!coord.coord); expected concrete coordinate-like type";
//      }
//
//      if (!isCoordLike(actual)) {
//        return emitOpError("operand at index ")
//               << i << " is not a coordinate-like type: got " << actual;
//      }
//
//      continue;
//    }
//
//    if (actual != expected) {
//      return emitOpError("operand type mismatch at index ")
//             << i << ": expected " << expected << ", got " << actual;
//    }
//  }
//
//  // Check result types.
//  for (auto [i, expected] : llvm::enumerate(funcType.getResults())) {
//    auto actual = results[i];
//
//    if (isa<CoordType>(expected)) {
//      hasCoordPlaceholder = true;
//
//      if (isa<CoordType>(actual)) {
//        return emitOpError("result at index ")
//               << i << " is also generic (!coord.coord); expected concrete coordinate-like type";
//      }
//
//      if (!isCoordLike(actual)) {
//        return emitOpError("result at index ")
//               << i << " is not a coordinate-like type: got " << actual;
//      }
//
//      continue;
//    }
//
//    if (actual != expected) {
//      return emitOpError("result type mismatch at index ")
//             << i << ": expected " << expected << ", got " << actual;
//    }
//  }
//
//  if (!hasCoordPlaceholder) {
//    return emitOpError("must call a generic function (at least one !coord.coord parameter or result)");
//  }
//
//  return mlir::success();
//}

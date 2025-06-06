#include "ConvertToTrait.hpp"
#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir-tuple-dialect/cpp/Ops.hpp>

using namespace mlir;

namespace mlir::coord {

struct SumOpLowering : public OpRewritePattern<SumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SumOp op, PatternRewriter &rewriter) const override {

    MLIRContext* ctx = rewriter.getContext();
    auto loc = op.getLoc();

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type operandTy = lhs.getType();

    // terminal case 1: operands are integers, lower to arith.addi
    if (operandTy.isInteger()) {
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, lhs, rhs);
      return success();
    }

    // recursive case: operands are tuples, lower to tuple.map + coord.sum
    if (auto tupleTy = dyn_cast<TupleType>(operandTy)) {
      // create tuple.map
      auto mapOp = rewriter.create<tuple::MapOp>(
        loc,
        tupleTy,
        ValueRange{lhs, rhs}
      );

      // create the region with recursive coord.sum
      Block* block = rewriter.createBlock(&mapOp.getBody());
      rewriter.setInsertionPointToStart(block);

      // block arguments are !coord.coord
      CoordType coordTy = CoordType::get(ctx);
      block->addArguments({coordTy, coordTy}, {loc, loc});

      // %res = coord.sum %lhs_elem, %rhs_elem : !coord.coord
      Value sum = rewriter.create<SumOp>(
        loc,
        coordTy,
        block->getArgument(0),  // lhs element
        block->getArgument(1)); // rhs element

      // yield %res : !coord.coord
      rewriter.create<tuple::YieldOp>(loc, sum);

      assert(succeeded(mapOp.verify()));

      // replace the original operation
      rewriter.replaceOp(op, mapOp.getResult());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsuppored operand type for coord.sum");
  }
};

void populateCoordToTraitConversionPatterns(RewritePatternSet& patterns) {
  // lower ops
  patterns.add<
    SumOpLowering
  >(patterns.getContext());
}

}

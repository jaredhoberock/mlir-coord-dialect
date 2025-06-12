#include "ConvertToTrait.hpp"
#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir-tuple-dialect/cpp/Ops.hpp>

using namespace mlir;

namespace mlir::coord {

struct AddOpLowering : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override {

    MLIRContext* ctx = rewriter.getContext();
    auto loc = op.getLoc();

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type operandTy = lhs.getType();

    // terminal case: operands are integers, lower to arith.addi
    if (operandTy.isInteger()) {
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, lhs, rhs);
      return success();
    }

    // recursive case: operands are tuples, lower to tuple.map + coord.add
    if (auto tupleTy = dyn_cast<TupleType>(operandTy)) {
      // create tuple.map
      auto mapOp = rewriter.create<tuple::MapOp>(
        loc,
        tupleTy,
        ValueRange{lhs, rhs}
      );

      // create the region with recursive coord.add
      Block* block = rewriter.createBlock(&mapOp.getBody());
      rewriter.setInsertionPointToStart(block);

      // block arguments are !coord.coord
      CoordType coordTy = CoordType::get(ctx);
      block->addArguments({coordTy, coordTy}, {loc, loc});

      // %res = coord.add %lhs_elem, %rhs_elem : !coord.coord
      Value res = rewriter.create<AddOp>(
        loc,
        coordTy,
        block->getArgument(0),  // lhs element
        block->getArgument(1)); // rhs element

      // yield %res : !coord.coord
      rewriter.create<tuple::YieldOp>(loc, res);

      assert(succeeded(mapOp.verify()));

      // replace the original operation
      rewriter.replaceOp(op, mapOp.getResult());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsuppored operand type for coord.add");
  }
};

struct InnerProductOpLowering : public OpRewritePattern<InnerProductOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InnerProductOp op, PatternRewriter &rewriter) const override {
    MLIRContext* ctx = rewriter.getContext();
    auto loc = op.getLoc();

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type operandTy = lhs.getType();

    // terminal case: operands are integers, lower to arith.muli
    if (operandTy.isInteger()) {
      rewriter.replaceOpWithNewOp<arith::MulIOp>(op, lhs, rhs);
      return success();
    }

    // recursive case: operands are tuples, lower to tuple.foldl + coord.inner_product
    if (auto tupleTy = dyn_cast<TupleType>(operandTy)) {
      // %init = arith.constant 0 : i64
      // %res = tuple.fold %init, %lhs, %rhs : i64, tuple<..>, tuple<..> -> i64 {
      // ^bb0(%acc: i64, %a: !coord.coord, %b: !coord.coord):
      //   %prod = coord.inner_product %a, %b : !coord.coord
      //   %res = arith.addi %acc, %prod : i64
      //   yield %res: i64
      // }

      auto i64Ty = rewriter.getIntegerType(64);

      // %init = arith.constant 0 : i64
      Value init = rewriter.create<arith::ConstantOp>(
        loc,
        rewriter.getIntegerAttr(i64Ty, 0)
      );

      // create tuple.foldl
      auto foldlOp = rewriter.create<tuple::FoldlOp>(
        loc,
        i64Ty,
        ValueRange{init, lhs, rhs}
      );

      // create the region with recursive coord.inner_product
      Block* block = rewriter.createBlock(&foldlOp.getBody());
      rewriter.setInsertionPointToStart(block);

      // block arguments are i64, !coord.coord, !coord.coord
      CoordType coordTy = CoordType::get(ctx);
      block->addArguments({i64Ty, coordTy, coordTy}, {loc, loc, loc});

      // %prod = coord.inner_product %lhs_elem, %rhs_elem : i64
      Value prod = rewriter.create<InnerProductOp>(
        loc,
        block->getArgument(1),  // lhs element
        block->getArgument(2)); // rhs element

      // %res = arith.addi %acc, %prod : i64
      Value res = rewriter.create<arith::AddIOp>(
        loc,
        block->getArgument(0), // acc
        prod);

      // yield %res : i64
      rewriter.create<tuple::YieldOp>(loc, res);

      assert(succeeded(foldlOp.verify()));

      // replace the original operation
      rewriter.replaceOp(op, foldlOp.getResult());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported operand type for coord.inner_product");
  }
};

struct SubOpLowering : public OpRewritePattern<SubOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SubOp op, PatternRewriter &rewriter) const override {

    MLIRContext* ctx = rewriter.getContext();
    auto loc = op.getLoc();

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type operandTy = lhs.getType();

    // terminal case: operands are integers, lower to arith.subi
    if (operandTy.isInteger()) {
      rewriter.replaceOpWithNewOp<arith::SubIOp>(op, lhs, rhs);
      return success();
    }

    // recursive case: operands are tuples, lower to tuple.map + coord.add
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

      // %res = coord.sub %lhs_elem, %rhs_elem : !coord.coord
      Value res = rewriter.create<SubOp>(
        loc,
        coordTy,
        block->getArgument(0),  // lhs element
        block->getArgument(1)); // rhs element

      // yield %res : !coord.coord
      rewriter.create<tuple::YieldOp>(loc, res);

      assert(succeeded(mapOp.verify()));

      // replace the original operation
      rewriter.replaceOp(op, mapOp.getResult());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsuppored operand type for coord.sub");
  }
};

struct ZeroOpLowering : public OpRewritePattern<ZeroOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ZeroOp op, PatternRewriter &rewriter) const override {
    Type resultTy = op.getResult().getType();

    if (trait::containsSymbolicType(resultTy))
      return rewriter.notifyMatchFailure(op, "result type is still symbolic");

    if (resultTy.isInteger()) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op,
        rewriter.getIntegerAttr(resultTy, 0)
      );
      return success();
    }

    TupleType tupleTy = dyn_cast<TupleType>(resultTy);
    if (tupleTy) {
      // emit a coord.zero op for each tuple element
      SmallVector<Value> elements;
      for (Type ty : tupleTy.getTypes()) {
        Value element = rewriter.create<ZeroOp>(
          op.getLoc(),
          ty
        );
        elements.push_back(element);
      }

      // replace the original op with a tuple.make
      rewriter.replaceOpWithNewOp<tuple::MakeOp>(
        op,
        elements
      );

      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported result type");
  }
};

void populateCoordToTraitConversionPatterns(RewritePatternSet& patterns) {
  // lower ops
  patterns.add<
    AddOpLowering,
    InnerProductOpLowering,
    SubOpLowering,
    ZeroOpLowering
  >(patterns.getContext());
}

}

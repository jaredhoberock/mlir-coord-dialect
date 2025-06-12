#include "ConvertToTrait.hpp"
#include "Dialect.hpp"
#include "Ops.hpp"
#include "Types.hpp"
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

    return rewriter.notifyMatchFailure(op, "unsupported operand types for coord.inner_product");
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

    if (auto tupleTy = dyn_cast<TupleType>(resultTy)) {
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

struct WeakProductOpLowering : public OpRewritePattern<WeakProductOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WeakProductOp op, PatternRewriter &rewriter) const override {
    MLIRContext* ctx = rewriter.getContext();
    auto loc = op.getLoc();

    Type lhsTy = op.getLhs().getType();
    Type rhsTy = op.getRhs().getType();

    // if operand types are symbolic, this operation isn't ready to be rewritten
    if (trait::containsSymbolicType(lhsTy) or trait::containsSymbolicType(rhsTy))
      return rewriter.notifyMatchFailure(op, "operand types are still symbolic");

    // if lhs & rhs are congruent:
    // coord.weak_product -> coord.inner_product
    if (isCongruentTo(lhsTy, rhsTy)) {
      rewriter.replaceOpWithNewOp<InnerProductOp>(
        op,
        op.getLhs(),
        op.getRhs()
      );

      return success();
    }

    // if lhsTy is an integer then rhsTy must be a tuple:
    // coord.weak_product -> tuple.map + coord.weak_product
    if (lhsTy.isInteger()) {
      // %res = tuple.map %rhs : !Rhs -> !R {
      // ^bb0(%e: !coord.poly<0>):
      //   %res = coord.weak_product %lhs, %e : !Lhs, !coord.poly<0> -> !coord.poly<1>
      //   yield %res : !coord.poly<1>
      // }
      auto mapOp = rewriter.create<tuple::MapOp>(
        loc,
        op.getResult().getType(),
        op.getRhs());

      // create the region with recursive coord.weak_product
      Block* block = rewriter.createBlock(&mapOp.getBody());
      rewriter.setInsertionPointToStart(block);

      // block argument is !coord.poly<0>
      // XXX TODO do we need to use a fresh poly type or not?
      block->addArguments({PolyType::get(ctx, 0)}, {loc});

      // %res = coord.weak_product %lhs, %e : !Lhs, !coord.poly<0> -> !coord.poly<1>
      Value res = rewriter.create<WeakProductOp>(
        loc,
        PolyType::get(ctx, 1),    // !coord.poly<1>
        op.getLhs(),              // %lhs : i64
        block->getArgument(0));   // %e : !coord.poly<0>

      // yield %res
      rewriter.create<tuple::YieldOp>(loc, res);

      // replace the original operation
      rewriter.replaceOp(op, mapOp.getResult());
      return success();
    }

    // otherwise, lhsTy and rhsTy must be two equal-rank tuples
    TupleType lhsTupleTy = dyn_cast<TupleType>(lhsTy);
    TupleType rhsTupleTy = dyn_cast<TupleType>(rhsTy);
    if (!lhsTupleTy or !rhsTupleTy)
      return rewriter.notifyMatchFailure(op, "expected tuple type"); 

    // coord.weak_product -> tuple.map + coord.weak_product
    //
    // !P = tuple<...>
    // %products = tuple.map %lhs, %rhs : !Lhs, !Rhs -> !P {
    // ^bb0(%a: !coord.poly<0>, %b: !coord.poly<1>):
    //   %res = coord.weak_product %a, %b : !coord.poly<0>, !coord.poly<1> -> !coord.poly<2>
    //   yield %res : !coord.poly<2>
    // }

    // infer the result type of this tuple.map
    SmallVector<Type> productResultTypes;
    for (auto [a, b] : llvm::zip(lhsTupleTy.getTypes(), rhsTupleTy.getTypes())) {
      Type ty = inferWeakProductReturnType(std::nullopt, a, b);
      productResultTypes.push_back(ty);
    }

    TupleType productsTy = TupleType::get(ctx, productResultTypes);

    // %products = tuple.map %lhs, %rhs : !Lhs, !Rhs -> !R { ... }
    auto mapOp = rewriter.create<tuple::MapOp>(
      loc,
      productsTy,
      ValueRange{op.getLhs(), op.getRhs()}
    );

    // create the region with recursive coord.weak_product
    {
      PatternRewriter::InsertionGuard guard(rewriter);

      Block* block = rewriter.createBlock(&mapOp.getBody());
      rewriter.setInsertionPointToStart(block);

      // block arguments are !coord.poly<0> and !coord.poly<1>
      // XXX TODO do we need to use fresh poly types?
      block->addArguments({PolyType::get(ctx, 0), PolyType::get(ctx, 1)}, {loc, loc});

      // %res = coord.weak_product %a, %b : !coord.poly<0>, !coord.poly<1> -> !coord.poly<2>
      Value res = rewriter.create<WeakProductOp>(
        loc,
        PolyType::get(ctx, 2), // !coord.poly<2>
        block->getArgument(0),
        block->getArgument(1));

      // yield %res
      rewriter.create<tuple::YieldOp>(loc, res);
    }

    Value products = mapOp.getResult();

    // when all elements of the products tuple are not congruent,
    // we return products directly
    if (!areCongruent(productsTy.getTypes())) {
      rewriter.replaceOp(op, products);
      return success();
    }

    // otherwise, we return the sum of the products
    
    // %init = coord.zero : !T
    // %res = tuple.foldl %init, %products : !T, !P -> !T {
    // ^bb0(%acc: !T, %e: !T):
    //   %res = coord.add %acc, %e : !T
    //   yield %res : !T
    // }

    Type resultTy = productsTy.getType(0);

    // %init = coord.zero : !T
    auto init = rewriter.create<ZeroOp>(loc, resultTy);

    // %res = tuple.foldl %init, %products : !T, !P -> !T { ... }
    auto foldlOp = rewriter.create<tuple::FoldlOp>(
      loc,
      resultTy,
      ValueRange{init, products}
    );

    // create the fold body
    {
      PatternRewriter::InsertionGuard guard(rewriter);

      Block* block = rewriter.createBlock(&foldlOp.getBody());
      rewriter.setInsertionPointToStart(block);

      // block arguments are resultTy
      block->addArguments({resultTy, resultTy}, {loc, loc});

      // %res = coord.add %acc, %e : !T
      Value res = rewriter.create<AddOp>(
        loc,
        block->getArgument(0), // %acc
        block->getArgument(1)  // %e
      );

      // yield %res : !T
      rewriter.create<tuple::YieldOp>(loc, res);
    }

    // replace the original operation
    rewriter.replaceOp(op, foldlOp.getResult());
    return success();
  }
};

void populateCoordToTraitConversionPatterns(RewritePatternSet& patterns) {
  // lower ops
  patterns.add<
    AddOpLowering,
    InnerProductOpLowering,
    SubOpLowering,
    WeakProductOpLowering,
    ZeroOpLowering
  >(patterns.getContext());
}

}

#include "Dialect.hpp"
#include "Lowering.hpp"
#include "Monomorphization.hpp"
#include "Ops.hpp"
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;

namespace mlir::coord {

/// Recursively count the number of scalar i64 leaves in a TupleType.
static int64_t countI64Leaves(TupleType tup) {
  int64_t n = 0;
  for (Type t : tup.getTypes()) {
    if (t.isInteger(64))
      ++n;
    else if (auto sub = dyn_cast<TupleType>(t))
      n += countI64Leaves(sub);
  }
  return n;
}

struct MakeTupleOpLowering : OpConversionPattern<MakeTupleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MakeTupleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // count the number of i64 leaves
    int64_t N = countI64Leaves(cast<TupleType>(op.getType()));

    // special case: tuple<> lowers to i1
    if (N == 0) {
      Type i1Ty = rewriter.getIntegerType(1);
      Value zero = rewriter.create<arith::ConstantOp>(loc, i1Ty, rewriter.getIntegerAttr(i1Ty, 0));
      rewriter.replaceOp(op, zero);
      return success();
    }

    // create result vector type
    auto i64Ty = rewriter.getIntegerType(64);
    auto resultVectorTy = VectorType::get({N}, i64Ty);

    // Begin from a single zero value across the entire vector
    auto zero = rewriter.create<arith::ConstantOp>(loc, i64Ty, rewriter.getIntegerAttr(i64Ty, 0));
    Value result = rewriter.create<vector::SplatOp>(loc, resultVectorTy, zero);

    // Insert operands into the result vector
    int64_t offset = 0;
    for (Value operand : adaptor.getOperands()) {
      // check the type of the lowered operand
      Type operandTy = operand.getType();

      if (!operandTy.isInteger(64) && !operandTy.isInteger(1) &&
          !(isa<VectorType>(operandTy) && cast<VectorType>(operandTy).getElementType().isInteger(64))) {
        return rewriter.notifyMatchFailure(op, "operand must be i1, i64, or vector<Nxi64>");
      }

      // skip i1 (empty tuple) operands, they contribute no elements
      if (operandTy.isInteger(1)) {
        continue;
      }

      // if operand is a vector, use vector.insert_strided_slice
      if (auto vecTy = dyn_cast<VectorType>(operandTy)) {
        int64_t numElements = vecTy.getNumElements();
        SmallVector<int64_t> offsets = {offset};
        SmallVector<int64_t> strides = {1};

        result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, 
          operand, 
          result, 
          offsets, 
          strides
        );

        offset += numElements;
      } else {
        // For scalar i64, use vector.insert
        result = rewriter.create<vector::InsertOp>(loc, operand, result, offset);
        offset += 1;
      }
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};


struct SumOpLowering : public OpConversionPattern<SumOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SumOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // The operands should already by lowered to i0 or vector<Nxi64>
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    if (lhs.getType() != rhs.getType())
      return rewriter.notifyMatchFailure(op, "operand types must match");

    if (!isa<VectorType>(lhs.getType()) && !lhs.getType().isInteger(1) && !lhs.getType().isInteger(64)) {
      return rewriter.notifyMatchFailure(op, "operands must be i1, i64, or vector<Nxi64>");
    }

    Value result = rewriter.create<arith::AddIOp>(loc, lhs, rhs);
    rewriter.replaceOp(op, result);

    return success();
  }
};


struct MonomorphizeModule : OpConversionPattern<ModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ModuleOp module, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Collect all mono_call ops in the module
    SmallVector<MonoCallOp> calls;
    module.walk([&](MonoCallOp call) {
      calls.push_back(call);
    });

    // collect polymorphs and their monomorphs instantiated by `coord.mono_call` operations
    std::set<func::FuncOp> polymorphs;
    DenseMap<std::pair<func::FuncOp, Type>, func::FuncOp> monomorphs;

    // replace each coord.mono_call with func.call to an instantiated monomorph
    for (MonoCallOp call : calls) {
      auto calleeAttr = call.getCalleeAttr();
      auto polymorph = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, calleeAttr);

      if (!polymorph)
        return call.emitOpError("could not find polymorphic callee");

      polymorphs.insert(polymorph);

      // XXX TODO generalize this
      Type concreteType = call.getOperandTypes().front();

      auto key = std::make_pair(polymorph, concreteType);
      func::FuncOp monomorph;

      if (auto it = monomorphs.find(key); it != monomorphs.end()) {
        monomorph = it->second;
      } else {
        func::FuncOp orphanMonomorph = monomorphize(polymorph, concreteType);
        if (!orphanMonomorph)
          return call.emitOpError("monomorphization failed");

        // clone the orphan monomorph using the rewriter to ensure it and its body
        // operations become visible to the lowering process
        rewriter.setInsertionPoint(polymorph);
        monomorphs[key] = monomorph = cast<func::FuncOp>(rewriter.clone(*orphanMonomorph));

        // we no longer need the orphanMonomorph
        orphanMonomorph.erase();
      }

      // Replace MonoCallOp with func.call to monomorph
      rewriter.setInsertionPoint(call);
      auto newCall = rewriter.create<func::CallOp>(
          call.getLoc(), monomorph.getSymNameAttr(),
          monomorph.getFunctionType().getResults(),
          call.getOperands());

      rewriter.replaceOp(call, newCall);
    }

    // after all coord.mono_calls are replaced, it should be safe to erase polymorphs
    // XXX TODO in general, there can be other users of the polymorphs
    //          somehow we need to monomorphize other possible uses
    for (func::FuncOp polymorph : polymorphs) {
      rewriter.eraseOp(polymorph);
    }

    return success();
  }
};


void populateCoordToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
  // add a type conversion for TupleTypes that are CoordLike
  typeConverter.addConversion([&](TupleType tupleTy) -> std::optional<Type> {
    // convert only TupleTypes that are coord-like
    if (!isCoordLike(tupleTy)) {
      // let other converters try
      return std::nullopt;
    }

    auto ctx = tupleTy.getContext();
    int64_t n = countI64Leaves(tupleTy);

    // 0-element vectors are illegal, so lower empty tuples to i1
    if (n == 0) return IntegerType::get(ctx, 1);

    // otherwise, use vector<nxi64>
    return VectorType::get({n}, IntegerType::get(ctx, 64));
  });

  // lower ops
  patterns.add<
    MakeTupleOpLowering,
    MonomorphizeModule,
    SumOpLowering
  >(typeConverter, patterns.getContext());

  vector::populateVectorInsertExtractStridedSliceTransforms(patterns);

  populateVectorToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
}

}

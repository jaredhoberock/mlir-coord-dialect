#include "Dialect.hpp"
#include "Lowering.hpp"
#include "Ops.hpp"
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h>

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
    SumOpLowering
  >(typeConverter, patterns.getContext());

  vector::populateVectorInsertExtractStridedSliceTransforms(patterns);

  populateVectorToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
}

}

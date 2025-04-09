#include "Dialect.hpp"
#include "Lowering.hpp"
#include "Ops.hpp"
#include "ShapeEncoding.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;

namespace mlir::coord {

struct FromScalarOpLowering : public OpConversionPattern<FromScalarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(FromScalarOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto coordTy = dyn_cast<CoordType>(op.getResult().getType());
    if (!coordTy || coordTy.getShape() != coord::ScalarShapeEncoding)
      return rewriter.notifyMatchFailure(op, "coord.from_scalar requires scalar shape encoding");

    auto loc = op.getLoc();
    auto vecTy = VectorType::get({1}, rewriter.getI64Type());

    Value result = rewriter.create<vector::FromElementsOp>(loc, vecTy, adaptor.getValue());

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct SumOpLowering : public OpConversionPattern<SumOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SumOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // The operands should already by lowered to vector<i64>
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    if (lhs.getType() != rhs.getType())
      return rewriter.notifyMatchFailure(op, "operand types must match");

    auto vecTy = dyn_cast<VectorType>(lhs.getType());
    if (!vecTy)
      return rewriter.notifyMatchFailure(op, "expected vector type");

    Value result = rewriter.create<arith::AddIOp>(loc, vecTy, lhs, rhs);
    rewriter.replaceOp(op, result);
    return success();
  }
};

void populateCoordToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
  // add a type conversion for CoordType
  typeConverter.addConversion([&](CoordType coordTy) -> Type {
    int64_t shape = coordTy.getShape();
    int64_t n = coord::getNumIntegersFromShape(shape);
    return VectorType::get({n}, IntegerType::get(coordTy.getContext(), 64));
  });

  // lower ops
  patterns.add<
    FromScalarOpLowering,
    SumOpLowering
  >(typeConverter, patterns.getContext());

  populateVectorToLLVMConversionPatterns(typeConverter, patterns);
}

}

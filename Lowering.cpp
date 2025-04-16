#include "Dialect.hpp"
#include "Lowering.hpp"
#include "Ops.hpp"
#include "ShapeEncoding.hpp"
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;

namespace mlir::coord {

// Lowering strategy for `coord.make`:
//
// Each `coord.make` op constructs a coordinate value from a variadic list of
// operands, each of which is either an `i64` scalar or another `coord.coord`.
// The result is a flattened vector of `i64` values.
//
// We lower this to a single `vector<Ni64>` where `N` is the total number of
// integer elements in the structure.
//
// Lowering steps:
//   1. Determine the total number of integer leaves (N) from the shape encoding.
//   2. Allocate an initial vector<Ni64> using `vector.splat` of zero.
//   3. For each operand:
//       a. If it's an `i64`, insert it directly at the current offset.
//       b. If it's a `coord.coord`, treat it as a flattened `vector<Mxi64>`
//          and insert it using `vector.insert`.
//   4. The final result is the fully assembled vector representing the coord.
//
// Example:
//   coord.make %a, %b : i64, !coord.coord<2> to !coord.coord<43>
//
//   would lower to something like:
//
//     %zero = arith.constant 0 : i64
//     %vinit = vector.splat %zero : vector<3xi64>
//     %v0 = vector.insert %a, %vinit[0] : i64 into vector<3xi64>
//     %v1 = vector.insert %b, %v0[1]     : vector<2xi64> into vector<3xi64>
//
//   where `%b` is a coord.coord<2> already represented as vector<2xi64>.
struct MakeOpLowering : public OpConversionPattern<MakeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(MakeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    auto resultCoordTy = dyn_cast<CoordType>(op.getType());
    if (!resultCoordTy)
      return rewriter.notifyMatchFailure(op, "result type must be coord");

    // Compute total number of i64 elements
    int64_t totalElements = coord::getNumIntegersFromShape(resultCoordTy.getShape());

    // Special case: coord<0> lowers to i1
    if (totalElements == 0) {
      Type i1Ty = rewriter.getIntegerType(1);
      Value zero = rewriter.create<arith::ConstantOp>(loc, i1Ty, rewriter.getIntegerAttr(i1Ty, 0));
      rewriter.replaceOp(op, zero);
      return success();
    }

    // Create result vector type
    auto i64Ty = rewriter.getIntegerType(64);
    auto resultVectorTy = VectorType::get({totalElements}, i64Ty);

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

      // skip i1 (empty) operands, they contribute no elements
      if (operandTy.isInteger(1)) {
        continue;
      }

      // If operand is a vector, use vector.insert_strided_slice
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



void populateCoordToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
  // add a type conversion for CoordType
  typeConverter.addConversion([&](CoordType coordTy) -> Type {
    int64_t shape = coordTy.getShape();
    int64_t n = coord::getNumIntegersFromShape(shape);
    auto ctx = coordTy.getContext();

    // 0-element vectors are illegal, so lower empty coordinates to i1
    if (n == 0) return IntegerType::get(ctx, 1);

    // otherwise, use vector<nxi64>
    return VectorType::get({n}, IntegerType::get(ctx, 64));
  });

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
    MakeOpLowering,
    MakeTupleOpLowering,
    SumOpLowering
  >(typeConverter, patterns.getContext());

  vector::populateVectorInsertExtractStridedSliceTransforms(patterns);

  populateVectorToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
}

}

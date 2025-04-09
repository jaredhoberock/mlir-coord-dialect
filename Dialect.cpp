#include "Dialect.hpp"
#include "Ops.hpp"
#include "Lowering.hpp"
#include "Types.hpp"
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace mlir::coord;

#include "Dialect.cpp.inc"

struct ConvertToLLVMInterface : public mlir::ConvertToLLVMPatternInterface {
  using mlir::ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(ConversionTarget& target,
                                               LLVMTypeConverter& typeConverter,
                                               RewritePatternSet& patterns) const override final {
    populateCoordToLLVMConversionPatterns(typeConverter, patterns);
  }
};

void CoordDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
  >();

  registerTypes();

  addInterfaces<
    ConvertToLLVMInterface
  >();
}

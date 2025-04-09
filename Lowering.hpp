#pragma once

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;

namespace coord {

void populateCoordToLLVMConversionPatterns(LLVMTypeConverter& typeConverter,
                                           RewritePatternSet& patterns);
}
}

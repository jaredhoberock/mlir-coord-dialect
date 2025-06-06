#pragma once

namespace mlir {

class RewritePatternSet;

namespace coord {

void populateCoordToTraitConversionPatterns(RewritePatternSet& patterns);
}
}

#include "ConvertToTrait.hpp"
#include "Dialect.hpp"
#include "Ops.hpp"
#include <llvm/ADT/STLExtras.h>
#include <iostream>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir-tuple-dialect/cpp/Dialect.hpp>
#include <mlir-trait-dialect/cpp/Dialect.hpp>

#include "Dialect.cpp.inc"

namespace mlir::coord {

struct ConvertToTraitInterface : public mlir::trait::ConvertToTraitPatternInterface {
  using mlir::trait::ConvertToTraitPatternInterface::ConvertToTraitPatternInterface;

  void populateConvertToTraitConversionPatterns(RewritePatternSet& patterns) const override final {
    populateCoordToTraitConversionPatterns(patterns);
  }
};

void CoordDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
  >();

  registerTypes();

  addInterfaces<
    ConvertToTraitInterface
  >();
}

}

//#include <llvm/Support/raw_ostream.h>
//#include <llvm/Support/CommandLine.h>
//#include <llvm/Support/Debug.h>
//
//// Insert the static initializer in an anonymous namespace:
//namespace {
//struct ForceDebugOptions {
//  ForceDebugOptions() {
//    // Supply the command-line arguments you want to force.
//    // This forces MLIR's debug options, for example, "-debug-only=dialect-conversion".
//    const char *argv[] = {"coord_plugin", "-debug-only=dialect-conversion"};
//    llvm::cl::ParseCommandLineOptions(2, argv, "Force MLIR debug options\n");
//    llvm::dbgs() << "Forced debug-only=dialect-conversion\n";
//  }
//};
//// Create a static instance to invoke the constructor at load time.
//static ForceDebugOptions forceDebugOptions;
//} // end anonymous namespace

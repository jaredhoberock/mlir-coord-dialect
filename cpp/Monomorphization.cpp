#include "Dialect.hpp"
#include "Ops.hpp"
#include "Monomorphization.hpp"
#include "Types.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace mlir::coord;
using namespace mlir::func;

/// Build a unique monomorphized function name by appending
/// “_mono” plus each arg‑ and result‑type.
static std::string mangleFunctionName(StringRef baseName,
                                      ArrayRef<Type> argTypes,
                                      ArrayRef<Type> resTypes) {
  std::string out;
  {
    llvm::raw_string_ostream os(out);
    os << baseName << "_mono";
    for (Type t : argTypes) os << "_" << t;
    for (Type t : resTypes) os << "_" << t;
  }
  return out;
}

bool isPolymorph(FuncOp fn) {
  FunctionType ty = fn.getFunctionType();

  return llvm::any_of(ty.getInputs(), [](Type t) { return isa<CoordType>(t); }) ||
         llvm::any_of(ty.getResults(),[](Type t) { return isa<CoordType>(t); });
}

struct ConvertAnyOpWithCoordTypes : ConversionPattern {
  ConvertAnyOpWithCoordTypes(TypeConverter &tc, MLIRContext *ctx)
      : ConversionPattern(tc, MatchAnyOpTypeTag(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto *tc = getTypeConverter();

    bool hasCoordTypes =
        llvm::any_of(op->getOperandTypes(), [](Type t) { return isa<CoordType>(t); }) ||
        llvm::any_of(op->getResultTypes(),  [](Type t) { return isa<CoordType>(t); });

    if (!hasCoordTypes) {
      return failure();
    }

    // Convert result types
    SmallVector<Type> newResults;
    for (Type t : op->getResultTypes()) {
      Type converted = tc->convertType(t);
      if (!converted) {
        return failure();
      }
      newResults.push_back(converted);
    }

    // Rebuild the op (using create with OperationState)
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(newResults);
    state.addAttributes(op->getAttrs());
    for (Region &region : op->getRegions())
      state.addRegion()->takeBody(region);

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

func::FuncOp monomorphize(func::FuncOp polymorph, Type concreteType) {
  if (polymorph.isExternal()) {
    polymorph.emitError("cannot monomorphize external function");
    return nullptr;
  }

  if (!isPolymorph(polymorph)) {
    polymorph.emitError("cannot monomorphize function that is not polymorphic");
    return nullptr;
  }

  auto module = polymorph->getParentOfType<ModuleOp>();

  // Compute the concrete argument and result types
  FunctionType polymorphicType = polymorph.getFunctionType();
  SmallVector<Type> newArgTypes, newResTypes;

  for (Type t : polymorphicType.getInputs())
    newArgTypes.push_back(isa<CoordType>(t) ? concreteType : t);
  for (Type t : polymorphicType.getResults())
    newResTypes.push_back(isa<CoordType>(t) ? concreteType : t);

  auto monomorphName = mangleFunctionName(polymorph.getName(), newArgTypes, newResTypes);
  if (auto existing = module.lookupSymbol<FuncOp>(monomorphName))
    return existing;

  auto* ctx = polymorph.getContext();

  // clone the polymorph
  OpBuilder builder(ctx);
  FuncOp monomorph = cast<func::FuncOp>(builder.clone(*polymorph));
  monomorph.setName(monomorphName);

  // type converter will convert CoordType to concreteType
  TypeConverter typeConverter;
  typeConverter.addConversion([=](Type t) -> std::optional<Type> {
    if (isa<CoordType>(t))
      return concreteType;
    return t;
  });

  // mark illegal any operation which involves !coord.coord
  // ConvertAnyOpWithCoordTypes will monomorphize these operations
  ConversionTarget target(*ctx);
  target.addDynamicallyLegalOp<FuncOp>([](FuncOp fn) {
    auto isCoord = [](Type t) { return isa<CoordType>(t); };
    auto fty = fn.getFunctionType();
    return llvm::none_of(fty.getInputs(), isCoord) &&
           llvm::none_of(fty.getResults(), isCoord);
  });
  target.markUnknownOpDynamicallyLegal([](Operation *op) {
    auto hasCoordType = [](Type t) { return isa<CoordType>(t); };
    return llvm::none_of(op->getOperandTypes(), hasCoordType) &&
           llvm::none_of(op->getResultTypes(), hasCoordType);
  });

  RewritePatternSet patterns(ctx);
  patterns.add<ConvertAnyOpWithCoordTypes>(typeConverter, ctx);
  populateFunctionOpInterfaceTypeConversionPattern("func.func", patterns, typeConverter);
  if (failed(applyPartialConversion(monomorph, target, FrozenRewritePatternSet(std::move(patterns))))) {
    monomorph.erase();
    return nullptr;
  }

  return monomorph;
}

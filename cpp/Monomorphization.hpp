#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>

bool isPolymorph(mlir::func::FuncOp fn);

mlir::func::FuncOp monomorphize(mlir::func::FuncOp polymorph,
                                mlir::Type concreteType);

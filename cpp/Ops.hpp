#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>

#include "OpInterfaces.hpp.inc"

#define GET_OP_CLASSES
#include "Ops.hpp.inc"

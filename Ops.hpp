#pragma once

#include "Types.hpp"
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>

#define GET_OP_CLASSES
#include "Ops.hpp.inc"

#pragma once

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Manually register the coord dialect with a context.
void coordRegisterDialect(MlirContext ctx);

/// Create a coord.add operation.
MlirOperation coordAddOpCreate(MlirLocation loc, MlirValue lhs, MlirValue rhs,
                               MlirType resultType);

/// Create a coord.sub operation.
MlirOperation coordSubOpCreate(MlirLocation loc, MlirValue lhs, MlirValue rhs,
                               MlirType resultType);

/// Return the !coord.coord type.
MlirType coordCoordTypeGet(MlirContext ctx);

/// Check whether a type is a !coord.coord type.
bool coordTypeIsCoord(MlirType type);

#ifdef __cplusplus
}
#endif

#pragma once

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Manually register the coord dialect with a context.
void coordRegisterDialect(MlirContext ctx);

/// Returns a coord.coord<shape> type.
MlirType coordCoordTypeGet(MlirContext ctx, int64_t shape);

/// Checks if the given type is a coord.coord.
bool coordTypeIsCoord(MlirType type);

/// Extract the shape value from a coord.coord type.
int64_t coordCoordTypeGetShape(MlirType type);

/// Create a coord.make operation.
MlirOperation coordMakeOpCreate(MlirLocation loc, MlirType resultType,
                                MlirValue* elements, intptr_t nElements);

/// Create a coord.sum operation.
MlirOperation coordSumOpCreate(MlirLocation loc, MlirValue lhs, MlirValue rhs,
                               MlirType resultType);

#ifdef __cplusplus
}
#endif

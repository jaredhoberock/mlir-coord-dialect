#pragma once

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Manually register the coord dialect with a context.
void coordRegisterDialect(MlirContext ctx);

/// Create a coord.make_tuple operation.
MlirOperation coordMakeTupleOpCreate(MlirLocation loc, MlirType resultType,
                                     MlirValue* elements, intptr_t nElements);

/// Create a coord.sum operation.
MlirOperation coordSumOpCreate(MlirLocation loc, MlirValue lhs, MlirValue rhs,
                               MlirType resultType);

/// Return the !coord.coord type.
MlirType coordCoordTypeGet(MlirContext ctx);

/// Check whether a type is a !coord.coord type.
bool coordTypeIsCoord(MlirType type);

#ifdef __cplusplus
}
#endif

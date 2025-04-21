use melior::{ir::{Location, Type, TypeLike, Value, ValueLike, Operation}, pass::Pass, Context, StringRef};
use mlir_sys::{MlirContext, MlirLocation, MlirOperation, MlirPass, MlirStringRef, MlirType, MlirValue};

#[link(name = "coord_dialect")]
unsafe extern "C" {
    fn coordRegisterDialect(ctx: MlirContext);
    fn coordMakeTupleOpCreate(loc: MlirLocation, result_ty: MlirType, elements: *const MlirValue, n: isize) -> MlirOperation;
    fn coordMonoCallOpCreate(loc: MlirLocation, callee: MlirStringRef,
                             arguments: *const MlirValue, n_arguments: isize,
                             result_types: *const MlirType, n_results: isize) -> MlirOperation;
    fn coordSumOpCreate(loc: MlirLocation, lhs: MlirValue, rhs: MlirValue, result_ty: MlirType) -> MlirOperation;
    fn coordCreateConvertCoordToLLVMPass() -> MlirPass;
    fn coordCoordTypeGet(ctx: MlirContext) -> MlirType;
    fn coordTypeIsCoord(ty: MlirType) -> bool;
}

pub fn register(context: &Context) {
    unsafe { coordRegisterDialect(context.to_raw()) }
}

pub fn create_coord_to_llvm() -> Pass {
    unsafe {
        Pass::from_raw(coordCreateConvertCoordToLLVMPass())
    }
}

pub fn make_tuple<'c>(loc: Location<'c>, result_ty: Type<'c>, values: &[Value<'c, '_>]) -> Operation<'c> {
    let op = unsafe {
        coordMakeTupleOpCreate(loc.to_raw(), result_ty.to_raw(), values.as_ptr() as *const _, values.len() as isize)
    };
    unsafe { Operation::from_raw(op) }
}

pub fn mono_call<'c>(
    loc: Location<'c>,
    callee: &str,
    arguments: &[Value<'c, '_>],
    result_types: &[Type<'c>],
) -> Operation<'c> {
    let callee_ref = StringRef::new(callee);
    let op = unsafe {
        coordMonoCallOpCreate(
            loc.to_raw(),
            callee_ref.to_raw(),
            arguments.as_ptr() as *const _,
            arguments.len() as isize,
            result_types.as_ptr() as *const _,
            result_types.len() as isize,
        )
    };
    unsafe { Operation::from_raw(op) }
}

pub fn sum<'c>(loc: Location<'c>, lhs: Value<'c, '_>, rhs: Value<'c, '_>, result_ty: Type<'c>) -> Operation<'c> {
    let op = unsafe {
        coordSumOpCreate(loc.to_raw(), lhs.to_raw(), rhs.to_raw(), result_ty.to_raw())
    };
    unsafe { Operation::from_raw(op) }
}

/// Construct the polymorphic `!coord.coord` type.
pub fn coord_type(context: &Context) -> Type {
    unsafe { Type::from_raw(coordCoordTypeGet(context.to_raw())) }
}

/// Check whether a type is the polymorphic `!coord.coord` type.
pub fn is_coord_type(ty: Type) -> bool {
    unsafe { coordTypeIsCoord(ty.to_raw()) }
}

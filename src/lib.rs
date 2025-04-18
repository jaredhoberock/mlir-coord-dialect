use melior::{Context, ir::{Location, Type, TypeLike, Value, ValueLike, Operation}, pass::Pass};
use mlir_sys::{MlirContext, MlirLocation, MlirOperation, MlirPass, MlirType, MlirValue};

#[link(name = "coord_dialect")]
unsafe extern "C" {
    fn coordRegisterDialect(ctx: MlirContext);
    fn coordMakeTupleOpCreate(loc: MlirLocation, result_ty: MlirType, elements: *const MlirValue, n: isize) -> MlirOperation;
    fn coordSumOpCreate(loc: MlirLocation, lhs: MlirValue, rhs: MlirValue, result_ty: MlirType) -> MlirOperation;
    fn coordCreateConvertCoordToLLVMPass() -> MlirPass;
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

pub fn sum<'c>(loc: Location<'c>, lhs: Value<'c, '_>, rhs: Value<'c, '_>, result_ty: Type<'c>) -> Operation<'c> {
    let op = unsafe {
        coordSumOpCreate(loc.to_raw(), lhs.to_raw(), rhs.to_raw(), result_ty.to_raw())
    };
    unsafe { Operation::from_raw(op) }
}

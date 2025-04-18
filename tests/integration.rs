use coord_dialect::{register, sum};
use melior::{
    Context,
    dialect::{func, DialectRegistry},
    ExecutionEngine,
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        Attribute,
        r#type::{FunctionType, IntegerType, TupleType, Type},
        Block, BlockLike, Location, Module, Region, RegionLike,
    },
    pass::{self, PassManager},
    utility::{register_all_dialects},
};

#[test]
fn test_coordinate_sum_jit() {
    // create a dialect registry and register all dialects
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    register(&context);

    // make all the dialects available
    context.load_all_available_dialects();

    // begin creating a module
    let location = Location::unknown(&context);
    let mut module = Module::new(location);

    let i64_ty = IntegerType::new(&context, 64).into();
    let coord_ty: Type = TupleType::new(&context, &[i64_ty, i64_ty]).into();

    let function_type = FunctionType::new(&context, &[coord_ty, coord_ty], &[coord_ty]);

    // Build the function body: %sum = coord.sum %a, %b
    let region = {
        let block = Block::new(&[(coord_ty, location), (coord_ty, location)]);
        
        let sum = block.append_operation(sum(
            location, 
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
            coord_ty,
        ));

        block.append_operation(func::r#return(&[sum.result(0).unwrap().into()], location));

        let region = Region::new();
        region.append_block(block);
        region
    };

    // Define the function
    let mut func_op = func::func(
        &context,
        StringAttribute::new(&context, "run"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        location,
    );

    // this attribute tells MLIR to create an additional wrapper function that we can use 
    // to invoke "run" via invoke_packed below
    func_op.set_attribute("llvm.emit_c_interface", Attribute::unit(&context));

    module.body().append_operation(func_op);
    assert!(module.as_operation().verify(), "MLIR module verification failed");

    // Lower to LLVM
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    assert!(pass_manager.run(&mut module).is_ok());

    // JIT compile the module
    let engine = ExecutionEngine::new(&module, 0, &[], false);

    // test that we can call the function and it produces the expected result
    let mut a = [5i64, 6i64];
    let mut b = [6i64, 16i64];
    let mut result = [0i64, 0i64];
    
    let mut args: [*mut (); 3] = [
        a.as_mut_ptr().cast::<()>(),
        b.as_mut_ptr().cast::<()>(),
        result.as_mut_ptr().cast::<()>(),
    ];
    
    unsafe {
        engine
            .invoke_packed("run", &mut args)
            .expect("JIT invocation failed");
    }
    
    assert_eq!(result, [11, 22]);
}

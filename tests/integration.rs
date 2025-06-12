use melior::{
    dialect::{func, DialectRegistry},
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType, TupleType, Type},
        Block, BlockLike, Location, Module, Operation, Region, RegionLike,
    },
    pass::{self, PassManager},
    utility::{register_all_dialects},
    Context,
    ExecutionEngine,
    StringRef,
};

use inline_dialect as inline;
use trait_dialect as trait_;
use coord_dialect as coord;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
struct I32x2(i32, i32);

fn build_run_func<'c>(
    ctx: &'c Context,
    loc: Location<'c>
) -> Operation<'c> {
    let i32_ty = IntegerType::new(ctx, 32).into();
    let coord_ty: Type = TupleType::new(ctx, &[i32_ty, i32_ty]).into();

    let function_type = FunctionType::new(ctx, &[coord_ty, coord_ty], &[coord_ty]);

    // Build the function body: %sum = coord.add %a, %b
    let region = {
        let block = Block::new(&[(coord_ty, loc), (coord_ty, loc)]);
        
        let sum = block.append_operation(coord::add(
            loc, 
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
            coord_ty,
        ));

        block.append_operation(func::r#return(&[sum.result(0).unwrap().into()], loc));

        let region = Region::new();
        region.append_block(block);
        region
    };

    // Define the function
    func::func(
        ctx,
        StringAttribute::new(ctx, "run"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        loc,
    )
}

fn append_invoke_run<'c>(
    block: &Block<'c>,
    loc: Location<'c>
) {
    let source = r#"
    func.func public @invoke_run(%res_ptr: !llvm.ptr,
                                 %a_ptr: !llvm.ptr,
                                 %b_ptr: !llvm.ptr)
      attributes { llvm.emit_c_interface } {
      %a = llvm.load %a_ptr : !llvm.ptr -> !llvm.struct<(i32, i32)>
      %b = llvm.load %b_ptr : !llvm.ptr -> !llvm.struct<(i32, i32)>
      %res = llvm.call @run(%a, %b) : (!llvm.struct<(i32, i32)>, !llvm.struct<(i32, i32)>) -> !llvm.struct<(i32, i32)>
      llvm.store %res, %res_ptr : !llvm.struct<(i32, i32)>, !llvm.ptr
      return
    }
    "#;

    inline::parse_source_into_block(
        loc,
        &[],
        &[],
        StringRef::new(source),
        block,
    ).expect("valid func.func");
}

#[test]
fn test_coordinate_add_jit() {
    // create a dialect registry and register all dialects
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    coord::register(&context);
    inline::register(&context);

    // make all the dialects available
    context.load_all_available_dialects();

    // begin creating a module
    let location = Location::unknown(&context);
    let mut module = Module::new(location);

    module.body().append_operation(
        build_run_func(&context, location)
    );

    assert!(module.as_operation().verify(), "MLIR module verification failed");

    // Lower to LLVM
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(trait_::create_monomorphize_pass());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    assert!(pass_manager.run(&mut module).is_ok());

    // add invoke_run after tuples have been lowered to !llvm.struct
    // and then lower to LLVM again
    append_invoke_run(&module.body(), location);
    assert!(pass_manager.run(&mut module).is_ok());

    // JIT compile the module
    let engine = ExecutionEngine::new(&module, 0, &[], false);

    // test that we can call the function and it produces the expected result
    let mut result = I32x2(0, 0);
    let a = I32x2(5, 6);
    let b = I32x2(6, 16);
    
    let invoke_run: extern "C" fn(*mut I32x2, *const I32x2, *const I32x2);
    invoke_run = unsafe {
        std::mem::transmute(engine.lookup("invoke_run"))
    };

    invoke_run(&mut result, &a, &b);
    
    assert_eq!(result, I32x2(11, 22));
}

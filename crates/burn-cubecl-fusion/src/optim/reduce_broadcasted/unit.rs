use crate::{
    engine::codegen::{
        DynElem, DynSize, DynVector,
        ir::{
            FuseArg, FuseBlockConfig, FuseType, GlobalArgs, MultiBlockPos,
            multi_block_variables_init,
        },
        kernel::{fuse_on_write, init_locals},
    },
    optim::reduce::args::{FusedReduceArgs, FusedReduceInput, FusedReduceOutput},
};
use cubecl::{
    define_size,
    ir::ElemType,
    prelude::{polyfills::set_polyfill, *},
    std::tensor::r#virtual::VirtualTensor,
};
use cubek::reduce::{
    ReduceInstruction, ReducePrecision, VectorizationMode,
    components::{
        args::NumericVector,
        global::{
            cube::GlobalFullCubeReduce, plane::GlobalFullPlaneReduce, unit::GlobalFullUnitReduce,
        },
        instructions::{ReduceOperation, ReduceOperationConfig},
    },
    init_tensors,
    routines::GlobalReduceBlueprint,
};

/// A configuration block for a reduction operation within a fused kernel.
///
/// This struct holds all the compile-time information needed to perform a
/// reduction, including the operation type (Sum, Max, etc.) and the layout
/// configuration for both input and output.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct ReduceFuseBlock {
    #[cube(comptime)]
    op: ReduceOperationConfig,
    #[cube(comptime)]
    config_input: FuseBlockConfig,
    #[cube(comptime)]
    config_output: FuseBlockConfig,
    #[cube(comptime)]
    input: FuseArg,
    #[cube(comptime)]
    output: FuseArg,
    /// Which workers reduce a row: one unit, one plane or one cube. The trailing
    /// elementwise block is spread over the same workers.
    #[cube(comptime)]
    blueprint: GlobalReduceBlueprint,
}

/// A configuration block for an elementwise operation that follows a reduction.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct ElemwiseFuseBlock {
    #[cube(comptime)]
    config: FuseBlockConfig,
}

/// The entry point for a broadcasted reduction kernel.
///
/// This kernel initializes local variables for multiple reduction blocks and then
/// executes the reduction sequence.
///
/// # Arguments
///
/// * `inputs` - Global arguments containing input tensor handles.
/// * `outputs` - Global arguments containing output tensor handles.
/// * `reduce_axis` - The dimension along which the reduction is performed.
/// * `rows` - How many vectors are reduced, one per worker.
/// * `blocks` - A sequence of reduction operations to execute.
/// * `block_end` - An optional elementwise block to execute after reductions are complete.
#[cube(launch_unchecked, address_type = "dynamic")]
pub fn reduce_kernel_broadcasted(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    reduce_axis: usize,
    out_vec_axis: usize,
    rows: usize,
    blocks: Sequence<ReduceFuseBlock>,
    block_end: ComptimeOption<ElemwiseFuseBlock>,
) {
    #[unroll]
    for i in 0..blocks.len() {
        let block = blocks.index(i);
        multi_block_variables_init(&block.config_input, &mut outputs.variables);
        multi_block_variables_init(&block.config_output, &mut outputs.variables);
    }

    reduce_many(
        inputs,
        outputs,
        reduce_axis,
        out_vec_axis,
        rows,
        blocks,
        block_end,
    );
}

define_scalar!(In);
define_scalar!(Acc);
define_scalar!(Out);

define_size!(InSize);
define_size!(OutSize);

/// Configures the precision polyfills for the reduction based on the block's `FuseType`.
#[cube]
fn set_polyfill_block(block: &ReduceFuseBlock) {
    let input_precision = comptime!(block.input.precision());
    let output_precision = comptime!(block.output.precision());
    let acc_precision = comptime!(match input_precision {
        FuseType::F64 => FuseType::F64,
        FuseType::F32 => FuseType::F32,
        FuseType::Flex32 => FuseType::F32,
        FuseType::F16 => FuseType::F32,
        FuseType::BF16 => FuseType::F32,
        FuseType::I64 => FuseType::I64,
        FuseType::I32 => FuseType::I32,
        FuseType::I16 => FuseType::I32,
        FuseType::I8 => FuseType::I32,
        FuseType::U64 => FuseType::U64,
        FuseType::U32 => FuseType::U32,
        FuseType::U16 => FuseType::U32,
        FuseType::U8 => FuseType::U32,
    });

    set_polyfill::<In, InSize>(comptime!(
        input_precision.into_type(block.config_input.width)
    ));
    set_polyfill::<Out, OutSize>(comptime!(
        output_precision.into_type(block.config_output.width)
    ));
    set_polyfill::<Acc, InSize>(comptime!(acc_precision.into_type(block.config_input.width)));
}

/// Internal logic for executing a sequence of reduction blocks followed by an optional
/// trailing elementwise block.
///
/// The elementwise block runs over the row each worker reduced: a unit walks its row
/// alone, while the lanes of a plane or the units of a cube stride over theirs, after a
/// sync that makes the reduced value written by their first lane visible to the others.
#[cube]
#[allow(clippy::clone_on_copy)]
fn reduce_many(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    reduce_axis: usize,
    out_vec_axis: usize,
    rows: usize,
    blocks: Sequence<ReduceFuseBlock>,
    block_end: ComptimeOption<ElemwiseFuseBlock>,
) {
    let mut axis_size = 0;

    #[unroll]
    for i in 0..blocks.len() {
        let block = blocks.index(i);
        let input = FusedReduceInput {
            global: inputs.clone(),
            config: comptime!(block.config_input.clone()),
            arg: comptime!(block.input.clone()),
        };
        let global = outputs.clone();
        let config = comptime!(block.config_output.clone());
        let arg = comptime!(block.output.clone());
        let mut output = FusedReduceOutput {
            global,
            config,
            arg,
        };

        set_polyfill_block(block);
        let (input, mut output) =
            init_tensors::<FusedReduceArgs, In, InSize, Out, OutSize>(&input, &mut output);

        axis_size = reduce_step::<(In, InSize, Acc), (Out, OutSize), ReduceOperation>(
            &input,
            &mut output,
            reduce_axis,
            out_vec_axis,
            block.op,
            comptime!(block.blueprint.clone()),
        );

        let shared = comptime!(!matches!(block.blueprint, GlobalReduceBlueprint::Unit(_)));

        #[comptime]
        if shared {
            share_variables(
                outputs,
                &block.config_output,
                comptime!(block.blueprint.clone()),
            );
        }
    }

    #[comptime]
    if let ComptimeOption::Some(block) = block_end {
        let first = blocks.index(0);
        let blueprint = comptime!(first.blueprint.clone());

        let mut row = ABSOLUTE_POS;
        let mut lane = 0usize.runtime();
        let mut stride = 1usize.runtime();

        #[comptime]
        if let GlobalReduceBlueprint::Plane(_) = blueprint {
            row = CUBE_POS * CUBE_DIM_Y as usize + UNIT_POS_Y as usize;
            lane = UNIT_POS_X as usize;
            stride = CUBE_DIM_X as usize;
            sync_plane();
        }

        #[comptime]
        if let GlobalReduceBlueprint::Cube(_) = blueprint {
            row = CUBE_POS;
            lane = UNIT_POS as usize;
            stride = CUBE_DIM as usize;
            sync_cube();
        }

        let width = block.config.width;
        let num_iter = axis_size / width;
        let size!(N) = width;

        // Workers past the last row are terminated inside the reduce on the backends
        // that support it; the check covers the ones that mask instead.
        if row < rows {
            for i in range_stepped(lane, num_iter, stride) {
                // Register block local inputs.
                let values = Registry::<FuseArg, Vector<f32, N>>::new();
                let args = comptime![Vec::<FuseArg>::new()];
                let index = row * num_iter + i;
                let mut locals = init_locals(inputs, outputs, &block.config);

                fuse_on_write::<f32, N>(
                    inputs,
                    outputs,
                    &mut locals,
                    index,
                    values,
                    args,
                    &block.config.clone(),
                )
            }
        }
    }
}

/// Hands the multi-block variables of a reduce's write block to every unit of the plane or
/// cube that reduced the row. They are registers: only the routine's writing unit holds the
/// reduced value and whatever the write block derived from it, while the next block's read
/// and the trailing elementwise block run on every unit.
#[cube]
fn share_variables(
    outputs: &mut GlobalArgs,
    #[comptime] block: &FuseBlockConfig,
    #[comptime] blueprint: GlobalReduceBlueprint,
) {
    let keys = comptime! {
        let mut keys = Vec::<(MultiBlockPos, ElemType)>::new();
        block.multi_block_variables(&mut keys);
        keys
    };

    #[unroll]
    for i in 0..comptime!(keys.len()) {
        let (key, dtype) = comptime!(keys.get(i).unwrap().clone());
        set_polyfill::<DynElem, DynSize>(comptime![Type::new(dtype).with_vector_size(block.width)]);

        #[comptime]
        if let GlobalReduceBlueprint::Plane(_) = blueprint {
            let value = outputs.variables.read(comptime!(key.clone()));
            outputs
                .variables
                .write(comptime!(key.clone()), plane_broadcast(value, 0u32));
        }

        #[comptime]
        if let GlobalReduceBlueprint::Cube(_) = blueprint {
            let mut slot = Shared::<[DynVector]>::new_slice(1usize);
            if UNIT_POS == 0 {
                slot[0] = outputs.variables.read(comptime!(key.clone()));
            }
            sync_cube();
            let value = slot[0];
            outputs.variables.write(comptime!(key.clone()), value);
        }
    }
}

#[cube]
/// Executes a single reduction step with the routine the blueprint names.
///
/// Returns the size of the axis that was reduced.
fn reduce_step<P: ReducePrecision, Out: NumericVector, I: ReduceInstruction<P>>(
    input: &VirtualTensor<P::EI, P::SI>,
    output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
    reduce_axis: usize,
    out_vec_axis: usize,
    #[comptime] config: I::Config,
    #[comptime] blueprint: GlobalReduceBlueprint,
) -> usize {
    let inst = I::from_config(config);
    let axis_size = input.shape(reduce_axis);

    match blueprint {
        GlobalReduceBlueprint::Unit(unit) => {
            GlobalFullUnitReduce::execute::<P, Out, I>(
                input,
                output,
                reduce_axis,
                out_vec_axis,
                &inst,
                VectorizationMode::Parallel,
                unit,
            );
        }
        GlobalReduceBlueprint::Plane(plane) => {
            GlobalFullPlaneReduce::execute::<P, Out, I>(
                input,
                output,
                reduce_axis,
                out_vec_axis,
                &inst,
                VectorizationMode::Parallel,
                plane,
            );
        }
        GlobalReduceBlueprint::Cube(cube) => {
            GlobalFullCubeReduce::execute::<P, Out, I>(
                input,
                output,
                reduce_axis,
                out_vec_axis,
                &inst,
                VectorizationMode::Parallel,
                cube,
            );
        }
    }

    axis_size
}

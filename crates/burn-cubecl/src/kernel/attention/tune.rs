use crate::{
    CubeTuneId,
    kernel::attention::{
        AttentionStrategy, Degrade, attention, attention_degrading,
        bounds::{Inputs, with_attention_bounds},
    },
    tensor::CubeTensor,
};
use burn_backend::DType;
use burn_backend::cubecl::{dtype_to_elem_type, dtype_to_storage_type};
use burn_backend::ops::AttentionModuleOptions;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner};
use cubek::attention::forward::{
    launch::AttentionAutotuneKey, routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
};

/// Executes autotune on attention operations
pub fn attention_autotune(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    mask: Option<CubeTensor>,
    attn_bias: Option<CubeTensor>,
    options: AttentionModuleOptions,
) -> CubeTensor {
    let client = query.client.clone();

    let accelerated_client = client.clone();

    static TUNER: LocalTuner<AttentionAutotuneKey, CubeTuneId> = local_tuner!();

    let tune_id = CubeTuneId::new(&accelerated_client, &query.device);
    let tunables = TUNER.init(&tune_id, move || {
        const PRIORITY_MAX: i8 = 3;
        const PRIORITY_HIGH: i8 = 2;
        const PRIORITY_LOW: i8 = 1;
        const PRIORITY_MIN: i8 = 0;

        // The blackbox routine runs its tile matmuls on f16 fragments whatever the problem
        // dtype is, and validates them against the `cmma` feature list, so that's what has to
        // be available here — `mma` or `tma` support alone wouldn't let the kernel run.
        let f16 = dtype_to_storage_type(DType::F16);
        let has_accelerated = client
            .properties()
            .features
            .matmul
            .cmma
            .iter()
            .any(|config| config.a_type == f16 && config.b_type == f16);

        // One group: the tuner tries the candidates a priority level at a time, highest
        // first, and stops at the first level where one runs. The levels, per shape:
        //
        // - `seq_q <= head_dim`: the fallback with the blackbox kernels, then the unit kernel;
        // - longer, with an accelerator: the blackbox kernels, then the fallback, then the
        //   unit kernel;
        // - longer, without one: the unit kernel with the (unsupported) blackbox kernels,
        //   then the fallback.
        let group = TuneGroup::<AttentionAutotuneKey>::new("attention", |_key| PRIORITY_MAX);

        let mut set = with_attention_bounds(TunableSet::new(create_key, input_gen));

        // First entry should always work, since it is considered the fallback.
        set = set.with(
            Tunable::new(
                "fallback",
                |(query, key, value, mask, attn_bias, options, _degrade)| {
                    attention(
                        query,
                        key,
                        value,
                        mask,
                        attn_bias,
                        options,
                        AttentionStrategy::Fallback,
                    )
                    .map_err(|err| std::format!("{err:?}"))
                },
            )
            .group(&group, move |key| {
                // The fallback materializes the full (total_batches, seq_q, seq_kv)
                // score matrix, which the flash kernels never allocate — and even
                // *benchmarking* it pays that allocation. Let it compete only while
                // that matrix is no bigger than an activation the model already
                // produces — `[batch, seq_kv, d_model]`, a full-head K/V-sized
                // tensor — so it fits a memory budget sized for the model's own
                // activations. With `total_batches = batch · heads` and
                // `d_model = heads · head_dim`, the bound reduces to
                // `seq_q <= head_dim`: decode-like and short-chunk shapes qualify,
                // long-prefill shapes never do.
                if key.seq_q <= key.head_dim {
                    PRIORITY_MAX
                } else if has_accelerated {
                    // For those it follows the blackbox kernels, which are expected to
                    // win, and precedes the unit kernel: where no blackbox kernel
                    // launches the shape (none does at 512 queries of head dim 128),
                    // the unit kernel takes minutes to compile and ran ten times slower
                    // than the fallback, whose score matrix is then the price.
                    PRIORITY_HIGH
                } else {
                    // Without an accelerator the unit kernel is the only flash kernel,
                    // and the fallback stays the last resort, since an
                    // O(seq_q · seq_kv) spike can exceed a flash-sized memory budget.
                    PRIORITY_MIN
                }
            }),
        );

        let seq_q = 1;
        let seq_kv = 1;
        for num_planes in [2, 4, 8] {
            let name = format!("blackbox_accelerated_{num_planes}_planes_p_{seq_q}-{seq_kv}");
            set = set.with(
                Tunable::new(
                    &name,
                    move |(query, key, value, mask, attn_bias, options, degrade)| {
                        attention_degrading(
                            query,
                            key,
                            value,
                            mask,
                            attn_bias,
                            options,
                            AttentionStrategy::FlashBlackboxAccelerated(
                                BlackboxAcceleratedStrategy {
                                    num_planes,
                                    seq_q,
                                    seq_kv,
                                },
                            ),
                            degrade,
                        )
                        .map_err(|err| std::format!("{err:?}"))
                    },
                )
                .group(&group, move |_key| {
                    // Unsupported kernels keep a low priority rather than being discarded,
                    // so they remain a last resort and the tune plan can never end up empty.
                    if has_accelerated {
                        PRIORITY_MAX
                    } else {
                        PRIORITY_LOW
                    }
                }),
            );
        }

        set = set.with(
            Tunable::new(
                "unit",
                |(query, key, value, mask, attn_bias, options, degrade)| {
                    attention_degrading(
                        query,
                        key,
                        value,
                        mask,
                        attn_bias,
                        options,
                        AttentionStrategy::FlashUnit,
                        degrade,
                    )
                    .map_err(|err| std::format!("{err:?}"))
                },
            )
            .group(&group, |_key| PRIORITY_LOW),
        );

        set
    });

    // The call's own inputs, which a cached winner runs on. The selection is cached per
    // anchored key, so a flash winner can meet a raw shape it cannot launch and must be
    // allowed to degrade; only the benchmark inputs forbid it, see `input_gen`.
    TUNER.execute(
        &tune_id,
        &accelerated_client,
        tunables,
        (
            query,
            key,
            value,
            mask,
            attn_bias,
            options,
            Degrade::ToFallback,
        ),
    )
}

fn create_key(
    (query, key, value, mask, _attn_bias, _options, _degrade): &Inputs,
) -> AttentionAutotuneKey {
    let total_batches = query.meta.shape[0] * query.meta.shape[1];
    let seq_q = query.meta.shape[2];
    let head_dim = query.meta.shape[3];
    let seq_kv = value.meta.shape[2];
    let val_dim = value.meta.shape[3];

    AttentionAutotuneKey::generate(
        dtype_to_elem_type(query.dtype),
        dtype_to_elem_type(key.dtype),
        dtype_to_elem_type(value.dtype),
        dtype_to_elem_type(query.dtype),
        total_batches,
        seq_q,
        head_dim,
        seq_kv,
        val_dim,
        mask.is_some(),
    )
}

/// The benchmark inputs: the call's tensors, run without degrading, so that a flash candidate
/// which cannot launch the shape loses instead of standing in for the fallback.
fn input_gen(
    _key: &AttentionAutotuneKey,
    (query, key, value, mask, attn_bias, options, _degrade): &Inputs,
) -> Inputs {
    (
        query.clone(),
        key.clone(),
        value.clone(),
        mask.clone(),
        attn_bias.clone(),
        *options,
        Degrade::Never,
    )
}

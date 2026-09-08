//! Qwen3-TTS-Tokenizer-12Hz: the neural audio codec used by Qwen3-TTS.
//!
//! The decoder turns 12.5 Hz frames of 16 codebook indices into 24 kHz audio in four stages:
//! a split residual vector quantizer producing latents, a sliding-window transformer, two
//! ConvNeXt upsampling stages and a SnakeBeta/transposed convolution vocoder (each frame yields
//! 1920 samples).
//!
//! The encoder, which is only needed to compute the codes of a reference recording for voice
//! cloning, is not ported here.

use burn::module::Param;
use burn::nn::conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear};
use burn::prelude::*;
use burn::tensor::activation::gelu;
use burn::tensor::ops::PadMode;
use burn_store::{KeyRemapper, ModuleSnapshot, SafetensorsStore};

use crate::config::{DecoderConfig, SpeechTokenizerConfig};
use crate::transformer::{Transformer, TransformerConfig, TransformerState};

impl DecoderConfig {
    fn transformer_config(&self) -> TransformerConfig {
        TransformerConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            hidden_act: self.hidden_act,
            attention_bias: self.attention_bias,
            qk_norm: false,
            layer_scale: true,
            sliding_window: self.sliding_window,
        }
    }
}

/// Extra right padding so that the last frame of a causal convolution is complete.
fn extra_padding(len: usize, kernel_size: usize, padding_total: usize, stride: usize) -> usize {
    let n_frames = (len as f64 + padding_total as f64 - kernel_size as f64) / stride as f64 + 1.;
    let ideal_len =
        (n_frames.ceil() as i64 - 1) * stride as i64 + (kernel_size - padding_total) as i64;
    (ideal_len - len as i64).max(0) as usize
}

/// Conv1d with causal (left) zero padding, matching `Qwen3TTSTokenizerV2CausalConvNet`.
#[derive(Module, Debug)]
struct CausalConv1d {
    conv: Conv1d,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl CausalConv1d {
    fn init(
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        dilation: usize,
        stride: usize,
        groups: usize,
        device: &Device,
    ) -> Self {
        let conv = Conv1dConfig::new(in_c, out_c, kernel_size)
            .with_stride(stride)
            .with_dilation(dilation)
            .with_groups(groups)
            .init(device);
        let kernel_size = (kernel_size - 1) * dilation + 1;
        Self {
            conv,
            kernel_size,
            stride,
            padding: kernel_size - stride,
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let len = xs.dims()[2];
        let extra = extra_padding(len, self.kernel_size, self.padding, self.stride);
        let xs = xs.pad([(self.padding, extra)], PadMode::Constant(0.));
        self.conv.forward(xs)
    }
}

/// ConvTranspose1d whose trailing `kernel_size - stride` samples are trimmed to keep it causal.
#[derive(Module, Debug)]
struct CausalConvTranspose1d {
    conv: ConvTranspose1d,
    right_pad: usize,
}

impl CausalConvTranspose1d {
    fn init(in_c: usize, out_c: usize, kernel_size: usize, stride: usize, device: &Device) -> Self {
        Self {
            conv: ConvTranspose1dConfig::new([in_c, out_c], kernel_size)
                .with_stride(stride)
                .init(device),
            right_pad: kernel_size - stride,
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let xs = self.conv.forward(xs);
        let len = xs.dims()[2];
        xs.narrow(2, 0, len - self.right_pad)
    }
}

#[derive(Module, Debug)]
struct ConvNeXtBlock {
    dwconv: CausalConv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Param<Tensor<1>>,
}

impl ConvNeXtBlock {
    fn init(dim: usize, device: &Device) -> Self {
        Self {
            dwconv: CausalConv1d::init(dim, dim, 7, 1, 1, dim, device),
            norm: LayerNormConfig::new(dim).with_epsilon(1e-6).init(device),
            pwconv1: crate::linear_config(dim, 4 * dim).init(device),
            pwconv2: crate::linear_config(4 * dim, dim).init(device),
            gamma: Param::from_tensor(Tensor::ones([dim], device)),
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        // The pointwise convolutions are linear layers over the channel dimension.
        let hidden = self.dwconv.forward(xs.clone()).swap_dims(1, 2);
        let hidden = self.norm.forward(hidden);
        let hidden = gelu(self.pwconv1.forward(hidden));
        let hidden = self.pwconv2.forward(hidden) * self.gamma.val().unsqueeze();
        xs + hidden.swap_dims(1, 2)
    }
}

/// `x + sin(x * exp(alpha))^2 / (exp(beta) + eps)` with per-channel `alpha` and `beta`.
#[derive(Module, Debug)]
struct SnakeBeta {
    alpha: Param<Tensor<1>>,
    beta: Param<Tensor<1>>,
}

impl SnakeBeta {
    fn init(channels: usize, device: &Device) -> Self {
        Self {
            alpha: Param::from_tensor(Tensor::zeros([channels], device)),
            beta: Param::from_tensor(Tensor::zeros([channels], device)),
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let channels = self.alpha.shape().dims::<1>()[0];
        let alpha = self.alpha.val().exp().reshape([1, channels, 1]);
        let inv_beta = (self.beta.val().exp() + 1e-9)
            .recip()
            .reshape([1, channels, 1]);
        let snake = (xs.clone() * alpha).sin().square() * inv_beta;
        xs + snake
    }
}

#[derive(Module, Debug)]
struct ResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv1d,
    act2: SnakeBeta,
    conv2: CausalConv1d,
}

impl ResidualUnit {
    fn init(dim: usize, dilation: usize, device: &Device) -> Self {
        Self {
            act1: SnakeBeta::init(dim, device),
            conv1: CausalConv1d::init(dim, dim, 7, dilation, 1, 1, device),
            act2: SnakeBeta::init(dim, device),
            conv2: CausalConv1d::init(dim, dim, 1, 1, 1, 1, device),
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let hidden = self.act1.forward(xs.clone());
        let hidden = self.conv1.forward(hidden);
        let hidden = self.act2.forward(hidden);
        xs + self.conv2.forward(hidden)
    }
}

#[derive(Module, Debug)]
struct DecoderBlock {
    act: SnakeBeta,
    upsample: CausalConvTranspose1d,
    residual_units: Vec<ResidualUnit>,
}

impl DecoderBlock {
    fn init(in_dim: usize, out_dim: usize, upsample_rate: usize, device: &Device) -> Self {
        Self {
            act: SnakeBeta::init(in_dim, device),
            upsample: CausalConvTranspose1d::init(
                in_dim,
                out_dim,
                2 * upsample_rate,
                upsample_rate,
                device,
            ),
            residual_units: [1, 3, 9]
                .into_iter()
                .map(|dilation| ResidualUnit::init(out_dim, dilation, device))
                .collect(),
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let mut xs = self.upsample.forward(self.act.forward(xs));
        for unit in self.residual_units.iter() {
            xs = unit.forward(xs);
        }
        xs
    }
}

/// One codebook, stored as the sum of the vectors assigned to each entry and their count.
#[derive(Module, Debug)]
struct Codebook {
    embedding_sum: Param<Tensor<2>>,
    cluster_usage: Param<Tensor<1>>,
}

impl Codebook {
    fn init(codebook_size: usize, dim: usize, device: &Device) -> Self {
        Self {
            embedding_sum: Param::from_tensor(Tensor::zeros([codebook_size, dim], device)),
            cluster_usage: Param::from_tensor(Tensor::ones([codebook_size], device)),
        }
    }

    /// Looks up `ids` (N) and returns the codebook entries (N, dim).
    fn decode(&self, ids: Tensor<1, Int>) -> Tensor<2> {
        let embedding_sum = self.embedding_sum.val().select(0, ids.clone());
        let cluster_usage = self
            .cluster_usage
            .val()
            .select(0, ids)
            .clamp_min(1e-5)
            .unsqueeze_dim::<2>(1);
        embedding_sum / cluster_usage
    }
}

/// Residual vector quantizer decoder: sums the codebook entries of every layer and projects the
/// result back to the latent dimension.
#[derive(Module, Debug)]
struct ResidualVectorQuantizer {
    codebooks: Vec<Codebook>,
    output_proj: Conv1d,
}

impl ResidualVectorQuantizer {
    fn init(
        num_quantizers: usize,
        codebook_size: usize,
        dim: usize,
        output_dim: usize,
        device: &Device,
    ) -> Self {
        Self {
            codebooks: (0..num_quantizers)
                .map(|_| Codebook::init(codebook_size, dim, device))
                .collect(),
            output_proj: Conv1dConfig::new(dim, output_dim, 1)
                .with_bias(false)
                .init(device),
        }
    }

    /// `codes` has shape (B, num_quantizers, T); returns (B, output_dim, T).
    fn decode(&self, codes: Tensor<3, Int>) -> Tensor<3> {
        let [b, nq, t] = codes.dims();
        assert_eq!(
            nq,
            self.codebooks.len(),
            "expected {} quantizer layers, got {nq}",
            self.codebooks.len()
        );
        let mut acc: Option<Tensor<3>> = None;
        for (i, codebook) in self.codebooks.iter().enumerate() {
            let ids = codes.clone().narrow(1, i, 1).reshape([b * t]);
            let quantized = codebook.decode(ids).reshape([b as i32, t as i32, -1]);
            acc = Some(match acc {
                None => quantized,
                Some(acc) => acc + quantized,
            });
        }
        let hidden = acc.expect("no quantizer layers").swap_dims(1, 2);
        self.output_proj.forward(hidden)
    }
}

#[derive(Module, Debug)]
struct SplitResidualVectorQuantizer {
    rvq_first: ResidualVectorQuantizer,
    rvq_rest: ResidualVectorQuantizer,
    n_q_semantic: usize,
}

impl SplitResidualVectorQuantizer {
    fn init(cfg: &DecoderConfig, device: &Device) -> Self {
        let dim = cfg.codebook_dim / 2;
        Self {
            rvq_first: ResidualVectorQuantizer::init(
                cfg.num_semantic_quantizers,
                cfg.codebook_size,
                dim,
                cfg.codebook_dim,
                device,
            ),
            rvq_rest: ResidualVectorQuantizer::init(
                cfg.num_quantizers - cfg.num_semantic_quantizers,
                cfg.codebook_size,
                dim,
                cfg.codebook_dim,
                device,
            ),
            n_q_semantic: cfg.num_semantic_quantizers,
        }
    }

    /// `codes` has shape (B, num_quantizers, T); returns (B, codebook_dim, T).
    fn decode(&self, codes: Tensor<3, Int>) -> Tensor<3> {
        let nq = codes.dims()[1];
        let first = self
            .rvq_first
            .decode(codes.clone().narrow(1, 0, self.n_q_semantic));
        if nq > self.n_q_semantic {
            let rest =
                self.rvq_rest
                    .decode(codes.narrow(1, self.n_q_semantic, nq - self.n_q_semantic));
            first + rest
        } else {
            first
        }
    }
}

#[derive(Module, Debug)]
struct PreTransformer {
    input_proj: Linear,
    model: Transformer,
    output_proj: Linear,
}

impl PreTransformer {
    fn init(cfg: &DecoderConfig, device: &Device) -> Self {
        Self {
            input_proj: crate::linear_config(cfg.latent_dim, cfg.hidden_size).init(device),
            model: Transformer::init(&cfg.transformer_config(), device),
            output_proj: crate::linear_config(cfg.hidden_size, cfg.latent_dim).init(device),
        }
    }

    fn forward(&self, xs: Tensor<3>, state: &mut TransformerState) -> Tensor<3> {
        state.reset();
        let xs = self.input_proj.forward(xs);
        self.output_proj.forward(self.model.forward(xs, 0, state))
    }
}

/// The SnakeBeta/transposed convolution vocoder turning latents into waveform samples.
#[derive(Module, Debug)]
struct Vocoder {
    pre_conv: CausalConv1d,
    blocks: Vec<DecoderBlock>,
    final_act: SnakeBeta,
    final_conv: CausalConv1d,
}

impl Vocoder {
    fn init(cfg: &DecoderConfig, device: &Device) -> Self {
        let blocks = cfg
            .upsample_rates
            .iter()
            .enumerate()
            .map(|(i, &rate)| {
                let in_dim = cfg.decoder_dim / 2usize.pow(i as u32);
                let out_dim = cfg.decoder_dim / 2usize.pow(i as u32 + 1);
                DecoderBlock::init(in_dim, out_dim, rate, device)
            })
            .collect();
        let output_dim = cfg.decoder_dim / 2usize.pow(cfg.upsample_rates.len() as u32);
        Self {
            pre_conv: CausalConv1d::init(cfg.latent_dim, cfg.decoder_dim, 7, 1, 1, 1, device),
            blocks,
            final_act: SnakeBeta::init(output_dim, device),
            final_conv: CausalConv1d::init(output_dim, 1, 7, 1, 1, 1, device),
        }
    }

    fn forward(&self, xs: Tensor<3>) -> Tensor<3> {
        let mut wav = self.pre_conv.forward(xs);
        for block in self.blocks.iter() {
            wav = block.forward(wav);
        }
        let wav = self.final_conv.forward(self.final_act.forward(wav));
        wav.clamp(-1., 1.)
    }
}

/// The speech-tokenizer decoder, the `decoder.` prefix of `speech_tokenizer/model.safetensors`.
#[derive(Module, Debug)]
pub struct Decoder {
    quantizer: SplitResidualVectorQuantizer,
    pre_conv: CausalConv1d,
    pre_transformer: PreTransformer,
    upsample: Vec<UpsampleStage>,
    vocoder: Vocoder,
    num_quantizers: usize,
    total_upsample: usize,
}

#[derive(Module, Debug)]
struct UpsampleStage {
    conv: CausalConvTranspose1d,
    convnext: ConvNeXtBlock,
}

impl Decoder {
    fn init(cfg: &DecoderConfig, device: &Device) -> Self {
        let upsample = cfg
            .upsampling_ratios
            .iter()
            .map(|&factor| UpsampleStage {
                conv: CausalConvTranspose1d::init(
                    cfg.latent_dim,
                    cfg.latent_dim,
                    factor,
                    factor,
                    device,
                ),
                convnext: ConvNeXtBlock::init(cfg.latent_dim, device),
            })
            .collect();
        Self {
            quantizer: SplitResidualVectorQuantizer::init(cfg, device),
            pre_conv: CausalConv1d::init(cfg.codebook_dim, cfg.latent_dim, 3, 1, 1, 1, device),
            pre_transformer: PreTransformer::init(cfg, device),
            upsample,
            vocoder: Vocoder::init(cfg, device),
            num_quantizers: cfg.num_quantizers,
            total_upsample: cfg.total_upsample(),
        }
    }

    /// Decodes a chunk of codes with shape (B, num_quantizers, T) into audio
    /// (B, 1, T * total_upsample).
    fn forward(&self, codes: Tensor<3, Int>, state: &mut TransformerState) -> Tensor<3> {
        let nq = codes.dims()[1];
        assert_eq!(
            nq, self.num_quantizers,
            "expected {} layers of codes, got {nq}",
            self.num_quantizers
        );
        let hidden = self.quantizer.decode(codes);
        let hidden = self.pre_conv.forward(hidden).swap_dims(1, 2);
        let hidden = self.pre_transformer.forward(hidden, state);
        let mut hidden = hidden.swap_dims(1, 2);
        for stage in self.upsample.iter() {
            hidden = stage.convnext.forward(stage.conv.forward(hidden));
        }
        self.vocoder.forward(hidden)
    }
}

/// The root of `speech_tokenizer/model.safetensors`.
#[derive(Module, Debug)]
pub struct SpeechTokenizerModel {
    decoder: Decoder,
}

/// The speech tokenizer together with the state its transformer needs.
#[derive(Debug)]
pub struct SpeechTokenizer {
    model: SpeechTokenizerModel,
    state: TransformerState,
    config: SpeechTokenizerConfig,
    device: Device,
}

impl SpeechTokenizer {
    /// Loads the decoder from a `speech_tokenizer/model.safetensors` file. The codec always
    /// runs in f32, as in the reference implementation.
    pub fn load(
        cfg: &SpeechTokenizerConfig,
        weights: &std::path::Path,
        device: &Device,
    ) -> Result<Self, String> {
        let decoder_cfg = &cfg.decoder_config;
        let mut model = SpeechTokenizerModel {
            decoder: Decoder::init(decoder_cfg, device),
        };
        let mut store = SafetensorsStore::from_file(weights)
            .with_from_adapter(crate::CheckpointAdapter)
            .remap(remapper(decoder_cfg)?)
            .allow_partial(true);
        let result = model
            .load_from(&mut store)
            .map_err(|err| format!("failed to load {}: {err}", weights.display()))?;
        crate::check_apply_result("speech tokenizer", &result)?;
        Ok(Self {
            model,
            state: TransformerState::new(
                &decoder_cfg.transformer_config(),
                burn::tensor::DType::F32,
                device,
            ),
            config: cfg.clone(),
            device: device.clone(),
        })
    }

    pub fn output_sample_rate(&self) -> usize {
        self.config.output_sample_rate
    }

    /// Number of audio samples produced for each codec frame.
    pub fn samples_per_frame(&self) -> usize {
        self.model.decoder.total_upsample
    }

    pub fn num_code_groups(&self) -> usize {
        self.model.decoder.num_quantizers
    }

    /// Decodes `frames` frames of `num_code_groups` codes, laid out frame by frame, into audio
    /// samples.
    ///
    /// Long sequences are decoded in chunks of `chunk_size` frames with `left_context` frames of
    /// context, mirroring the reference `chunked_decode`.
    pub fn decode(&mut self, codes: &[u32], chunk_size: usize, left_context: usize) -> Vec<f32> {
        assert!(chunk_size > 0, "the chunk size must be at least one frame");
        let num_code_groups = self.num_code_groups();
        let num_frames = self.num_frames(codes);
        let mut pcm = Vec::with_capacity(num_frames * self.samples_per_frame());
        let mut start = 0;
        while start < num_frames {
            let end = usize::min(start + chunk_size, num_frames);
            let context = usize::min(left_context, start);
            let chunk = &codes[(start - context) * num_code_groups..end * num_code_groups];
            let wav = self.decode_window(chunk, context, left_context + chunk_size);
            pcm.extend_from_slice(&wav);
            start = end;
        }
        pcm
    }

    /// Decodes the frames of `codes` after its first `context` ones, which are the context
    /// they are decoded with and whose audio is dropped, through a window of `window` frames:
    /// the codes are padded on the right with copies of their last frame up to the window, so
    /// that every chunk of an utterance reaches the decoder with the same shape, the first
    /// ones with their shorter context and the last one with its few frames included. The
    /// decoder is causal, so the padding changes nothing in the samples returned. A GPU
    /// backend compiles and autotunes its kernels per shape, a couple of minutes each on a
    /// cold cache for this decoder, and one shape means one such cost for any stream.
    pub fn decode_window(&mut self, codes: &[u32], context: usize, window: usize) -> Vec<f32> {
        let num_code_groups = self.num_code_groups();
        let frames = self.num_frames(codes);
        assert!(
            context < frames,
            "a window decodes at least one frame past its context"
        );
        assert!(
            frames <= window,
            "{frames} frames do not fit a window of {window}"
        );
        let samples_per_frame = self.samples_per_frame();
        let padded;
        let codes = if frames < window {
            let tail = codes[(frames - 1) * num_code_groups..].repeat(window - frames);
            padded = [codes, tail.as_slice()].concat();
            padded.as_slice()
        } else {
            codes
        };
        let mut wav = self.decode_chunk(codes);
        wav.truncate(frames * samples_per_frame);
        wav.drain(..context * samples_per_frame);
        wav
    }

    /// Decodes every frame of `codes` in a single pass, at whatever length they come; see
    /// [`decode_window`](Self::decode_window) for the variant that keeps one shape.
    pub fn decode_chunk(&mut self, codes: &[u32]) -> Vec<f32> {
        let frames = self.num_frames(codes);
        let codes = self.codes_tensor(codes, frames);
        let wav = self.model.decoder.forward(codes, &mut self.state);
        let len = wav.dims()[2];
        wav.reshape([len])
            .into_data()
            .try_to_vec::<f32>()
            .expect("the decoder returns f32 samples")
    }

    fn num_frames(&self, codes: &[u32]) -> usize {
        let num_code_groups = self.num_code_groups();
        assert_eq!(
            codes.len() % num_code_groups,
            0,
            "expected a multiple of {num_code_groups} codes"
        );
        codes.len() / num_code_groups
    }

    /// Builds a (1, num_code_groups, frames) tensor out of frame-major codes.
    fn codes_tensor(&self, codes: &[u32], frames: usize) -> Tensor<3, Int> {
        let num_code_groups = self.num_code_groups();
        let codes: Vec<i64> = codes.iter().map(|&code| code as i64).collect();
        Tensor::<3, Int>::from_data(
            TensorData::new(codes, [1, frames, num_code_groups]),
            &self.device,
        )
        .swap_dims(1, 2)
    }
}

/// Maps the names of the checkpoint onto the module tree above.
fn remapper(cfg: &DecoderConfig) -> Result<KeyRemapper, String> {
    let mut patterns = vec![
        // The codebooks are nested one level deeper in the checkpoint.
        (
            r"\.vq\.layers\.(\d+)\._codebook\.".to_string(),
            ".codebooks.$1.".to_string(),
        ),
        // The pre-transformer holds its stack directly rather than under `model`.
        (
            r"^decoder\.pre_transformer\.(layers|norm)\.".to_string(),
            "decoder.pre_transformer.model.$1.".to_string(),
        ),
        // The upsampling stages are `nn.Sequential`s of a transposed convolution and a
        // ConvNeXt block.
        (
            r"^decoder\.upsample\.(\d+)\.0\.".to_string(),
            "decoder.upsample.$1.conv.".to_string(),
        ),
        (
            r"^decoder\.upsample\.(\d+)\.1\.".to_string(),
            "decoder.upsample.$1.convnext.".to_string(),
        ),
        // So is the vocoder: a convolution, one block per upsampling rate, an activation and a
        // final convolution.
        (
            r"^decoder\.decoder\.0\.".to_string(),
            "decoder.vocoder.pre_conv.".to_string(),
        ),
    ];
    let n = cfg.upsample_rates.len();
    for i in 1..=n {
        let prefix = format!(r"^decoder\.decoder\.{i}\.block\.");
        let block = format!("decoder.vocoder.blocks.{}.", i - 1);
        patterns.push((format!("{prefix}0\\."), format!("{block}act.")));
        patterns.push((format!("{prefix}1\\."), format!("{block}upsample.")));
        for j in 0..3 {
            patterns.push((
                format!("{prefix}{}\\.", j + 2),
                format!("{block}residual_units.{j}."),
            ));
        }
    }
    patterns.push((
        format!(r"^decoder\.decoder\.{}\.", n + 1),
        "decoder.vocoder.final_act.".to_string(),
    ));
    patterns.push((
        format!(r"^decoder\.decoder\.{}\.", n + 2),
        "decoder.vocoder.final_conv.".to_string(),
    ));
    KeyRemapper::from_patterns(patterns).map_err(|err| format!("invalid remapping: {err}"))
}

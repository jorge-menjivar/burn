//! Configuration of the Qwen3-TTS checkpoints, as shipped in their `config.json` files.

use std::collections::HashMap;

use burn::prelude::*;
use burn::tensor::activation::{gelu, silu};
use serde::Deserialize;

/// Activation of the MLP blocks. Every published checkpoint uses `silu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Silu,
    Gelu,
}

impl Activation {
    pub fn forward<const D: usize>(&self, xs: Tensor<D>) -> Tensor<D> {
        match self {
            Self::Silu => silu(xs),
            Self::Gelu => gelu(xs),
        }
    }
}

/// The small transformer predicting the codebooks `1..num_code_groups` of a frame.
#[derive(Debug, Clone, Deserialize)]
pub struct CodePredictorConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub num_code_groups: usize,
    pub hidden_act: Activation,
    #[serde(default)]
    pub attention_bias: bool,
}

/// Value of the `spk_is_dialect` entries: either `false` or the name of a dialect that
/// overrides the language tag when the language is Chinese (or auto).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Dialect {
    Flag(bool),
    Name(String),
}

#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub text_hidden_size: usize,
    pub text_vocab_size: usize,
    pub num_code_groups: usize,
    pub hidden_act: Activation,
    #[serde(default)]
    pub attention_bias: bool,
    pub code_predictor_config: CodePredictorConfig,
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_pad_id: u32,
    pub codec_think_id: u32,
    pub codec_nothink_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    #[serde(default)]
    pub codec_language_id: HashMap<String, u32>,
    #[serde(default)]
    pub spk_id: HashMap<String, u32>,
    #[serde(default)]
    pub spk_is_dialect: HashMap<String, Dialect>,
}

/// The `config.json` of a Qwen3-TTS checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub talker_config: TalkerConfig,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    pub im_start_token_id: u32,
    pub im_end_token_id: u32,
    pub assistant_token_id: u32,
    /// `base`, `custom_voice` or `voice_design`.
    #[serde(default)]
    pub tts_model_type: String,
    #[serde(default)]
    pub tts_model_size: String,
}

/// The decoder of the speech tokenizer, the `decoder_config` section of
/// `speech_tokenizer/config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub num_quantizers: usize,
    #[serde(default = "default_num_semantic_quantizers")]
    pub num_semantic_quantizers: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    pub hidden_act: Activation,
    #[serde(default)]
    pub attention_bias: bool,
}

fn default_num_semantic_quantizers() -> usize {
    1
}

impl DecoderConfig {
    /// Number of audio samples produced per codec frame.
    pub fn total_upsample(&self) -> usize {
        self.upsample_rates.iter().product::<usize>()
            * self.upsampling_ratios.iter().product::<usize>()
    }
}

/// The `speech_tokenizer/config.json` file. Only the decoder is used here, the encoder is
/// needed for voice cloning which this example does not support.
#[derive(Debug, Clone, Deserialize)]
pub struct SpeechTokenizerConfig {
    pub decoder_config: DecoderConfig,
    pub input_sample_rate: usize,
    pub output_sample_rate: usize,
    pub decode_upsample_rate: usize,
}

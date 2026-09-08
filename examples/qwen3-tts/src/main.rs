//! Text to speech with Qwen3-TTS.
//!
//! ```bash
//! cargo run --release -p qwen3-tts --features cuda -- \
//!     --text "Hello there, this is a test of text to speech with burn." --speaker ryan
//! ```

use std::error::Error;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use burn::prelude::*;
use burn::tensor::DType;
use clap::Parser;
use hf_hub::{HFClientSync, split_id};
use tokenizers::Tokenizer;

use qwen3_tts::audio::WavWriter;
use qwen3_tts::config::{Config, SpeechTokenizerConfig};
use qwen3_tts::model::{GenerationConfig, Prompt, Qwen3Tts, Voice};
use qwen3_tts::sampling::Sampling;
use qwen3_tts::speech_tokenizer::SpeechTokenizer;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "0.6b-custom-voice")]
    CustomVoice0_6B,
    #[value(name = "1.7b-custom-voice")]
    CustomVoice1_7B,
    #[value(name = "1.7b-voice-design")]
    VoiceDesign1_7B,
}

impl Which {
    fn model_id(&self) -> &'static str {
        match self {
            Self::CustomVoice0_6B => "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            Self::CustomVoice1_7B => "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            Self::VoiceDesign1_7B => "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The model variant to use.
    #[arg(long, default_value = "0.6b-custom-voice")]
    which: Which,

    /// The text to synthesize.
    #[arg(
        long,
        default_value = "Hello there, this is a test of text to speech with burn."
    )]
    text: String,

    /// The predefined speaker to use with the CustomVoice models, see --list-speakers.
    #[arg(long)]
    speaker: Option<String>,

    /// The language of the text, "auto" for automatic detection.
    #[arg(long, default_value = "auto")]
    language: String,

    /// A natural language description of the voice/style, VoiceDesign and 1.7B CustomVoice
    /// models only.
    #[arg(long)]
    instruct: Option<String>,

    /// Print the speakers and languages supported by the model and exit.
    #[arg(long)]
    list_speakers: bool,

    /// Feed the text to the talker one token per generated frame rather than putting all of it
    /// in the prefix. This is about the input text, --stream is about the output audio.
    #[arg(long)]
    streaming_text: bool,

    /// The output file using the wav format.
    #[arg(long, default_value = "out.wav")]
    out_file: String,

    /// Decode and write the audio while it is being generated rather than at the end.
    #[arg(long)]
    stream: bool,

    /// The number of frames decoded at a time when streaming, 12.5 frames per second of audio.
    #[arg(long, default_value_t = 25)]
    stream_chunk: usize,

    /// The number of preceding frames each streamed chunk is decoded with, so that the audio
    /// matches what a single decode of everything would produce. Below 50 the chunk boundaries
    /// become audible, above it there is nothing left to gain.
    #[arg(long, default_value_t = 50)]
    stream_context: usize,

    /// Use greedy decoding for both the talker and the code predictor.
    #[arg(long)]
    greedy: bool,

    /// The temperature used to sample the first codebook.
    #[arg(long, default_value_t = 0.9)]
    temperature: f32,

    /// Only sample among the top K samples for the first codebook.
    #[arg(long, default_value_t = 50)]
    top_k: usize,

    /// Nucleus sampling probability cutoff for the first codebook.
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,

    /// The temperature used to sample the other codebooks.
    #[arg(long, default_value_t = 0.9)]
    subtalker_temperature: f32,

    /// Only sample among the top K samples for the other codebooks.
    #[arg(long, default_value_t = 50)]
    subtalker_top_k: usize,

    /// Nucleus sampling probability cutoff for the other codebooks.
    #[arg(long, default_value_t = 1.0)]
    subtalker_top_p: f32,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.05)]
    repetition_penalty: f32,

    /// The maximum number of codec frames to generate (12.5 frames per second).
    #[arg(long, default_value_t = 8192)]
    max_new_tokens: usize,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Also write the generated codec tokens to this file, as a JSON array of frames. Handy to
    /// compare a run with another implementation.
    #[arg(long)]
    codes_file: Option<String>,

    /// The dtype of the talker weights: f32, bf16 or f16. Defaults to bf16 on GPU and f32 on
    /// CPU. The speech tokenizer always runs in f32.
    #[arg(long)]
    dtype: Option<String>,

    /// Compile every GPU kernel again instead of reusing the ones kept on disk by a previous run.
    #[arg(long)]
    no_kernel_cache: bool,

    /// Run the code predictor as plain operations instead of replaying the graphs captured from
    /// its forward passes.
    #[arg(long)]
    eager_code_predictor: bool,

    /// Run the talker's decode steps as plain operations instead of replaying the graphs
    /// captured from them.
    #[arg(long)]
    eager_talker: bool,

    /// Sample the tokens with tensor operations instead of the sampling kernel.
    #[arg(long)]
    sampling_ops: bool,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    /// The tokenizer.json file, defaults to the one of Qwen/Qwen3-0.6B which shares the
    /// vocabulary of the TTS models.
    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    config: Option<String>,

    #[arg(long)]
    weights: Option<String>,

    #[arg(long)]
    speech_tokenizer_config: Option<String>,

    #[arg(long)]
    speech_tokenizer_weights: Option<String>,
}

/// The device the model runs on, and whether it is a GPU.
fn select_device(cpu: bool) -> (Device, bool) {
    if !cpu {
        #[cfg(feature = "cuda")]
        return (Device::cuda(0), true);
        #[cfg(all(not(feature = "cuda"), feature = "rocm"))]
        return (Device::rocm(0), true);
        #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "metal"))]
        return (Device::metal(DeviceKind::DefaultDevice), true);
        #[cfg(all(
            not(feature = "cuda"),
            not(feature = "rocm"),
            not(feature = "metal"),
            feature = "vulkan"
        ))]
        return (Device::vulkan(DeviceKind::DefaultDevice), true);
        #[cfg(all(
            not(feature = "cuda"),
            not(feature = "rocm"),
            not(feature = "metal"),
            not(feature = "vulkan"),
            feature = "wgpu"
        ))]
        return (Device::wgpu(DeviceKind::DefaultDevice), true);
    }
    #[cfg(feature = "cpu")]
    return (Device::cpu(), false);
    #[cfg(all(not(feature = "cpu"), feature = "flex"))]
    return (Device::flex(), false);
    #[cfg(all(not(feature = "cpu"), not(feature = "flex")))]
    (Device::default(), false)
}

/// Downloads `filename` from the hub unless it is already in the cache.
fn hub_file(
    client: &HFClientSync,
    model_id: &str,
    revision: &str,
    filename: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    let (owner, name) = split_id(model_id);
    let repo = client.model(owner, name);
    let cached = repo
        .download_file()
        .filename(filename)
        .revision(revision)
        .local_files_only(true)
        .send();
    match cached {
        Ok(path) => Ok(path),
        Err(_) => {
            eprintln!("downloading {model_id}/{filename}");
            Ok(repo
                .download_file()
                .filename(filename)
                .revision(revision)
                .send()?)
        }
    }
}

/// Decodes the generated frames as they come and appends the audio to the output file.
///
/// Each chunk is decoded together with the frames that precede it, whose audio is then
/// discarded, so that the result matches decoding everything at once.
struct Streamer<'a> {
    speech_tokenizer: &'a mut SpeechTokenizer,
    writer: WavWriter,
    codes: Vec<u32>,
    num_code_groups: usize,
    /// Frames whose audio was written already.
    written: usize,
    chunk: usize,
    left_context: usize,
    samples_per_frame: usize,
    first_audio: Option<Duration>,
    start: Instant,
}

impl<'a> Streamer<'a> {
    fn new(
        speech_tokenizer: &'a mut SpeechTokenizer,
        out_file: &str,
        chunk: usize,
        left_context: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let sample_rate = speech_tokenizer.output_sample_rate() as u32;
        let samples_per_frame = speech_tokenizer.samples_per_frame();
        let num_code_groups = speech_tokenizer.num_code_groups();
        Ok(Self {
            speech_tokenizer,
            writer: WavWriter::new(out_file, sample_rate)?,
            codes: Vec::new(),
            num_code_groups,
            written: 0,
            chunk,
            left_context,
            samples_per_frame,
            first_audio: None,
            start: Instant::now(),
        })
    }

    fn push(&mut self, frame: &[u32]) -> Result<(), Box<dyn Error>> {
        self.codes.extend_from_slice(frame);
        let pending = self.codes.len() / self.num_code_groups - self.written;
        if pending >= self.chunk {
            self.decode_pending()?
        }
        Ok(())
    }

    /// Decodes everything that has been generated since the last chunk.
    fn decode_pending(&mut self) -> Result<(), Box<dyn Error>> {
        let frames = self.codes.len() / self.num_code_groups;
        if frames <= self.written {
            return Ok(());
        }
        let context = usize::min(self.left_context, self.written);
        let start = (self.written - context) * self.num_code_groups;
        let pcm = self
            .speech_tokenizer
            .decode_chunk(&self.codes[start..frames * self.num_code_groups]);
        let skip = context * self.samples_per_frame;
        self.writer.write(&pcm[skip..])?;
        self.written = frames;
        if self.first_audio.is_none() {
            self.first_audio = Some(self.start.elapsed())
        }
        Ok(())
    }

    fn finish(mut self) -> Result<(usize, Option<Duration>), Box<dyn Error>> {
        self.decode_pending()?;
        let samples = self.writer.samples();
        let first_audio = self.first_audio;
        self.writer.finish()?;
        Ok((samples, first_audio))
    }
}

/// Compiles every GPU kernel again instead of reusing the ones CubeCL kept on disk.
///
/// CubeCL caches the kernels it compiles at runtime next to its autotune results, in
/// `target/environment` when running from a cargo workspace and in the user cache directory
/// otherwise. This example needs a few hundred of them, worth about 20 seconds of compilation
/// spread over the first frames of the generation and the first speech decoding, so the cache is
/// what makes every run of a build but the first fast.
///
/// Must run before the first device is created, because the configuration is frozen the first
/// time something reads it. A `cubecl.toml` or `burn.toml` up the directory tree and the
/// `CUBECL_*` environment variables are honored for everything else.
fn disable_kernel_cache() {
    use burn::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    config.compilation.cache = false;
    // `false` means the configuration was already read, and set by whoever did so.
    let _ = CubeClRuntimeConfig::try_set(config);
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.no_kernel_cache {
        disable_kernel_cache();
    }

    let start = Instant::now();
    let client = HFClientSync::new()?;
    let model_id = args
        .model_id
        .clone()
        .unwrap_or_else(|| args.which.model_id().to_string());
    let path = |arg: &Option<String>, name: &str| -> Result<PathBuf, Box<dyn Error>> {
        match arg {
            Some(path) => Ok(PathBuf::from(path)),
            None => hub_file(&client, &model_id, &args.revision, name),
        }
    };
    let config_file = path(&args.config, "config.json")?;
    let weights_file = path(&args.weights, "model.safetensors")?;
    let st_config_file = path(
        &args.speech_tokenizer_config,
        "speech_tokenizer/config.json",
    )?;
    let st_weights_file = path(
        &args.speech_tokenizer_weights,
        "speech_tokenizer/model.safetensors",
    )?;
    let tokenizer_file = match &args.tokenizer {
        Some(file) => PathBuf::from(file),
        None => hub_file(&client, "Qwen/Qwen3-0.6B", "main", "tokenizer.json")?,
    };
    println!("retrieved the files in {:?}", start.elapsed());

    let start = Instant::now();
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(|err| err.to_string())?;
    let config: Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
    let st_config: SpeechTokenizerConfig = serde_json::from_slice(&std::fs::read(st_config_file)?)?;
    if config.tts_model_type == "base" {
        return Err(
            "the Base checkpoints clone a voice from a recording, which needs the speaker \
             encoder and the codec encoder: neither is part of this example, use a CustomVoice \
             or a VoiceDesign model"
                .into(),
        );
    }
    let (device, is_gpu) = select_device(args.cpu);
    let dtype = match args.dtype.as_deref() {
        Some("f32") => DType::F32,
        Some("bf16") => DType::BF16,
        Some("f16") => DType::F16,
        Some(dtype) => return Err(format!("unsupported dtype {dtype}").into()),
        None if is_gpu => DType::BF16,
        None => DType::F32,
    };
    println!("running on {device:?} with the talker in {dtype:?}");
    let mut model = Qwen3Tts::load(&config, &weights_file, dtype, &device)?;
    if !args.eager_code_predictor {
        model.enable_code_predictor_graphs();
    }
    if !args.eager_talker {
        model.enable_talker_graphs();
    }
    if args.sampling_ops {
        qwen3_tts::sampling::use_kernel(false);
    }
    // The speech tokenizer always runs in f32, as in the reference implementation.
    let mut speech_tokenizer = SpeechTokenizer::load(&st_config, &st_weights_file, &device)?;
    println!("loaded the models in {:?}", start.elapsed());

    let speakers = model.supported_speakers();
    if args.list_speakers {
        println!("model type: {}", config.tts_model_type);
        println!("speakers: {speakers:?}");
        println!("languages: {:?}", model.supported_languages());
        return Ok(());
    }

    let speaker = match &args.speaker {
        Some(speaker) => Some(speaker.clone()),
        None if config.tts_model_type == "custom_voice" => {
            let speaker = if speakers.contains(&"ryan") {
                "ryan"
            } else {
                speakers.first().copied().unwrap_or_default()
            };
            println!("no speaker specified, using {speaker:?} (see --list-speakers)");
            Some(speaker.to_string())
        }
        None => None,
    };
    let voice = match &speaker {
        Some(speaker) => Voice::Speaker(speaker),
        None => Voice::None,
    };

    let encode = |text: &str| -> Result<Vec<u32>, Box<dyn Error>> {
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|err| err.to_string())?;
        Ok(encoding.get_ids().to_vec())
    };
    let input_ids = encode(&format!(
        "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
        args.text
    ))?;
    let instruct = args.instruct.as_deref().filter(|s| !s.is_empty());
    let instruct = match instruct {
        Some(_) if config.tts_model_size == "0b6" => {
            println!("--instruct is not supported by the 0.6B models, ignoring it");
            None
        }
        instruct => instruct,
    };
    let instruct_ids = match instruct {
        Some(instruct) => Some(encode(&format!(
            "<|im_start|>user\n{instruct}<|im_end|>\n"
        ))?),
        None => None,
    };
    let prompt = Prompt {
        input_ids: &input_ids,
        instruct_ids: instruct_ids.as_deref(),
        language: Some(&args.language),
        voice,
        non_streaming_mode: !args.streaming_text,
    };

    // Degenerate sampling parameters are turned into greedy decoding rather than being
    // rejected by the sampler in the middle of the generation.
    let sampling = |k: usize, p: f32, temperature: f32| {
        if args.greedy || temperature <= 0. {
            Sampling::ArgMax
        } else if k == 0 {
            Sampling::TopP { p, temperature }
        } else {
            Sampling::TopKThenTopP { k, p, temperature }
        }
    };
    let generation = GenerationConfig {
        max_new_tokens: args.max_new_tokens,
        sampling: sampling(args.top_k, args.top_p, args.temperature),
        subtalker_sampling: sampling(
            args.subtalker_top_k,
            args.subtalker_top_p,
            args.subtalker_temperature,
        ),
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
    };

    let sample_rate = speech_tokenizer.output_sample_rate() as u32;
    let mut streamer = if args.stream {
        if args.stream_chunk == 0 {
            return Err("--stream-chunk should be at least one frame".into());
        }
        println!("streaming the audio to {}", args.out_file);
        Some(Streamer::new(
            &mut speech_tokenizer,
            &args.out_file,
            args.stream_chunk,
            args.stream_context,
        )?)
    } else {
        None
    };

    let start = Instant::now();
    let mut num_frames = 0;
    let mut stream_error = None;
    let frames = model.generate_with_callback(&prompt, &generation, |frame| {
        num_frames += 1;
        print!(
            "\rgenerated {num_frames} frames ({:.1}s)",
            num_frames as f64 / 12.5
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();
        if let (Some(streamer), None) = (streamer.as_mut(), &stream_error)
            && let Err(err) = streamer.push(frame)
        {
            stream_error = Some(err.to_string())
        }
    })?;
    println!();
    if let Some(err) = stream_error {
        return Err(err.into());
    }
    for (what, hardware) in [
        (
            "the code predictor's passes",
            model.code_predictor_graph_is_hardware(),
        ),
        ("the talker's steps", model.talker_graph_is_hardware()),
    ] {
        match hardware {
            Some(true) => println!("{what} replayed graphs"),
            Some(false) => println!("this device does not replay graphs, {what} ran eagerly"),
            None => (),
        }
    }
    let elapsed = start.elapsed();
    println!(
        "generated {} frames in {:.1}s ({:.1} frames/s)",
        frames.len(),
        elapsed.as_secs_f64(),
        frames.len() as f64 / elapsed.as_secs_f64()
    );
    if frames.is_empty() {
        return Err("no audio frames were generated".into());
    }

    if let Some(codes_file) = &args.codes_file {
        std::fs::write(codes_file, serde_json::to_string(&frames)?)?;
        println!("wrote the codec tokens to {codes_file}");
    }

    if let Some(streamer) = streamer {
        let (samples, first_audio) = streamer.finish()?;
        if let Some(first_audio) = first_audio {
            println!("first audio chunk after {first_audio:?}");
        }
        println!(
            "wrote {:.2}s of audio to {}",
            samples as f64 / sample_rate as f64,
            args.out_file
        );
        return Ok(());
    }

    let start = Instant::now();
    let codes: Vec<u32> = frames.into_iter().flatten().collect();
    let pcm = speech_tokenizer.decode(&codes, 300, 25);
    println!(
        "decoded {:.2}s of audio in {:?}",
        pcm.len() as f64 / sample_rate as f64,
        start.elapsed()
    );
    println!("writing output file {}", args.out_file);
    let mut writer = WavWriter::new(&args.out_file, sample_rate)?;
    writer.write(&pcm)?;
    writer.finish()?;
    Ok(())
}

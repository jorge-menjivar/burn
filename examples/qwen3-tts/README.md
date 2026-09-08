# Qwen3-TTS

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) is a text-to-speech model from the Qwen team.
The talker, a Qwen3 transformer, generates 12.5 Hz frames of 16 codec tokens, and the decoder of
the speech tokenizer turns them into 24 kHz audio: every frame yields 1920 samples.

Each frame is produced in two steps:

- the *talker* is fed with the sum of the text and codec embeddings and predicts the first
  codebook entry of the next frame,
- the *code predictor*, a small Qwen3 transformer conditioned on the talker hidden state,
  autoregressively predicts the 15 remaining entries of that frame.

## Usage

The weights and the tokenizer are fetched from the Hugging Face hub on the first run
(around 2.5 GB for the 0.6B model and its speech tokenizer).

```bash
cargo run --release -p qwen3-tts --features cuda -- \
    --speaker ryan --language english \
    --text "The quick brown fox jumps over the lazy dog."
```

The result is written to `out.wav`. Nothing is played back: use e.g. `ffplay -autoexit -nodisp
out.wav` to listen to it.

With `--stream` the audio is decoded and appended to the file while it is generated, rather than
in one pass at the end. Chunks are decoded together with the `--stream-context` frames that
precede them, whose audio is then dropped, so that the result matches a single decode of
everything, and through a window of `--stream-context` plus `--stream-chunk` frames every time,
padded on the right, so that the decoder meets one shape (see the Performance section).

```bash
cargo run --release -p qwen3-tts --features cuda -- --speaker ryan --stream \
    --text "Streaming decodes the audio while it is being generated."
```

Pick the backend with a feature: `cuda`, `rocm`, `metal`, `vulkan`, `wgpu`, `cpu`, or the default
`flex` (pure Rust CPU). The talker runs in bf16 on a GPU and in f32 otherwise, overridden with
`--dtype`; the speech tokenizer always runs in f32.

## Models

| `--which`            | Model                                    |
|----------------------|------------------------------------------|
| `0.6b-custom-voice`  | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`   |
| `1.7b-custom-voice`  | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`   |
| `1.7b-voice-design`  | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`   |

The CustomVoice checkpoints ship a set of speakers, listed together with the supported languages
by `--list-speakers`. Two of them, `eric` and `dylan`, are dialect speakers: they switch the
language tag to Sichuanese and to the Beijing dialect.

```bash
cargo run --release -p qwen3-tts --features cuda -- --list-speakers
```

The VoiceDesign checkpoint has no predefined speaker, the voice is described in natural language
instead. The 1.7B CustomVoice checkpoint accepts the same instruction on top of its speaker.

```bash
cargo run --release -p qwen3-tts --features cuda -- --which 1.7b-voice-design \
    --instruct "A calm elderly storyteller with a warm, low voice." \
    --text "Once upon a time, in a village at the edge of a forest."
```

The `Base` checkpoints clone a voice from a recording. They need the speaker encoder and the
encoder of the speech tokenizer, which are not part of this example.

## Performance

Measured on an RTX 3090, the 0.6B talker generates 72 to 79 frames per second in bf16 once its
kernels are compiled, and 48 to 67 when the audio is decoded along the way with `--stream`, which
pauses the generation while each chunk is decoded: six times the 12.5 frames per second of
real time. The candle port of the same model runs at around 58 frames per second on the same
GPU; the rest of this section is about how the gap was closed and then passed.

### Kernel compilation and autotuning

CubeCL compiles every kernel it meets at runtime, and picks the implementation of its matmuls,
convolutions, reductions and attention by compiling and timing every candidate it has, once per
shape it has not met. A run of this example meets a few hundred kernels and, for its first
prompt, 130 such decisions, and their candidates are what a cold start costs: on a host that
has never run the example, the first frame comes out after three minutes and the first chunk
of audio after six, for a 300-character prompt with `--stream` (measured with the driver's
cache disabled, `CUDA_CACHE_DISABLE=1`; the generation runs at full speed in between). No
candidate dominates: the time is the driver generating machine code for a hundred-odd kernels,
a few seconds each for the double-buffered matmuls of the talker and for the speech decoder,
whose window of frames goes through some thirty convolutions and matmuls, each with a decision
of its own, at the first chunk.

It took eight minutes before two changes to the autotune lists of `burn-cubecl` on this branch.
The register-tile matmul candidates, which compiled for 16 to 24 seconds each at the tiny
shapes of the code predictor's attention and never won on a tensor-core device, now sit behind
the accelerated kernels there. And a flash attention candidate that cannot launch a shape now
loses outright, where it used to run the fallback under its own name and pay the fallback's
matmul autotunes as its own compile time. On this model no flash attention kernel launches at
all: their stage must divide the number of queries, which a prompt's token count rarely is a
multiple of, and every attention runs the fallback, whose score matrix is small here.

Two caches keep the result. CubeCL stores the compiled kernels with the autotune results, in
`target/environment/default.db` when running from a cargo workspace and in the user cache
directory otherwise (`~/.cache/cubecl` on Linux), and the driver keeps the machine code it
generates from them in `~/.nv/ComputeCache`. Rebuilding the binary loses the first and keeps
the second, and the autotune results with it. Counted from the launch of the process:

| host | first frame | first audio | 300 frames done |
|---|---|---|---|
| never ran the example | 3 min | 6 min | 6 min |
| tuned, CubeCL's database deleted, driver cache kept | 66 s | 157 s | 162 s |
| tuned, binary rebuilt | 9 s | 16 s | 21 s |
| tuned, same binary | 2.5 s | 3 s | 7 s |

`--no-kernel-cache` compiles the kernels on every run, as does a `cubecl.toml` with
`[compilation] cache = false` in the working directory or one of its parents, and
`CUBECL_ENVIRONMENT=<name>` tunes again into another database next to the default one.

### One shape per request, and what a warm-up covers

The decisions are kept per shape, rounded up to a power of two, so a shape the host has not met
tunes again, before the first frame of the request that brings it or in the middle of it. What
varies from one request to the next:

- The prompt, through the prefill: its text tokens and its whole prefix, each rounded up to a
  power of two. On a tuned host the first prompt of a size class costs 3 to 10 seconds before
  the first frame, and 25 seconds at 512 tokens (a 1200-character prompt), where the attention
  fallback tunes two large matmuls of its own; prompts of the same class then cost nothing
  more.
- Nothing else. The talker's decode step has one shape whatever the capacity of its cache: the
  1200-character prompt, whose cache holds 1024 positions rather than 512, tuned nothing but
  its prefill, and growing the cache during a generation captures the step again but tunes
  nothing. The code predictor's passes are the same for every frame. And the speech decoder
  reads one window per stream, `--stream-context` plus `--stream-chunk` frames, padded on the
  right with copies of the last frame, which the causal decoder keeps out of the samples
  returned (`SpeechTokenizer::decode_window`; padding with zeros instead gives the same samples,
  bit for bit). Without the window, the first two chunks of a stream, decoded with less
  context, and the last one, with whatever frames remained, had shapes of their own, tuned at
  frames 50 and 75 and at the end of the first utterance of each length: a host warmed up on
  25-frame generations then produced a ten-second utterance at 2.6 frames per second, and the
  cold start above took ten minutes instead of eight.

A warm-up therefore covers a deployment with one prompt per size class of the prompts it will
serve, however short the speech: the first frame meets every shape of the code predictor and of
the talker's step, and the first chunk, padded to the window, every shape of the decoder.

### Where the time goes

Generating one frame runs 103 transformer layer passes on a single token: 28 for the talker,
then 5 for each of the 15 codebooks the code predictor adds. Each pass is about 35 kernel
launches: 9 matmuls, the two layer norms, the two head norms with the rotary embedding folded
in, the key/value cache update, the attention and its softmax, and a few casts. That is about
3500 launches per frame, and traced with CUPTI they keep the GPU busy for 14 ms. The candle port
launches about as many kernels per frame, 3100, which take 10.5 ms, and its frame takes 17 ms.
Run as plain operations, this port's frame took 55 ms, so the GPU idled three quarters of the
time: an operation in candle is a function that allocates and launches on the calling thread,
while here every operation (13,000 per frame, counting the reshapes and the drops) goes to
Burn's fusion thread, which turns it into a plan and hands every launch to the CUDA thread. The
fusion thread was busy 94% of the time at about 5.5 µs per operation, the main thread waited on
it 77% of the time, and the CUDA thread and the GPU waited on the fusion thread in turn. Two
costs on that path were removed locally. The CUDA thread merged the free slices of every memory
page before every reservation, a walk that grew with the key/value cache and cost it half its
time; it now merges only when a reservation fails. And each thread only took work from the next
in full batches of 32, so the three alternated between stalling and idling instead of
overlapping; a thread now takes what is queued as soon as its producer pauses.

Most of that runtime is gone: the example captures the model's forward passes as CUDA graphs,
with `burn::tensor::capture`, and replays them, one driver call per pass instead of a few
hundred operations through the fusion thread. A graph replays against the exact buffers it was
recorded with, which shapes the code around it. The key/value caches are preallocated and
written in place at the position of the pass, instead of growing by concatenation, and the
inputs of a pass are written in place into fixed buffers before each replay. The code predictor
is the simple case: its fifteen passes per frame are the two-token first pass and single-token
passes at positions 2 to 15, always the same, so they are fifteen graphs over a cache of
sixteen positions. The talker's pass is the same computation at a different position every
frame, with a cache that grows across frames, so it is one graph that does not read its
position: the position is a one-element tensor, the rotary rows are gathered by it, the token's
keys and values are written where it says, and the attention covers the whole cache through a
mask that hides the positions after it. The cache starts at a few hundred positions past the
prompt and doubles, which moves it and captures the pass again, when a generation outgrows it.
Reading the cache through a prefix instead, one graph per prefix length, turned out worse:
Burn copies such a prefix rather than viewing it, and every graph pinned its copies.

Two things bit on the way. Each pass runs once before it is captured, because the capture pins
every buffer its warm-up runs allocate, and the first run of a pass autotunes its matmuls on
dozens of candidates: captured cold, the code predictor's passes pinned 14 GB and the audio
decoder then failed to allocate. And a fused reduce whose reference tensor is one of its
outputs, which happens when the zero-filled input buffer of a pass gets fused into the first
norm, resolved that reference against its inputs and crashed; fixed locally in Burn. The code
predictor's codes are identical to the eager path's, and in f32 with greedy decoding to the
transformers reference; the talker's graph attends over a longer, masked row, which rounds
differently and flips a near-tie in the code predictor after a few frames, as any change of
kernels does. `--eager-code-predictor` and `--eager-talker` run either as plain operations.

With the passes captured, a frame took 20 ms for 14 ms of GPU time, and what kept it from the
GPU-bound figure was the sampling: every codebook read its logits back to the host, and the
sixteen reads per frame each drained the pipeline and waited for the GPU, 11 ms in total, with
the 0.3 ms launch of the next graph only starting after. So the sampling moved to the device,
and into the graphs: each pass ends by drawing its code and writing the embedding the next pass
reads, the talker's step by drawing the first code of the next frame and writing the input of
the code predictor, and the host reads the sixteen codes of a frame once, after the last pass.
The draw is the Gumbel-max trick, the largest of the kept logits plus Gumbel noise, from a
buffer of uniform noise refreshed by the device's generator every frame. The repetition penalty
reads a mask of the first codes generated so far, which the talker's step also updates, and the
end-of-speech suppression is a bias tensor the host rewrites twice per generation. In f32 with
greedy decoding the codes are still identical to the reference.

The draw was first written as tensor operations: the device has no sort and the vocabularies
are small, so the top-k and top-p filters compared every entry with every other one, a few
million comparisons in an elementwise kernel and a reduction over a vocabulary-squared matrix,
about 40 µs per draw and eight kernels. It is now one kernel of 11 µs, in `sampling_kernel.rs`:
a single cube loads the logits into shared memory, finds the top-k threshold by a radix select
over the histogram of their ordered bits, four rounds of eight bits with the histogram scanned
by one plane, finds the nucleus threshold by the same select over the probability mass, and
takes the largest of the kept logits plus the noise. The kernel reaches the tensors through a
backend extension implemented for the CubeCL backend, which launches it, and for the fusion
backend, which registers the draw as a custom operation of its stream, so it sits among the
fused operations and gets captured with them. Half a millisecond per frame, 5% in paired
runs, which is how to see it: the rates in this section move by 10% with the GPU's temperature
and with what else is using it, and the autotune picks matter as much, a fresh tune on a busy
desktop landed 10% below the cached ones, so keep the `target/environment` database around.
`--sampling-ops` keeps the tensor operations for comparison, and they remain the path of the
backends the kernel is not written for.

A frame now takes 13 to 14 ms: 13 ms of GPU time, 3160 kernels launched by sixteen graph
replays and five plain launches, 45 driver calls where there were 13,000, and the GPU busy about
90% of the time; the one read per frame waits for the GPU, which is where the time goes. The
frame is GPU-bound and the kernels below are what is left. The process uses 3.7 GB of device
memory with the default token limit.

The single-token forward is written to fuse. The rotary embedding is a gather along the channels
and arithmetic that Burn folds into the surrounding kernels, rather than two narrows and a
concatenation per tensor, which cost six kernels each. The attention of one query over the cache
does not go through the attention op, which has no kernel for a single query and falls back to
a dozen small ones on top of the copies expanding the keys and values to every head: grouping
the query heads by their key/value head turns it into two batched matmuls and a softmax, and the
grouping is a reshape. In f32 with greedy decoding the codes match the transformers reference
exactly over the 15 frames of the test prompt.

The matmuls are half of the GPU time. Multiplying a single row by a `[1024, 3072]` bf16 weight is
bound by reading the weight once, about 7µs on this GPU, and cuBLAS's gemv does it in 6.8µs. The
kernels Burn's autotune has for it are several times slower when the weight is stored row-major,
the default layout of `burn::nn::Linear`, than when it is stored column-major. This example uses
`LinearLayout::Col`, which is also the layout of the checkpoints so the weights load without a
transpose. Most linear layers are directly followed by an elementwise op, and the matmul fused
with it used to fall back to a slow kernel whatever the layout, because the fast kernel writes
its output one element at a time and the fusion insisted on a vectorized output. The fused
matmul can now run that kernel, in 7.6µs.

The norms came next. A norm is a reduce followed by a broadcast, which Burn fuses into one
kernel, and that kernel reduced its whole row on a single thread and then walked the row again
for the epilogue: 14 µs for a thousand elements where candle's norm takes 2.4 µs, 276 times a
frame, 4 ms of the 15. Fixed locally in Burn: the fused kernel now takes the plane and cube
routines the plain fused reduce already had, so 32 or 256 threads reduce a row and stride over
it for the epilogue. The catch is that the values passed from the reduce to the epilogue, the
reduced sum and what the write block derives from it, are per-thread registers that only the
writing thread held, which came out as wrong codes in every codebook of the first frame; they
are now broadcast from that thread after each reduce, over the plane or through shared memory.
The autotuner picks the cube here: 4.3 µs, 1.4 ms per frame, 13 ms of GPU time per frame
instead of 15.6, and 17% more frames per second in paired runs against the previous build.
What is left on the GPU side: the softmax runs as six kernels because the fusion of a reduce
followed by a broadcast stops at the second reduce, a norm whose squaring was absorbed by the
matmul before it keeps its mean unfused, and the fused matmul's 8 µs against cuBLAS's 6.8.

## Comparing with another implementation

`--codes-file` writes the generated codec tokens as a JSON array of frames. With `--greedy` and
`--dtype f32` the run is deterministic, which is what makes it comparable against the reference
implementation: the same text, speaker and language give the same codes frame by frame until the
first near-tie in the code predictor is decided differently by the two sets of kernels. Note that
Burn's accelerated matmul kernels read f32 inputs as TF32 on tensor cores, so `--dtype f32` is
only exact where the plain kernels are picked, as they are for the single-row products of the
column-major linear layers.

## Sampling

`--temperature`, `--top-k` and `--top-p` control the sampling of the first codebook, their
`--subtalker-` counterparts the sampling of the other fifteen, and `--greedy` turns both into
greedy decoding. `--seed` makes a run reproducible: it seeds the device's random number
generator, which the sampling draws its noise from. The sampling runs on the device, as one
kernel on the GPU backends or as tensor operations with `--sampling-ops`, see the performance
section.

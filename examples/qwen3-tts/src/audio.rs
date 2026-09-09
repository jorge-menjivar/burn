//! Writing the generated samples to a wav file, incrementally when streaming, and reading the
//! reference recording of voice cloning.

use std::io::BufWriter;
use std::path::Path;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter as HoundWriter};

/// Reads a wav file into mono samples in `[-1, 1]`, returned with the sample rate. The channels
/// of a multi-channel file are averaged.
pub fn read_wav(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32), hound::Error> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels).max(1);
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        SampleFormat::Int => {
            let scale = 1. / (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|sample| sample as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono = samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok((mono, spec.sample_rate))
}

/// Windowed-sinc resampling of a mono signal.
pub fn resample(input: &[f32], sr_in: u32, sr_out: u32) -> Vec<f32> {
    if sr_in == sr_out || sr_in == 0 || sr_out == 0 {
        return input.to_vec();
    }
    let ratio = sr_out as f64 / sr_in as f64;
    let out_len = (input.len() as f64 * ratio).round() as usize;
    // Low-pass at the smaller of the two Nyquist frequencies, relative to the input rate.
    let cutoff = ratio.min(1.) * 0.95;
    let taps = 32i64;
    let len = input.len() as i64;
    let mut output = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 / ratio;
        let center = pos.floor() as i64;
        let mut acc = 0f64;
        let mut norm = 0f64;
        for t in -taps..=taps {
            let idx = center + t;
            if idx < 0 || idx >= len {
                continue;
            }
            let x = idx as f64 - pos;
            let sinc = if x == 0. {
                1.
            } else {
                let a = std::f64::consts::PI * cutoff * x;
                a.sin() / a
            };
            let window = 0.5 * (1. + (std::f64::consts::PI * x / (taps as f64 + 1.)).cos());
            let w = sinc * window;
            acc += input[idx as usize] as f64 * w;
            norm += w;
        }
        output.push(if norm.abs() > 1e-12 {
            (acc / norm) as f32
        } else {
            0.
        });
    }
    output
}

/// A 16 bit mono wav writer. The header is fixed up when [`WavWriter::finish`] is called, so a
/// file that is still being written has a zero length in its header: players that follow a
/// growing file handle it, the others need the file to be finished first.
pub struct WavWriter {
    writer: HoundWriter<BufWriter<std::fs::File>>,
    samples: usize,
}

impl WavWriter {
    pub fn new(path: impl AsRef<Path>, sample_rate: u32) -> Result<Self, hound::Error> {
        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        Ok(Self {
            writer: HoundWriter::create(path, spec)?,
            samples: 0,
        })
    }

    pub fn write(&mut self, pcm: &[f32]) -> Result<(), hound::Error> {
        for &sample in pcm {
            let sample = (sample.clamp(-1., 1.) * i16::MAX as f32) as i16;
            self.writer.write_sample(sample)?;
        }
        self.samples += pcm.len();
        // Flush so that a player reading the file while it grows sees the new samples.
        self.writer.flush()
    }

    pub fn samples(&self) -> usize {
        self.samples
    }

    pub fn finish(self) -> Result<(), hound::Error> {
        self.writer.finalize()
    }
}

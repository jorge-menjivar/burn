//! Writing the generated samples to a wav file, incrementally when streaming.

use std::io::BufWriter;
use std::path::Path;

use hound::{SampleFormat, WavSpec, WavWriter as HoundWriter};

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

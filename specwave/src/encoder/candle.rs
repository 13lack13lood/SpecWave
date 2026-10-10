//! The encoder on candle, for sentence-transformers models stored as Hugging Face snapshot directories.
//!
//! A snapshot holds `config.json`, `model.safetensors` and `tokenizer.json`, plus the sentence-transformers settings
//! in `modules.json`, `sentence_bert_config.json`, `config_sentence_transformers.json` and the pooling folder that
//! `modules.json` names, eg. `1_Pooling/config.json`. The query prompt and the pooling come from those settings. For
//! arctic-embed-m that is the state of the classification (CLS) token, with the prefix "Represent this sentence for
//! searching relevant passages: " in front of each query. For all-MiniLM-L6-v2 it is the average over the tokens,
//! with no prefix.
//!
//! The encoder takes one text at a time, so there is no padding and the average over the tokens is the masked
//! average. Each layer's summary, concat(average, first token), is computed on the device, and only the summaries the
//! hook wants are copied back.

use std::collections::HashMap;
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use tokenizers::{PostProcessor, Tokenizer, TruncationParams};

use super::bert::{BertModel, Config};
use super::{LayerEncoder, LayerHook, NoHook};

/// Where the model runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// The processor (CPU). On macOS its matrix products go through Accelerate.
    Cpu,
    /// The graphics processor (GPU) through Metal, on macOS only.
    Metal,
}

/// How the last layer's token states become the embedding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    /// The first token's state, the classification (CLS) token of BERT-style (Bidirectional Encoder Representations
    /// from Transformers) models.
    Cls,
    /// The average over the tokens.
    Mean,
}

/// A sentence-transformers BERT model on candle, with its tokenizer, query prompt and pooling.
pub struct CandleEncoder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    prompt: String,
    pooling: Pooling,
    dim: usize,
    layers: usize,
    /// The model's positions, the most tokens it can read.
    positions: usize,
}

impl CandleEncoder {
    /// Loads the model in `snapshot`, a Hugging Face snapshot directory, onto `backend`. A missing file keeps its
    /// `NotFound`, a damaged one gives `InvalidData`, and settings this encoder can't follow, eg. an extra dense layer
    /// after the pooling, give `Unsupported`, as does Metal where it isn't available.
    pub fn open(snapshot: &Path, backend: Backend) -> Result<CandleEncoder, Error> {
        let config: Config = read_json(&snapshot.join("config.json"))?;
        let settings = Settings::read(snapshot)?;

        if config.num_hidden_layers == 0
            || config.num_attention_heads == 0
            || !config.hidden_size.is_multiple_of(config.num_attention_heads)
        {
            return Err(damaged("config.json has a shape no BERT model has"));
        }

        let device = match backend {
            Backend::Cpu => Device::Cpu,
            Backend::Metal => Device::new_metal(0).map_err(|e| Error::new(ErrorKind::Unsupported, e.to_string()))?,
        };
        let max_len = settings.max_len.unwrap_or(config.max_position_embeddings);
        let tokenizer = read_tokenizer(
            &snapshot.join("tokenizer.json"),
            max_len.min(config.max_position_embeddings),
        )?;

        // An id past the model's vocabulary fails on the processor, but Metal's lookup reads the last row instead. So
        // every token the tokenizer knows, its added tokens too, needs a row, even where the ids leave gaps.
        let largest = tokenizer.get_vocab(true).into_values().max();

        if largest.is_some_and(|id| id as usize >= config.vocab_size) {
            return Err(damaged("tokenizer.json has a token id past config.json's vocabulary"));
        }

        let weights = std::fs::read(snapshot.join("model.safetensors"))?;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device).map_err(damaged)?;
        let model = BertModel::load(vb, &config).map_err(damaged)?;
        Ok(CandleEncoder {
            model,
            tokenizer,
            device,
            prompt: settings.prompt,
            pooling: settings.pooling,
            dim: config.hidden_size,
            layers: config.num_hidden_layers,
            positions: config.max_position_embeddings,
        })
    }

    /// Cuts every text at `max` tokens, counting the prompt and the special tokens, or at the model's last position if
    /// that comes first. Until this is called the snapshot's own limit holds, eg. 512 tokens for arctic-embed-m. A
    /// limit that leaves no room for the text gives `InvalidInput`.
    pub fn set_max_tokens(&mut self, max: usize) -> Result<(), Error> {
        let max = max.min(self.positions);

        if max <= special_tokens(&self.tokenizer) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "the limit leaves no room for the text",
            ));
        }

        truncate(&mut self.tokenizer, max).map_err(|e| Error::new(ErrorKind::InvalidInput, e.to_string()))
    }

    /// The most tokens a text is cut to, counting the prompt and the special tokens.
    pub fn max_tokens(&self) -> usize {
        self.tokenizer.get_truncation().map_or(self.positions, |t| t.max_length)
    }

    /// The text put in front of every query, eg. "Represent this sentence for searching relevant passages: ".
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    pub fn pooling(&self) -> Pooling {
        self.pooling
    }

    /// How many tokens the model sees for `text`, counting the prompt and the special tokens.
    pub fn tokens(&self, text: &str) -> Result<usize, Error> {
        Ok(self.tokenize(text)?.get_ids().len())
    }

    /// Runs the model on `text` up to layer `layer`, then waits until the device has finished. On Metal the work is
    /// queued, so the wait marks the moment layer `layer` is done. Timing this gives T_l, the model time to layer l.
    pub fn run_to(&self, text: &str, layer: usize) -> Result<(), Error> {
        if layer > self.layers {
            return Err(Error::new(ErrorKind::InvalidInput, "the model has fewer layers"));
        }

        self.run(text, layer, &mut NoHook)?;
        self.device.synchronize().map_err(model_failure)
    }

    fn tokenize(&self, text: &str) -> Result<tokenizers::Encoding, Error> {
        let input = format!("{}{text}", self.prompt);
        self.tokenizer
            .encode(input.as_str(), true)
            .map_err(|e| Error::other(e.to_string()))
    }

    /// Runs the embeddings and `layers` transformer layers, hands the hook the summaries it wants, and returns the
    /// last state.
    fn run(&self, text: &str, layers: usize, hook: &mut dyn LayerHook) -> Result<Tensor, Error> {
        let encoding = self.tokenize(text)?;
        let input_ids = Tensor::new(encoding.get_ids(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(model_failure)?;
        let token_type_ids = Tensor::new(encoding.get_type_ids(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(model_failure)?;
        let mut observe = |layer: usize, state: &Tensor| -> candle_core::Result<()> {
            if hook.wants(layer) {
                let pooled = summary(state)?.to_vec1::<f32>()?;
                hook.on_layer(layer, &pooled);
            }
            Ok(())
        };
        self.model
            .forward_layers(&input_ids, &token_type_ids, None, layers, &mut observe)
            .map_err(model_failure)
    }
}

impl LayerEncoder for CandleEncoder {
    fn layers(&self) -> usize {
        self.layers
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn encode(&mut self, text: &str, hook: &mut dyn LayerHook) -> Result<Vec<f32>, Error> {
        let state = self.run(text, self.layers, hook)?;
        let pooled = match self.pooling {
            Pooling::Cls => state.i((0, 0)),
            Pooling::Mean => state.mean(1).and_then(|m| m.squeeze(0)),
        };
        let mut embedding = pooled.and_then(|p| p.to_vec1::<f32>()).map_err(model_failure)?;
        let norm = embedding
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            .sqrt();
        let scale = (1.0 / norm.max(1e-12)) as f32;
        embedding.iter_mut().for_each(|x| *x *= scale);
        Ok(embedding)
    }

    fn queues_work(&self) -> bool {
        self.device.is_metal()
    }
}

/// concat(average over the tokens, first token) of a state shaped (1, tokens, d), as 2d values.
fn summary(state: &Tensor) -> candle_core::Result<Tensor> {
    let mean = state.mean(1)?;
    let first = state.i((.., 0))?;
    Tensor::cat(&[&mean, &first], 1)?.squeeze(0)
}

/// The snapshot directory of the Hugging Face model `repo`, eg. "Snowflake/snowflake-arctic-embed-m-v1.5", in the
/// local cache. The cache is `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`. It is the
/// snapshot that `refs/main` names, or else the only snapshot there. Nothing is downloaded.
pub fn snapshot_dir(repo: &str) -> Result<PathBuf, Error> {
    let hub = match (std::env::var_os("HF_HUB_CACHE"), std::env::var_os("HF_HOME")) {
        (Some(hub), _) => PathBuf::from(hub),
        (None, Some(home)) => PathBuf::from(home).join("hub"),
        (None, None) => {
            let home = std::env::var_os("HOME").ok_or_else(|| Error::new(ErrorKind::NotFound, "HOME isn't set"))?;
            PathBuf::from(home).join(".cache/huggingface/hub")
        }
    };
    snapshot_in(&hub, repo)
}

pub(crate) fn snapshot_in(hub: &Path, repo: &str) -> Result<PathBuf, Error> {
    let model = hub.join(format!("models--{}", repo.replace('/', "--")));
    let snapshots = model.join("snapshots");

    if let Ok(commit) = std::fs::read_to_string(model.join("refs").join("main")) {
        let dir = snapshots.join(commit.trim());
        if dir.is_dir() {
            return Ok(dir);
        }
    }

    let mut dirs = Vec::new();

    for entry in std::fs::read_dir(&snapshots)? {
        let path = entry?.path();
        if path.is_dir() {
            dirs.push(path);
        }
    }

    match dirs.len() {
        1 => Ok(dirs.remove(0)),
        0 => Err(Error::new(ErrorKind::NotFound, format!("no snapshot of {repo}"))),
        _ => Err(Error::new(
            ErrorKind::InvalidInput,
            format!("{repo} has several snapshots and no refs/main to choose one"),
        )),
    }
}

/// The sentence-transformers settings of a snapshot.
struct Settings {
    prompt: String,
    pooling: Pooling,
    max_len: Option<usize>,
}

#[derive(Deserialize)]
struct ModuleEntry {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    path: String,
}

#[derive(Deserialize)]
struct PoolingConfig {
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_max_tokens: bool,
    #[serde(default)]
    pooling_mode_mean_sqrt_len_tokens: bool,
    #[serde(default)]
    pooling_mode_weightedmean_tokens: bool,
    #[serde(default)]
    pooling_mode_lasttoken: bool,
    #[serde(default = "yes")]
    include_prompt: bool,
}

#[derive(Deserialize, Default)]
struct PromptConfig {
    #[serde(default)]
    prompts: HashMap<String, String>,
}

#[derive(Deserialize, Default)]
struct SentenceBertConfig {
    max_seq_length: Option<usize>,
}

fn yes() -> bool {
    true
}

impl Settings {
    fn read(snapshot: &Path) -> Result<Settings, Error> {
        let modules: Vec<ModuleEntry> = read_json(&snapshot.join("modules.json"))?;
        let mut pooling_dir = None;

        for module in &modules {
            match module.kind.rsplit('.').next() {
                Some("Transformer" | "Normalize") => {}
                Some("Pooling") => pooling_dir = Some(snapshot.join(&module.path)),
                _ => return Err(unsupported(&format!("a {} module", module.kind))),
            }
        }

        let pooling_dir = pooling_dir.ok_or_else(|| damaged("modules.json has no pooling module"))?;
        let config: PoolingConfig = read_json(&pooling_dir.join("config.json"))?;
        let prompts: PromptConfig = read_json_or_default(&snapshot.join("config_sentence_transformers.json"))?;
        let sbert: SentenceBertConfig = read_json_or_default(&snapshot.join("sentence_bert_config.json"))?;
        let prompt = prompts.prompts.get("query").cloned().unwrap_or_default();
        let others = config.pooling_mode_max_tokens
            || config.pooling_mode_mean_sqrt_len_tokens
            || config.pooling_mode_weightedmean_tokens
            || config.pooling_mode_lasttoken;
        let pooling = match (config.pooling_mode_cls_token, config.pooling_mode_mean_tokens, others) {
            (true, false, false) => Pooling::Cls,
            (false, true, false) => Pooling::Mean,
            _ => {
                return Err(unsupported(
                    "a pooling other than the first token's state or the plain average",
                ));
            }
        };

        if pooling == Pooling::Mean && !config.include_prompt && !prompt.is_empty() {
            return Err(unsupported("an average that leaves out the prompt"));
        }

        Ok(Settings {
            prompt,
            pooling,
            max_len: sbert.max_seq_length,
        })
    }
}

/// The tokenizer in `path`, set to cut texts at `max_len` tokens and never to pad. Some snapshots pad every text to a
/// fixed length, eg. all-MiniLM-L6-v2 to 128 tokens, which sentence-transformers turns off too.
fn read_tokenizer(path: &Path, max_len: usize) -> Result<Tokenizer, Error> {
    let bytes = std::fs::read(path)?;
    let mut tokenizer = Tokenizer::from_bytes(&bytes).map_err(|e| damaged(format!("tokenizer.json: {e}")))?;

    if max_len <= special_tokens(&tokenizer) {
        return Err(damaged("the longest input leaves no room for the text"));
    }

    tokenizer.with_padding(None);
    truncate(&mut tokenizer, max_len).map_err(|e| damaged(format!("tokenizer.json: {e}")))?;
    Ok(tokenizer)
}

/// The special tokens the tokenizer adds to a text, eg. 2 for the classification token and the separator.
fn special_tokens(tokenizer: &Tokenizer) -> usize {
    tokenizer.get_post_processor().map_or(0, |p| p.added_tokens(false))
}

/// Sets the tokenizer to cut texts at `max_len` tokens, which must be more than its special tokens.
fn truncate(tokenizer: &mut Tokenizer, max_len: usize) -> Result<(), tokenizers::Error> {
    tokenizer.with_truncation(Some(TruncationParams {
        max_length: max_len,
        ..TruncationParams::default()
    }))?;
    Ok(())
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, Error> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|e| damaged(format!("{}: {e}", path.display())))
}

/// Like `read_json`, but a missing file gives the default.
fn read_json_or_default<T: DeserializeOwned + Default>(path: &Path) -> Result<T, Error> {
    match read_json(path) {
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(T::default()),
        other => other,
    }
}

fn damaged(e: impl ToString) -> Error {
    Error::new(ErrorKind::InvalidData, e.to_string())
}

fn unsupported(what: &str) -> Error {
    Error::new(ErrorKind::Unsupported, format!("this encoder can't run {what}"))
}

fn model_failure(e: candle_core::Error) -> Error {
    Error::other(e.to_string())
}

#[cfg(test)]
#[path = "../../tests/unit/encoder/candle.rs"]
mod tests;

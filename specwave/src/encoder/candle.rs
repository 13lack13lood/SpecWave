//! The encoder on candle, for sentence-transformers models stored as Hugging Face snapshot directories.
//!
//! A snapshot holds `config.json`, `model.safetensors` and `tokenizer.json`, plus the sentence-transformers settings
//! in `modules.json`, `sentence_bert_config.json`, `config_sentence_transformers.json` and the pooling folder that
//! `modules.json` names, eg. `1_Pooling/config.json`. The network and its sizes come from `config.json`, and the rest
//! from those settings, as sentence-transformers reads them.
//! - **Network.** The `model_type` of `config.json` picks it: BERT (Bidirectional Encoder Representations from
//!   Transformers) for `bert`, eg. arctic-embed-m, NomicBERT for `nomic_bert`, i.e. nomic-embed-text-v1.5, and Qwen3
//!   for `qwen3`, eg. Qwen3-Embedding-0.6B. A config without one is BERT.
//! - **Pooling.** The state of the classification (CLS) token, eg. for arctic-embed-m, the average over the tokens,
//!   eg. for all-MiniLM-L6-v2 and e5-base-v2, or the state of the last token, eg. for Qwen3-Embedding-0.6B.
//! - **Prompts.** The text in front of each query and of each passage, i.e. each document of the corpus, eg. "query: "
//!   and "passage: " for e5-base-v2. A model whose settings don't hold its prompts, eg. bge-small-en-v1.5, gets them
//!   from the [`models`](super::models) list, which names the file of the snapshot that gives them.
//! - **Lowercasing.** `do_lower_case` puts a lowercasing step in front of the tokenizer's own steps, unless they hold
//!   one already.
//! - **Normalization.** The encoder always returns an embedding of length 1, for searches by cosine. It refuses a
//!   model with no normalizing module whose settings name a similarity other than cosine.
//!
//! The encoder takes one text at a time, so there is no padding and the average over the tokens is the masked
//! average. Each layer's summary, concat(average, pooling token), is computed on the device, and only the summaries the
//! hook wants are copied back. The pooling token is the first token, or the last one for a causal network such as
//! Qwen3, where only the last token sees the whole text.

use std::collections::HashMap;
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use tokenizers::normalizers::{Lowercase, Sequence};
use tokenizers::{NormalizerWrapper, PostProcessor, Tokenizer, TruncationParams};

use super::bert::{self, BertModel};
use super::models::{self, Model};
use super::nomic_bert::{self, NomicBertModel};
use super::qwen3;
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
    /// The last token's state, eg. the end-of-text token that Qwen3-Embedding's tokenizer appends.
    LastToken,
}

/// A network of the encoder, with its weights.
enum Network {
    Bert(BertModel),
    NomicBert(NomicBertModel),
    Qwen3(qwen3::Model),
}

impl Network {
    /// Runs the embeddings and the first `layers` layers on `input_ids`, and shows each state to `observe` as the
    /// network's `forward_layers` does. `token_type_ids` go to the networks that read them.
    fn forward_layers(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        layers: usize,
        observe: &mut dyn FnMut(usize, &Tensor) -> candle_core::Result<()>,
    ) -> candle_core::Result<Tensor> {
        match self {
            Network::Bert(model) => model.forward_layers(input_ids, token_type_ids, None, layers, observe),
            Network::NomicBert(model) => model.forward_layers(input_ids, token_type_ids, None, layers, observe),
            Network::Qwen3(model) => model.forward_layers(input_ids, layers, observe),
        }
    }

    /// Whether each token sees only itself and the tokens before it, so only the last token sees the whole text.
    fn causal(&self) -> bool {
        matches!(self, Network::Qwen3(_))
    }
}

/// The settings in a snapshot's `config.json`, for the network its `model_type` names.
enum NetworkConfig {
    Bert(bert::Config),
    NomicBert(nomic_bert::Config),
    Qwen3(qwen3::Config),
}

#[derive(Deserialize)]
struct ModelType {
    model_type: Option<String>,
}

/// The sizes of a network that the encoder reads.
struct Shape {
    layers: usize,
    dim: usize,
    /// The positions, the most tokens the network can read.
    positions: usize,
    vocab: usize,
}

impl NetworkConfig {
    /// Reads `config.json` at `path`. A model type this encoder has no network for gives `Unsupported`.
    fn read(path: &Path) -> Result<NetworkConfig, Error> {
        let kind: ModelType = read_json(path)?;

        match kind.model_type.as_deref() {
            None | Some("bert") => Ok(NetworkConfig::Bert(read_json(path)?)),
            Some("nomic_bert") => Ok(NetworkConfig::NomicBert(read_json(path)?)),
            Some("qwen3") => Ok(NetworkConfig::Qwen3(read_json(path)?)),
            Some(other) => Err(unsupported(&format!("a model of type {other}"))),
        }
    }

    /// The network's sizes. A shape that no network of its kind has gives `InvalidData`, and a setting the network
    /// can't run gives `Unsupported`.
    fn shape(&self) -> Result<Shape, Error> {
        match self {
            NetworkConfig::Bert(c) => {
                if c.num_hidden_layers == 0
                    || c.num_attention_heads == 0
                    || !c.hidden_size.is_multiple_of(c.num_attention_heads)
                {
                    return Err(damaged("config.json has a shape no BERT model has"));
                }
                Ok(Shape {
                    layers: c.num_hidden_layers,
                    dim: c.hidden_size,
                    positions: c.max_position_embeddings,
                    vocab: c.vocab_size,
                })
            }
            NetworkConfig::NomicBert(c) => {
                // Each head turns its values in pairs for the rotary embedding, so its size must be even.
                if c.n_layer == 0 || c.n_head == 0 || !c.n_embd.is_multiple_of(2 * c.n_head) {
                    return Err(damaged("config.json has a shape no NomicBERT model has"));
                }
                if let Some(what) = c.unsupported() {
                    return Err(unsupported(what));
                }
                Ok(Shape {
                    layers: c.n_layer,
                    dim: c.n_embd,
                    positions: c.n_positions,
                    vocab: c.vocab_size,
                })
            }
            NetworkConfig::Qwen3(c) => {
                if c.num_hidden_layers == 0
                    || c.num_key_value_heads == 0
                    || c.num_attention_heads == 0
                    || !c.num_attention_heads.is_multiple_of(c.num_key_value_heads)
                    || c.head_dim == 0
                    || !c.head_dim.is_multiple_of(2)
                {
                    return Err(damaged("config.json has a shape no Qwen3 model has"));
                }
                if let Some(what) = c.unsupported() {
                    return Err(unsupported(what));
                }
                Ok(Shape {
                    layers: c.num_hidden_layers,
                    dim: c.hidden_size,
                    positions: c.max_position_embeddings,
                    vocab: c.vocab_size,
                })
            }
        }
    }

    fn load(&self, vb: VarBuilder) -> candle_core::Result<Network> {
        match self {
            NetworkConfig::Bert(c) => BertModel::load(vb, c).map(Network::Bert),
            NetworkConfig::NomicBert(c) => NomicBertModel::load(vb, c).map(Network::NomicBert),
            NetworkConfig::Qwen3(c) => qwen3::Model::load(vb, c).map(Network::Qwen3),
        }
    }
}

/// A sentence-transformers model on candle, with its tokenizer, prompts and pooling.
pub struct CandleEncoder {
    model: Network,
    tokenizer: Tokenizer,
    device: Device,
    prompt: String,
    passage_prompt: String,
    pooling: Pooling,
    dim: usize,
    layers: usize,
    /// The model's positions, the most tokens it can read.
    positions: usize,
}

impl CandleEncoder {
    /// Loads the model in `snapshot`, a Hugging Face snapshot directory, onto `backend`. When the snapshot sits in the
    /// Hugging Face cache under a model of the [`models`](super::models) list, eg. bge-small-en-v1.5, the prompts that
    /// the list holds fill in what its settings leave out. A missing file keeps its `NotFound`, a damaged one gives
    /// `InvalidData`, as does a listed prompt that its settings give otherwise. Settings this encoder can't follow, eg.
    /// a model type it has no network for or an extra dense layer after the pooling, give `Unsupported`, as does Metal
    /// where it isn't available. The weights load in f32 whatever their stored precision, eg. bfloat16 for
    /// Qwen3-Embedding-0.6B.
    pub fn open(snapshot: &Path, backend: Backend) -> Result<CandleEncoder, Error> {
        let config = NetworkConfig::read(&snapshot.join("config.json"))?;
        let listed = match repo_of(snapshot) {
            Some(repo) => models::find(&repo)?,
            None => None,
        };
        let settings = Settings::read(snapshot, listed)?;
        let shape = config.shape()?;
        let device = match backend {
            Backend::Cpu => Device::Cpu,
            Backend::Metal => Device::new_metal(0).map_err(|e| Error::new(ErrorKind::Unsupported, e.to_string()))?,
        };
        let max_len = settings.max_len.unwrap_or(shape.positions);
        let tokenizer = read_tokenizer(
            &snapshot.join("tokenizer.json"),
            max_len.min(shape.positions),
            settings.lowercase,
        )?;

        // An id past the model's vocabulary fails on the processor, but Metal's lookup reads the last row instead. So
        // every token the tokenizer knows, its added tokens too, needs a row, even where the ids leave gaps.
        let largest = tokenizer.get_vocab(true).into_values().max();

        if largest.is_some_and(|id| id as usize >= shape.vocab) {
            return Err(damaged("tokenizer.json has a token id past config.json's vocabulary"));
        }

        let weights = std::fs::read(snapshot.join("model.safetensors"))?;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device).map_err(damaged)?;
        let model = config.load(vb).map_err(damaged)?;
        Ok(CandleEncoder {
            model,
            tokenizer,
            device,
            prompt: settings.query,
            passage_prompt: settings.passage,
            pooling: settings.pooling,
            dim: shape.dim,
            layers: shape.layers,
            positions: shape.positions,
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

    /// The text put in front of every passage, eg. "passage: " for e5-base-v2. Most models have none.
    pub fn passage_prompt(&self) -> &str {
        &self.passage_prompt
    }

    pub fn pooling(&self) -> Pooling {
        self.pooling
    }

    /// How many tokens the model sees for the query `text`, counting the prompt and the special tokens.
    pub fn tokens(&self, text: &str) -> Result<usize, Error> {
        Ok(self.tokenize(&self.prompt, text)?.get_ids().len())
    }

    /// Runs the model on the query `text` up to layer `layer`, then waits until the device has finished. On Metal the
    /// work is queued, so the wait marks the moment layer `layer` is done. Timing this gives T_l, the model time to
    /// layer l.
    pub fn run_to(&self, text: &str, layer: usize) -> Result<(), Error> {
        if layer > self.layers {
            return Err(Error::new(ErrorKind::InvalidInput, "the model has fewer layers"));
        }

        self.run(&self.prompt, text, layer, &mut NoHook)?;
        self.device.synchronize().map_err(model_failure)
    }

    /// Runs the model on the passage `text`, a document of the corpus, with the passage prompt. It calls `hook` and
    /// returns the final normalized embedding as `encode` does for a query.
    pub fn encode_passage(&self, text: &str, hook: &mut dyn LayerHook) -> Result<Vec<f32>, Error> {
        self.embed(&self.passage_prompt, text, hook)
    }

    fn tokenize(&self, prompt: &str, text: &str) -> Result<tokenizers::Encoding, Error> {
        let input = format!("{prompt}{text}");
        self.tokenizer
            .encode(input.as_str(), true)
            .map_err(|e| Error::other(e.to_string()))
    }

    /// Runs the embeddings and `layers` transformer layers on `prompt` followed by `text`, hands the hook the summaries
    /// it wants, and returns the last state.
    fn run(&self, prompt: &str, text: &str, layers: usize, hook: &mut dyn LayerHook) -> Result<Tensor, Error> {
        let encoding = self.tokenize(prompt, text)?;
        let input_ids = Tensor::new(encoding.get_ids(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(model_failure)?;
        let token_type_ids = Tensor::new(encoding.get_type_ids(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(model_failure)?;
        let last = self.model.causal();
        let mut observe = |layer: usize, state: &Tensor| -> candle_core::Result<()> {
            if hook.wants(layer) {
                let pooled = summary(state, last)?.to_vec1::<f32>()?;
                hook.on_layer(layer, &pooled);
            }
            Ok(())
        };
        self.model
            .forward_layers(&input_ids, &token_type_ids, layers, &mut observe)
            .map_err(model_failure)
    }

    /// Runs the whole model on `prompt` followed by `text` and returns its pooled last state, scaled to length 1.
    fn embed(&self, prompt: &str, text: &str, hook: &mut dyn LayerHook) -> Result<Vec<f32>, Error> {
        let state = self.run(prompt, text, self.layers, hook)?;
        let pooled = match self.pooling {
            Pooling::Cls => state.i((0, 0)),
            Pooling::Mean => state.mean(1).and_then(|m| m.squeeze(0)),
            Pooling::LastToken => state.dim(1).and_then(|tokens| state.i((0, tokens.saturating_sub(1)))),
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
}

impl LayerEncoder for CandleEncoder {
    fn layers(&self) -> usize {
        self.layers
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn encode(&mut self, text: &str, hook: &mut dyn LayerHook) -> Result<Vec<f32>, Error> {
        self.embed(&self.prompt, text, hook)
    }

    fn queues_work(&self) -> bool {
        self.device.is_metal()
    }
}

/// concat(average over the tokens, pooling token) of a state shaped (1, tokens, d), as 2d values. The pooling token is
/// the first token, or with `last` the last one.
fn summary(state: &Tensor, last: bool) -> candle_core::Result<Tensor> {
    let mean = state.mean(1)?;
    let token = match last {
        true => state.i((.., state.dim(1)?.saturating_sub(1)))?,
        false => state.i((.., 0))?,
    };
    Tensor::cat(&[&mean, &token], 1)?.squeeze(0)
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

/// The repository id that a snapshot's place in the Hugging Face cache names, eg. "BAAI/bge-small-en-v1.5" for
/// `<hub>/models--BAAI--bge-small-en-v1.5/snapshots/<commit>`, or None for a folder laid out otherwise. Repository ids
/// never hold "--", so the cache's names map back to them.
fn repo_of(snapshot: &Path) -> Option<String> {
    let snapshots = snapshot.parent()?;

    if snapshots.file_name()? != "snapshots" {
        return None;
    }

    let name = snapshots.parent()?.file_name()?.to_str()?.strip_prefix("models--")?;
    Some(name.replacen("--", "/", 1))
}

/// The sentence-transformers settings of a snapshot, with the prompts that the models list adds.
struct Settings {
    query: String,
    passage: String,
    pooling: Pooling,
    max_len: Option<usize>,
    /// Whether each text is lowercased before the tokenizer's own steps.
    lowercase: bool,
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
    similarity_fn_name: Option<String>,
}

#[derive(Deserialize, Default)]
struct SentenceBertConfig {
    max_seq_length: Option<usize>,
    #[serde(default)]
    do_lower_case: bool,
}

fn yes() -> bool {
    true
}

/// The names a passage prompt goes by in sentence-transformers' settings, in the order it tries them.
const PASSAGE_NAMES: [&str; 3] = ["document", "passage", "corpus"];

impl Settings {
    /// Reads the settings of `snapshot`. `listed` is its model's entry in the models list, if it has one.
    fn read(snapshot: &Path, listed: Option<&Model>) -> Result<Settings, Error> {
        let modules: Vec<ModuleEntry> = read_json(&snapshot.join("modules.json"))?;
        let (mut pooling_dir, mut normalizes) = (None, false);

        for module in &modules {
            match module.kind.rsplit('.').next() {
                Some("Transformer") => {}
                Some("Normalize") => normalizes = true,
                Some("Pooling") => pooling_dir = Some(snapshot.join(&module.path)),
                _ => return Err(unsupported(&format!("a {} module", module.kind))),
            }
        }

        let pooling_dir = pooling_dir.ok_or_else(|| damaged("modules.json has no pooling module"))?;
        let config: PoolingConfig = read_json(&pooling_dir.join("config.json"))?;
        let prompts: PromptConfig = read_json_or_default(&snapshot.join("config_sentence_transformers.json"))?;
        let sbert: SentenceBertConfig = read_json_or_default(&snapshot.join("sentence_bert_config.json"))?;

        if !normalizes && prompts.similarity_fn_name.as_deref().is_some_and(|s| s != "cosine") {
            return Err(unsupported(
                "a model scored by a similarity other than cosine without a normalizing module",
            ));
        }

        let query = pick_prompt(
            &prompts.prompts,
            &["query"],
            listed.and_then(|m| m.query_prompt.as_deref()),
        )?;
        let passage = pick_prompt(
            &prompts.prompts,
            &PASSAGE_NAMES,
            listed.and_then(|m| m.passage_prompt.as_deref()),
        )?;
        let others = config.pooling_mode_max_tokens
            || config.pooling_mode_mean_sqrt_len_tokens
            || config.pooling_mode_weightedmean_tokens;
        let modes = (
            config.pooling_mode_cls_token,
            config.pooling_mode_mean_tokens,
            config.pooling_mode_lasttoken,
            others,
        );
        let pooling = match modes {
            (true, false, false, false) => Pooling::Cls,
            (false, true, false, false) => Pooling::Mean,
            (false, false, true, false) => Pooling::LastToken,
            _ => {
                return Err(unsupported(
                    "a pooling other than the first token's state, the plain average or the last token's state",
                ));
            }
        };

        if pooling == Pooling::Mean && !config.include_prompt && !(query.is_empty() && passage.is_empty()) {
            return Err(unsupported("an average that leaves out the prompt"));
        }

        Ok(Settings {
            query,
            passage,
            pooling,
            max_len: sbert.max_seq_length,
            lowercase: sbert.do_lower_case,
        })
    }
}

/// The prompt under the first of `names` in a snapshot's `prompts`, or else the prompt the models list gives, or else
/// none. A listed prompt that the snapshot gives otherwise gives `InvalidData`, so the list never hides a change.
fn pick_prompt(prompts: &HashMap<String, String>, names: &[&str], listed: Option<&str>) -> Result<String, Error> {
    match (names.iter().find_map(|name| prompts.get(*name)), listed) {
        (Some(own), Some(listed)) if own != listed => Err(damaged(format!(
            "the models list gives the prompt {listed:?}, but the snapshot gives {own:?}"
        ))),
        (Some(own), _) => Ok(own.clone()),
        (None, listed) => Ok(listed.unwrap_or_default().to_string()),
    }
}

/// The tokenizer in `path`, set to cut texts at `max_len` tokens and never to pad. Some snapshots pad every text to a
/// fixed length, eg. all-MiniLM-L6-v2 to 128 tokens, which sentence-transformers turns off too. With `lowercase` a
/// lowercasing step goes in front of the tokenizer's own steps unless they hold one already, as sentence-transformers
/// does for `do_lower_case`.
fn read_tokenizer(path: &Path, max_len: usize, lowercase: bool) -> Result<Tokenizer, Error> {
    let bytes = std::fs::read(path)?;
    let mut tokenizer = Tokenizer::from_bytes(&bytes).map_err(|e| damaged(format!("tokenizer.json: {e}")))?;

    if max_len <= special_tokens(&tokenizer) {
        return Err(damaged("the longest input leaves no room for the text"));
    }

    tokenizer.with_padding(None);
    truncate(&mut tokenizer, max_len).map_err(|e| damaged(format!("tokenizer.json: {e}")))?;

    if lowercase {
        let steps = match tokenizer.get_normalizer() {
            None => vec![],
            Some(NormalizerWrapper::Sequence(steps)) => steps.as_ref().to_vec(),
            Some(step) => vec![step.clone()],
        };
        if !steps.iter().any(|s| matches!(s, NormalizerWrapper::Lowercase(_))) {
            let mut lowered: Vec<NormalizerWrapper> = vec![Lowercase.into()];
            lowered.extend(steps);
            tokenizer.with_normalizer(Some(Sequence::new(lowered)));
        }
    }

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

//! The embedding models SpecWave knows by a short name, eg. `bge-small`, with the settings their snapshots leave out.
//!
//! The list is `models.json`, next to this file, which the data scripts read too. Each entry holds these fields.
//! - `name` is the short name, and `repo` the Hugging Face repository.
//! - `layers` are the layers that may fire, i.e. the layers whose states get a predictor. Layer l is the state after
//!   l transformer layers.
//! - `query_prompt` and `passage_prompt` are the texts put in front of each query and each passage, for a model whose
//!   sentence-transformers settings don't hold them. `prompts_from` names the file of the snapshot that gives them, eg.
//!   the model card `README.md`. A prompt that the settings do hold comes from there, and the list leaves it out.
//!
//! Everything else, eg. the sizes, the pooling and the normalization, comes from the snapshot's own files.

use std::io::{Error, ErrorKind};
use std::sync::OnceLock;

/// One model of the list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    /// The short name, eg. "bge-small".
    pub name: String,
    /// The Hugging Face repository id, eg. "BAAI/bge-small-en-v1.5".
    pub repo: String,
    /// The layers that may fire, in increasing order.
    pub layers: Vec<usize>,
    pub query_prompt: Option<String>,
    pub passage_prompt: Option<String>,
    /// The file of the snapshot that gives the prompts, if the list holds any.
    pub prompts_from: Option<String>,
}

const LIST: &str = include_str!("models.json");

/// The fields an entry may have.
const FIELDS: [&str; 6] = [
    "name",
    "repo",
    "layers",
    "query_prompt",
    "passage_prompt",
    "prompts_from",
];

/// Every model of the list, in its order. A damaged list gives `InvalidData`.
pub fn registered() -> Result<&'static [Model], Error> {
    static PARSED: OnceLock<Result<Vec<Model>, String>> = OnceLock::new();

    match PARSED.get_or_init(|| parse(LIST).map_err(|e| e.to_string())) {
        Ok(models) => Ok(models),
        Err(e) => Err(Error::new(ErrorKind::InvalidData, e.clone())),
    }
}

/// The model whose short name or repository id is `name`, if the list has it.
pub fn find(name: &str) -> Result<Option<&'static Model>, Error> {
    Ok(registered()?.iter().find(|m| m.name == name || m.repo == name))
}

/// The repository id of the short name `name`, or `name` itself when it isn't one, eg. a repository id.
pub fn repo(name: &str) -> Result<&str, Error> {
    Ok(find(name)?.map_or(name, |m| m.repo.as_str()))
}

/// The models of a list in the format of `models.json`. A list that isn't an array of entries with the fields above,
/// a name or repository given twice, layers that aren't increasing, or prompts without their file give `InvalidData`.
fn parse(text: &str) -> Result<Vec<Model>, Error> {
    let value: serde_json::Value = serde_json::from_str(text).map_err(|e| damaged(&format!("models.json: {e}")))?;
    let entries = value.as_array().ok_or_else(|| damaged("models.json isn't a list"))?;
    let mut models: Vec<Model> = Vec::with_capacity(entries.len());

    for (i, entry) in entries.iter().enumerate() {
        let model = entry_of(entry).map_err(|e| damaged(&format!("models.json entry {i}: {e}")))?;
        if models.iter().any(|m| m.name == model.name || m.repo == model.repo) {
            return Err(damaged(&format!("models.json names {} twice", model.name)));
        }
        models.push(model);
    }

    Ok(models)
}

fn entry_of(entry: &serde_json::Value) -> Result<Model, String> {
    let fields = entry.as_object().ok_or("it isn't an object")?;

    if let Some(other) = fields.keys().find(|k| !FIELDS.contains(&k.as_str())) {
        return Err(format!("it has an unknown field {other}"));
    }

    let text = |key: &str| match fields.get(key) {
        None => Ok(None),
        Some(v) => v
            .as_str()
            .map(|s| Some(s.to_string()))
            .ok_or(format!("{key} isn't text")),
    };
    let name = text("name")?.filter(|n| !n.is_empty()).ok_or("it has no name")?;
    let repo = text("repo")?
        .filter(|r| r.split('/').count() == 2 && !r.contains("--"))
        .ok_or("it has no repository id of the form org/name")?;
    let layers = fields
        .get("layers")
        .and_then(|v| v.as_array())
        .ok_or("it has no layers")?
        .iter()
        .map(|l| l.as_u64().map(|l| l as usize).ok_or("a layer isn't a whole number"))
        .collect::<Result<Vec<usize>, _>>()?;

    if layers.is_empty() || !layers.windows(2).all(|w| w[0] < w[1]) {
        return Err("its layers aren't one or more increasing numbers".to_string());
    }

    let (query_prompt, passage_prompt) = (text("query_prompt")?, text("passage_prompt")?);
    let prompts_from = text("prompts_from")?.filter(|f| !f.is_empty());

    if (query_prompt.is_some() || passage_prompt.is_some()) != prompts_from.is_some() {
        return Err("it needs prompts_from exactly when it has a prompt".to_string());
    }

    Ok(Model {
        name,
        repo,
        layers,
        query_prompt,
        passage_prompt,
        prompts_from,
    })
}

fn damaged(message: &str) -> Error {
    Error::new(ErrorKind::InvalidData, message.to_string())
}

#[cfg(test)]
#[path = "../../tests/unit/encoder/models.rs"]
mod tests;

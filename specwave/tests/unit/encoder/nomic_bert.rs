//! Unit tests for the copied NomicBERT network.

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;

use super::*;
use crate::encoder::testutil::host::{Rows, add, attend, flat, layer_norm, linear, rotate, row, rows, silu};
use crate::encoder::testutil::{assert_close, tiny_nomic_config, tiny_nomic_weights, values};

fn tiny_model(layers: usize) -> NomicBertModel {
    let config = tiny_nomic_config(layers);
    let vb = VarBuilder::from_tensors(tiny_nomic_weights(&config, 7), DType::F32, &Device::Cpu);
    NomicBertModel::load(vb, &config).unwrap()
}

fn ids(rows: &[&[u32]]) -> Tensor {
    let rows: Vec<Tensor> = rows.iter().map(|r| Tensor::new(*r, &Device::Cpu).unwrap()).collect();
    Tensor::stack(&rows, 0).unwrap()
}

/// Every state `forward_layers` shows, with its layer number.
fn observed(
    model: &NomicBertModel,
    ids: &Tensor,
    mask: Option<&Tensor>,
    layers: usize,
) -> (Vec<(usize, Vec<f32>)>, Tensor) {
    let mut seen = Vec::new();
    let types = ids.zeros_like().unwrap();
    let last = model
        .forward_layers(ids, &types, mask, layers, &mut |layer, state| {
            seen.push((layer, values(state)));
            Ok(())
        })
        .unwrap();
    (seen, last)
}

#[test]
fn layer_0_is_the_embedding_norm_output_and_layer_i_plus_1_is_the_output_of_block_i() {
    let model = tiny_model(2);
    let ids = ids(&[&[2, 4, 5, 6, 7, 8, 9, 3]]);
    let types = ids.zeros_like().unwrap();
    let (seen, last) = observed(&model, &ids, None, 2);

    let embedded = model.embeddings.forward(&ids, &types).unwrap();
    let h0 = model.emb_ln.forward(&embedded).unwrap();
    let rotary = &model.encoder.rotary_emb;
    let h1 = model.encoder.layers[0].forward(&h0, None, rotary).unwrap();
    let h2 = model.encoder.layers[1].forward(&h1, None, rotary).unwrap();
    let expected = [values(&h0), values(&h1), values(&h2)];
    assert_eq!(seen.iter().map(|(layer, _)| *layer).collect::<Vec<_>>(), [0, 1, 2]);

    for (layer, state) in &seen {
        assert_eq!(*state, expected[*layer], "layer {layer}");
    }

    assert_eq!(values(&last), expected[2]);
    assert_eq!(last.dims(), [1, 8, 16]);
    assert_ne!(values(&embedded), expected[0]);
    assert_ne!(expected[0], expected[1]);
    assert_ne!(expected[1], expected[2]);
}

#[test]
fn fewer_layers_stop_early() {
    let model = tiny_model(2);
    let ids = ids(&[&[2, 10, 11, 12, 3]]);
    let (all, _) = observed(&model, &ids, None, 2);
    let (one, last) = observed(&model, &ids, None, 1);
    let (none, first) = observed(&model, &ids, None, 0);

    assert_eq!(one, all[..2]);
    assert_eq!(values(&last), all[1].1);
    assert_eq!(none, all[..1]);
    assert_eq!(values(&first), all[0].1);
}

/// The states of a text with token ids `ids`, layer by layer, from NomicBERT's math written out on the host. Each block
/// turns the query and key of each head by the token's position, attends over every token, then gates its
/// feed-forward network with SiLU, i.e. fc2(fc11(x)·silu(fc12(x))), with a layer norm after each residual sum.
fn plain_states(weights: &HashMap<String, Tensor>, config: &Config, ids: &[u32]) -> Vec<Rows> {
    let (h, heads, eps) = (config.n_embd, config.n_head, config.layer_norm_epsilon);
    let d = h / heads;
    let words = rows(weights, "embeddings.word_embeddings.weight");
    let types = rows(weights, "embeddings.token_type_embeddings.weight");
    let embedded: Rows = ids.iter().map(|&id| words[id as usize].clone()).collect();
    let typed = add(&embedded, &vec![types[0].clone(); ids.len()]);
    let mut x = layer_norm(
        &typed,
        &row(weights, "emb_ln.weight"),
        &row(weights, "emb_ln.bias"),
        eps,
    );
    let mut states = vec![x.clone()];

    for layer in 0..config.n_layer {
        let w = |name: &str| rows(weights, &format!("encoder.layers.{layer}.{name}"));
        let norm =
            |x: &Rows, name: &str| layer_norm(x, &w(&format!("{name}.weight"))[0], &w(&format!("{name}.bias"))[0], eps);
        let qkv = linear(&x, &w("attn.Wqkv.weight"));
        let mut context = vec![vec![0.0; h]; ids.len()];
        for head in 0..heads {
            // The fused projection holds the queries, then the keys, then the values, each head after head.
            let part = |start: usize, turn: bool| -> Rows {
                qkv.iter()
                    .enumerate()
                    .map(|(position, r)| {
                        let values = &r[start + head * d..start + (head + 1) * d];
                        if turn {
                            rotate(values, position, config.rotary_emb_base)
                        } else {
                            values.to_vec()
                        }
                    })
                    .collect()
            };
            let out = attend(&part(0, true), &part(h, true), &part(2 * h, false), false);
            for (c, o) in context.iter_mut().zip(&out) {
                c[head * d..(head + 1) * d].copy_from_slice(o);
            }
        }
        let attended = norm(&add(&x, &linear(&context, &w("attn.out_proj.weight"))), "norm1");
        let (up, gate) = (
            linear(&attended, &w("mlp.fc11.weight")),
            linear(&attended, &w("mlp.fc12.weight")),
        );
        let gated: Rows = up
            .iter()
            .zip(&gate)
            .map(|(u, g)| u.iter().zip(g).map(|(a, b)| a * silu(*b)).collect())
            .collect();
        x = norm(&add(&attended, &linear(&gated, &w("mlp.fc2.weight"))), "norm2");
        states.push(x.clone());
    }

    states
}

#[test]
fn every_layer_matches_the_math_written_out() {
    let config = tiny_nomic_config(2);
    let weights = tiny_nomic_weights(&config, 7);
    let model = NomicBertModel::load(
        VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu),
        &config,
    )
    .unwrap();
    let text = [2, 4, 5, 6, 7, 8, 9, 3];
    let (seen, _) = observed(&model, &ids(&[&text]), None, 2);
    let expected = plain_states(&weights, &config, &text);

    for ((layer, state), plain) in seen.iter().zip(&expected) {
        assert_close(state, &flat(plain), 1e-4, &format!("layer {layer}"));
    }
}

#[test]
fn a_padded_batch_gives_each_text_the_states_it_has_alone() {
    let model = tiny_model(2);
    let long: &[u32] = &[2, 4, 5, 6, 7, 8, 3];
    let short: &[u32] = &[2, 10, 12, 3];
    let padded: &[u32] = &[2, 10, 12, 3, 0, 0, 0];
    let batch = ids(&[long, padded]);
    let mask = ids(&[&[1; 7], &[1, 1, 1, 1, 0, 0, 0]]);
    let (_, both) = observed(&model, &batch, Some(&mask), 2);
    let (_, alone_long) = observed(&model, &ids(&[long]), None, 2);
    let (_, alone_short) = observed(&model, &ids(&[short]), None, 2);

    assert_close(
        &values(&both.i(0).unwrap()),
        &values(&alone_long),
        1e-5,
        "the long text",
    );
    let real = both.i((1, 0..short.len())).unwrap();
    assert_close(
        &values(&real),
        &values(&alone_short),
        1e-5,
        "the real tokens of the short text",
    );
}

#[test]
fn weights_under_the_model_type_load_too() {
    let config = tiny_nomic_config(2);
    let prefixed = tiny_nomic_weights(&config, 7)
        .into_iter()
        .map(|(name, tensor)| (format!("nomic_bert.{name}"), tensor))
        .collect();
    let vb = VarBuilder::from_tensors(prefixed, DType::F32, &Device::Cpu);
    let model = NomicBertModel::load(vb, &config).unwrap();
    let ids = ids(&[&[2, 4, 7, 3]]);

    assert_eq!(
        observed(&model, &ids, None, 2).0,
        observed(&tiny_model(2), &ids, None, 2).0
    );
}

#[test]
fn a_missing_weight_fails_to_load() {
    let config = tiny_nomic_config(2);
    let mut weights = tiny_nomic_weights(&config, 7);
    weights.remove("encoder.layers.1.mlp.fc12.weight");
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);

    assert!(NomicBertModel::load(vb, &config).is_err());
}

#[test]
fn settings_this_copy_cant_run_are_named() {
    let fine = tiny_nomic_config(2);
    let cases = [
        Config {
            activation_function: "gelu".to_string(),
            ..fine.clone()
        },
        Config {
            rotary_emb_fraction: 0.5,
            ..fine.clone()
        },
        Config {
            rotary_scaling_factor: Some(2.0),
            ..fine.clone()
        },
        Config {
            rotary_emb_scale_base: Some(512.0),
            ..fine.clone()
        },
        Config {
            prenorm: true,
            ..fine.clone()
        },
    ];

    assert_eq!(fine.unsupported(), None);

    for config in &cases {
        assert!(config.unsupported().is_some(), "{config:?}");
    }
}

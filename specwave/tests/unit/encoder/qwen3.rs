//! Unit tests for the copied Qwen3 network.

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;

use super::*;
use crate::encoder::testutil::host::{Rows, add, attend, flat, linear, rms_norm, rotate, row, rows, silu};
use crate::encoder::testutil::{SplitMix64, assert_close, tiny_qwen3_config, tiny_qwen3_weights, values};

fn tiny_model(layers: usize) -> Model {
    let config = tiny_qwen3_config(layers);
    let vb = VarBuilder::from_tensors(tiny_qwen3_weights(&config, 7), DType::F32, &Device::Cpu);
    Model::load(vb, &config).unwrap()
}

fn ids(ids: &[u32]) -> Tensor {
    Tensor::new(ids, &Device::Cpu).unwrap().unsqueeze(0).unwrap()
}

/// Every state `forward_layers` shows, with its layer number.
fn observed(model: &Model, ids: &Tensor, layers: usize) -> (Vec<(usize, Vec<f32>)>, Tensor) {
    let mut seen = Vec::new();
    let last = model
        .forward_layers(ids, layers, &mut |layer, state| {
            seen.push((layer, values(state)));
            Ok(())
        })
        .unwrap();
    (seen, last)
}

#[test]
fn layer_0_is_the_embedding_output_and_the_last_layer_comes_after_the_final_norm() {
    let model = tiny_model(2);
    let ids = ids(&[4, 5, 6, 7, 8, 9, 3]);
    let (seen, last) = observed(&model, &ids, 2);

    let h0 = model.embed_tokens.forward(&ids).unwrap();
    let h1 = model.layers[0].forward(&h0, None).unwrap();
    let h2 = model.layers[1].forward(&h1, None).unwrap();
    let normed = model.norm.forward(&h2).unwrap();
    let expected = [values(&h0), values(&h1), values(&normed)];
    assert_eq!(seen.iter().map(|(layer, _)| *layer).collect::<Vec<_>>(), [0, 1, 2]);

    for (layer, state) in &seen {
        assert_eq!(*state, expected[*layer], "layer {layer}");
    }

    assert_eq!(values(&last), expected[2]);
    assert_eq!(last.dims(), [1, 7, 16]);
    assert_ne!(values(&h2), expected[2]);
}

#[test]
fn a_run_that_stops_early_skips_the_final_norm() {
    let model = tiny_model(2);
    let ids = ids(&[10, 11, 12, 3]);
    let (all, _) = observed(&model, &ids, 2);
    let (one, last) = observed(&model, &ids, 1);
    let (none, first) = observed(&model, &ids, 0);

    assert_eq!(one, all[..2]);
    assert_eq!(values(&last), all[1].1);
    assert_eq!(none, all[..1]);
    assert_eq!(values(&first), all[0].1);
}

/// The states of a text with token ids `ids`, layer by layer, from Qwen3's math written out on the host. Each layer
/// norms its input, norms each head's query and key and turns them by the token's position, lets each token attend to
/// itself and the tokens before it, and gates its feed-forward network with SiLU, i.e. down(silu(gate(x))·up(x)).
/// Query head h reads key and value head h / (query heads per key head). The last layer's output goes through the
/// final norm.
fn plain_states(weights: &HashMap<String, Tensor>, config: &Config, ids: &[u32]) -> Vec<Rows> {
    let (heads, d, eps) = (config.num_attention_heads, config.head_dim, config.rms_norm_eps);
    let shared = heads / config.num_key_value_heads;
    let embed = rows(weights, "embed_tokens.weight");
    let mut x: Rows = ids.iter().map(|&id| embed[id as usize].clone()).collect();
    let mut states = vec![x.clone()];
    let norm_rows = |x: &Rows, weight: &[f64]| -> Rows { x.iter().map(|r| rms_norm(r, weight, eps)).collect() };

    for layer in 0..config.num_hidden_layers {
        let w = |name: &str| rows(weights, &format!("layers.{layer}.{name}"));
        let v = |name: &str| row(weights, &format!("layers.{layer}.{name}"));
        let normed = norm_rows(&x, &v("input_layernorm.weight"));
        let (q, k, values) = (
            linear(&normed, &w("self_attn.q_proj.weight")),
            linear(&normed, &w("self_attn.k_proj.weight")),
            linear(&normed, &w("self_attn.v_proj.weight")),
        );
        let head_of = |m: &Rows, head: usize, norm: Option<&[f64]>| -> Rows {
            m.iter()
                .enumerate()
                .map(|(position, r)| {
                    let part = &r[head * d..(head + 1) * d];
                    match norm {
                        Some(weight) => rotate(&rms_norm(part, weight, eps), position, config.rope_theta),
                        None => part.to_vec(),
                    }
                })
                .collect()
        };
        let (q_norm, k_norm) = (v("self_attn.q_norm.weight"), v("self_attn.k_norm.weight"));
        let mut context = vec![vec![0.0; heads * d]; ids.len()];
        for head in 0..heads {
            let kv = head / shared;
            let out = attend(
                &head_of(&q, head, Some(&q_norm)),
                &head_of(&k, kv, Some(&k_norm)),
                &head_of(&values, kv, None),
                true,
            );
            for (c, o) in context.iter_mut().zip(&out) {
                c[head * d..(head + 1) * d].copy_from_slice(o);
            }
        }
        x = add(&x, &linear(&context, &w("self_attn.o_proj.weight")));
        let normed = norm_rows(&x, &v("post_attention_layernorm.weight"));
        let (gate, up) = (
            linear(&normed, &w("mlp.gate_proj.weight")),
            linear(&normed, &w("mlp.up_proj.weight")),
        );
        let gated: Rows = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| g.iter().zip(u).map(|(a, b)| silu(*a) * b).collect())
            .collect();
        x = add(&x, &linear(&gated, &w("mlp.down_proj.weight")));
        match layer + 1 == config.num_hidden_layers {
            true => states.push(norm_rows(&x, &row(weights, "norm.weight"))),
            false => states.push(x.clone()),
        }
    }

    states
}

#[test]
fn every_layer_matches_the_math_written_out() {
    let config = tiny_qwen3_config(2);
    let weights = tiny_qwen3_weights(&config, 7);
    let model = Model::load(
        VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu),
        &config,
    )
    .unwrap();
    let text = [4, 5, 6, 7, 8, 9, 3];
    let (seen, _) = observed(&model, &ids(&text), 2);
    let expected = plain_states(&weights, &config, &text);

    for ((layer, state), plain) in seen.iter().zip(&expected) {
        assert_close(state, &flat(plain), 1e-4, &format!("layer {layer}"));
    }
}

#[test]
fn a_token_never_sees_the_tokens_after_it() {
    let model = tiny_model(2);
    let (a, _) = observed(&model, &ids(&[4, 5, 6, 7, 8]), 2);
    let (b, _) = observed(&model, &ids(&[4, 5, 6, 10, 11]), 2);

    for ((layer, first), (_, second)) in a.iter().zip(&b) {
        assert_close(&first[..3 * 16], &second[..3 * 16], 1e-6, &format!("layer {layer}"));
        assert_ne!(first[3 * 16..], second[3 * 16..], "layer {layer}");
    }
}

/// A random tensor of `shape` with values in [-1, 1).
fn random(shape: (usize, usize, usize, usize), rng: &mut SplitMix64) -> Tensor {
    let n = shape.0 * shape.1 * shape.2 * shape.3;
    Tensor::from_vec((0..n).map(|_| rng.signed()).collect::<Vec<f32>>(), shape, &Device::Cpu).unwrap()
}

#[test]
fn the_processors_fused_attention_matches_the_plain_one_with_its_mask() {
    let model = tiny_model(1);
    let attention = &model.layers[0].self_attn;
    let mut rng = SplitMix64::new(3);
    let (b, l) = (1, 6);
    let q = random((b, 4, l, 8), &mut rng);
    let k = random((b, 2, l, 8), &mut rng);
    let v = random((b, 2, l, 8), &mut rng);
    let mask = model.causal_mask(b, l, &Device::Cpu).unwrap();
    let fused = attention.forward_cpu_flash_attn(&q, &k, &v, b, l).unwrap();
    let plain = attention.forward_standard_attn(&q, &k, &v, Some(&mask), b, l).unwrap();

    assert_close(&values(&fused), &values(&plain), 1e-5, "the attention output");
    let unmasked = attention.forward_standard_attn(&q, &k, &v, None, b, l).unwrap();
    assert_ne!(values(&unmasked), values(&plain));
}

#[test]
fn repeated_heads_give_query_head_h_the_key_head_h_over_the_repeats() {
    let mut rng = SplitMix64::new(5);
    let kv = random((1, 2, 3, 4), &mut rng);
    let repeated = repeat_kv(kv.clone(), 3).unwrap();
    assert_eq!(repeated.dims(), [1, 6, 3, 4]);

    for head in 0..6 {
        assert_eq!(
            values(&repeated.i((0, head)).unwrap()),
            values(&kv.i((0, head / 3)).unwrap()),
            "head {head}"
        );
    }

    assert_eq!(values(&repeat_kv(kv.clone(), 1).unwrap()), values(&kv));
}

#[test]
fn weights_under_model_load_too() {
    let config = tiny_qwen3_config(2);
    let prefixed = tiny_qwen3_weights(&config, 7)
        .into_iter()
        .map(|(name, tensor)| (format!("model.{name}"), tensor))
        .collect();
    let vb = VarBuilder::from_tensors(prefixed, DType::F32, &Device::Cpu);
    let model = Model::load(vb, &config).unwrap();
    let ids = ids(&[4, 7, 3]);

    assert_eq!(observed(&model, &ids, 2).0, observed(&tiny_model(2), &ids, 2).0);
}

#[test]
fn a_missing_weight_or_another_precision_fails_to_load() {
    let config = tiny_qwen3_config(2);
    let mut weights = tiny_qwen3_weights(&config, 7);
    let half = VarBuilder::from_tensors(weights.clone(), DType::F16, &Device::Cpu);
    assert!(Model::load(half, &config).is_err());

    weights.remove("layers.1.self_attn.k_norm.weight");
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    assert!(Model::load(vb, &config).is_err());
}

#[test]
fn settings_this_copy_cant_run_are_named() {
    let fine = tiny_qwen3_config(2);
    let window = Config {
        use_sliding_window: true,
        ..fine.clone()
    };
    let scaled = Config {
        rope_scaling: Some(serde_json::json!({"rope_type": "yarn", "factor": 4.0})),
        ..fine.clone()
    };

    assert_eq!(fine.unsupported(), None);
    assert!(window.unsupported().is_some());
    assert!(scaled.unsupported().is_some());
}

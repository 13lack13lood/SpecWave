//! Unit tests for the copied BERT (Bidirectional Encoder Representations from Transformers) network.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;

use super::*;
use crate::encoder::testutil::{assert_close, tiny_config, tiny_weights, values};

fn tiny_model(layers: usize) -> BertModel {
    let config = tiny_config(layers);
    let vb = VarBuilder::from_tensors(tiny_weights(&config, 7), DType::F32, &Device::Cpu);
    BertModel::load(vb, &config).unwrap()
}

fn ids(rows: &[&[u32]]) -> Tensor {
    let rows: Vec<Tensor> = rows.iter().map(|r| Tensor::new(*r, &Device::Cpu).unwrap()).collect();
    Tensor::stack(&rows, 0).unwrap()
}

/// Every state `forward_layers` shows, with its layer number.
fn observed(model: &BertModel, ids: &Tensor, mask: Option<&Tensor>, layers: usize) -> (Vec<(usize, Vec<f32>)>, Tensor) {
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
fn layer_0_is_the_embedding_output_and_layer_i_plus_1_is_the_output_of_layer_i() {
    let model = tiny_model(2);
    let ids = ids(&[&[2, 4, 5, 6, 7, 8, 9, 3]]);
    let types = ids.zeros_like().unwrap();
    let (seen, last) = observed(&model, &ids, None, 2);

    let h0 = model.embeddings.forward(&ids, &types).unwrap();
    let h1 = model.encoder.layers[0].forward(&h0, None).unwrap();
    let h2 = model.encoder.layers[1].forward(&h1, None).unwrap();
    let expected = [values(&h0), values(&h1), values(&h2)];
    assert_eq!(seen.iter().map(|(layer, _)| *layer).collect::<Vec<_>>(), [0, 1, 2]);

    for (layer, state) in &seen {
        assert_eq!(*state, expected[*layer], "layer {layer}");
    }

    assert_eq!(values(&last), expected[2]);
    assert_eq!(last.dims(), [1, 8, 16]);
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
    let config = tiny_config(2);
    let prefixed = tiny_weights(&config, 7)
        .into_iter()
        .map(|(name, tensor)| (format!("bert.{name}"), tensor))
        .collect();
    let vb = VarBuilder::from_tensors(prefixed, DType::F32, &Device::Cpu);
    let model = BertModel::load(vb, &config).unwrap();
    let ids = ids(&[&[2, 4, 7, 3]]);

    assert_eq!(
        observed(&model, &ids, None, 2).0,
        observed(&tiny_model(2), &ids, None, 2).0
    );
}

#[test]
fn a_missing_weight_fails_to_load() {
    let config = tiny_config(2);
    let mut weights = tiny_weights(&config, 7);
    weights.remove("encoder.layer.1.output.dense.weight");
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);

    assert!(BertModel::load(vb, &config).is_err());
}

/// Every multiple of 1/10000 from -8 to 8, as f32.
fn grid() -> Vec<f32> {
    (-80_000..=80_000).map(|i| (f64::from(i) / 1e4) as f32).collect()
}

/// The exact GELU x·Φ(x) in f64, where Φ is the standard normal distribution function. It comes from candle's
/// `erfc_f64`, which calls the `libm` crate, a Rust port of the musl C library's math functions.
fn exact_gelu(x: f32) -> f64 {
    let x = f64::from(x);
    0.5 * x * candle_core::cpu::erf::erfc_f64(-x * FRAC_1_SQRT_2)
}

#[test]
fn the_gelu_matches_the_exact_one_from_minus_8_to_8() {
    let xs = grid();
    let found = values(&gelu_erf(&Tensor::new(&xs[..], &Device::Cpu).unwrap()).unwrap());
    let scalar = values(&Tensor::new(&xs[..], &Device::Cpu).unwrap().gelu_erf().unwrap());
    let (mut worst, mut worst_scalar, mut worst_tail) = (0f64, 0f64, 0f64);

    for ((&x, &f), &s) in xs.iter().zip(&found).zip(&scalar) {
        let g = exact_gelu(x);
        let error = (f64::from(f) - g).abs() / (1.0 + g.abs());
        assert!(error <= 1.5e-7, "gelu({x}) is {f}, exact {g}");
        worst = worst.max(error);
        worst_scalar = worst_scalar.max((f64::from(s) - g).abs() / (1.0 + g.abs()));
        // Below -1 the value is tiny, so the error must be small next to the value itself. The rounding of x² to f32
        // limits it to a few millionths at -6.
        if (-6.0..=-1.0).contains(&x) {
            let relative = (f64::from(f) - g).abs() / g.abs();
            assert!(relative <= 4e-6, "gelu({x}) is {f}, exact {g}");
            worst_tail = worst_tail.max(relative);
        }
    }

    eprintln!(
        "worst error {worst:.1e} next to 1 + |gelu|, {worst_scalar:.1e} for candle's, {worst_tail:.1e} relative below -1"
    );
}

#[test]
fn the_processor_takes_the_tensor_op_gelu_and_the_others_keep_theirs() {
    let xs = Tensor::new(&grid()[..], &Device::Cpu).unwrap();
    let found = values(&HiddenAct::Gelu.forward(&xs).unwrap());
    assert_eq!(found, values(&gelu_erf(&xs).unwrap()));
    assert_ne!(found, values(&xs.gelu_erf().unwrap()));
    assert_eq!(
        values(&HiddenAct::GeluApproximate.forward(&xs).unwrap()),
        values(&xs.gelu().unwrap())
    );
    assert_eq!(
        values(&HiddenAct::Relu.forward(&xs).unwrap()),
        values(&xs.relu().unwrap())
    );
}

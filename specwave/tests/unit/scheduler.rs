//! Unit tests for the layer profile and the rule that picks the fire layer, on fake encoders and synthetic profiles.

use std::cell::RefCell;
use std::thread::sleep;

use super::*;

/// An encoder of `layers` layers that runs no model. It asks the hook about each layer in order and counts the
/// summaries it hands over. It can pause before one layer, in every call or in the first only, and skip or repeat a
/// layer.
struct Fake {
    layers: usize,
    pause: Option<(usize, Duration)>,
    pause_first_only: bool,
    skip: Option<usize>,
    twice: Option<usize>,
    past_last: bool,
    queues: bool,
    texts: Vec<String>,
    copies: usize,
}

impl Fake {
    fn new(layers: usize) -> Fake {
        Fake {
            layers,
            pause: None,
            pause_first_only: false,
            skip: None,
            twice: None,
            past_last: false,
            queues: false,
            texts: Vec::new(),
            copies: 0,
        }
    }
}

impl LayerEncoder for Fake {
    fn layers(&self) -> usize {
        self.layers
    }

    fn dim(&self) -> usize {
        4
    }

    fn encode(&mut self, text: &str, hook: &mut dyn LayerHook) -> Result<Vec<f32>, Error> {
        let first = self.texts.is_empty();
        self.texts.push(text.to_string());
        let last = self.layers + usize::from(self.past_last);

        for layer in 0..=last {
            if let Some((at, pause)) = self.pause
                && at == layer
                && (first || !self.pause_first_only)
            {
                sleep(pause);
            }
            if self.skip == Some(layer) {
                continue;
            }
            if hook.wants(layer) {
                self.copies += 1;
                hook.on_layer(layer, &[0.0; 8]);
            }
            if self.twice == Some(layer) {
                hook.wants(layer);
            }
        }

        Ok(vec![0.5; 4])
    }

    fn queues_work(&self) -> bool {
        self.queues
    }
}

#[test]
fn the_profile_times_every_layer_without_copying_any() {
    let mut fake = Fake::new(3);
    fake.pause = Some((2, Duration::from_millis(3)));
    let profile = LayerProfile::measure(&mut fake, &["a", "b"], 5).unwrap();
    let layers: Vec<usize> = profile.times().iter().map(|&(l, _)| l).collect();

    assert_eq!(layers, [0, 1, 2, 3]);
    assert_eq!(fake.texts, ["a", "b", "a", "b", "a"]);
    assert_eq!(fake.copies, 0);
    assert_eq!(profile.last_layer(), 3);
    assert!(profile.at(2).unwrap() - profile.at(1).unwrap() >= 3.0);
    assert!(profile.left_after(1).unwrap() >= 3.0);
    assert!(profile.left_after(2).unwrap() < 3.0);
    assert_eq!(profile.left_after(3), Some(0.0));
    assert_eq!(profile.total_ms(), profile.at(3).unwrap());
}

#[test]
fn the_profile_keeps_each_layers_median_so_a_slow_first_call_drops_out() {
    let mut fake = Fake::new(2);
    fake.pause = Some((1, Duration::from_millis(40)));
    fake.pause_first_only = true;
    let profile = LayerProfile::measure(&mut fake, &["a"], 5).unwrap();

    assert!(profile.at(1).unwrap() < 5.0, "{profile:?}");
    assert_eq!(median(&[5.0, 1.0, 3.0, 2.0, 4.0]), 3.0);
    assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.0);
    assert_eq!(median(&[7.0]), 7.0);
}

#[test]
fn timing_needs_texts_calls_and_a_device_that_runs_its_work_as_issued() {
    let no_texts: [&str; 0] = [];
    let kind =
        |fake: &mut Fake, texts: &[&str], calls: usize| LayerProfile::measure(fake, texts, calls).unwrap_err().kind();

    assert_eq!(kind(&mut Fake::new(2), &no_texts, 5), ErrorKind::InvalidInput);
    assert_eq!(kind(&mut Fake::new(2), &["a"], 0), ErrorKind::InvalidInput);

    let mut queues = Fake::new(2);
    queues.queues = true;
    assert_eq!(kind(&mut queues, &["a"], 5), ErrorKind::Unsupported);
    assert!(queues.texts.is_empty());
}

#[test]
fn an_encoder_that_skips_repeats_or_adds_a_layer_gives_an_error() {
    let mut skips = Fake::new(3);
    skips.skip = Some(1);
    let mut repeats = Fake::new(3);
    repeats.twice = Some(2);
    let mut adds = Fake::new(3);
    adds.past_last = true;

    for mut fake in [skips, repeats, adds] {
        let found = LayerProfile::measure(&mut fake, &["a"], 3).unwrap_err();
        assert_eq!(found.kind(), ErrorKind::Other);
    }
}

#[test]
fn a_profile_needs_increasing_layers_and_times_that_never_fall() {
    let profile = LayerProfile::new(vec![(0, 0.1), (3, 4.0), (9, 12.0), (12, 16.0)]).unwrap();
    assert_eq!(profile.last_layer(), 12);
    assert_eq!(profile.total_ms(), 16.0);
    assert_eq!(profile.at(3), Some(4.0));
    assert_eq!(profile.at(4), None);
    assert_eq!(profile.left_after(9), Some(4.0));
    assert_eq!(profile.left_after(10), None);
    assert!(LayerProfile::new(vec![(0, 0.0), (1, 0.0)]).is_ok());

    let bad: [Vec<(usize, f64)>; 7] = [
        vec![],
        vec![(0, 1.0), (0, 2.0)],
        vec![(3, 1.0), (2, 2.0)],
        vec![(0, 2.0), (1, 1.0)],
        vec![(0, -1.0), (1, 1.0)],
        vec![(0, 0.0), (1, f64::NAN)],
        vec![(0, 0.0), (1, f64::INFINITY)],
    ];

    for times in bad {
        let found = LayerProfile::new(times.clone()).unwrap_err().kind();
        assert_eq!(found, ErrorKind::InvalidInput, "{times:?}");
    }
}

/// Read times on straight lines: `fixed` ms for every chain, then `per_cluster` ms per cluster and `per_row` ms per
/// row. It keeps every (clusters, rows) it is asked about.
struct Line {
    tables: f64,
    fixed: f64,
    per_cluster: f64,
    per_row: f64,
    asked: RefCell<Vec<(usize, usize)>>,
}

impl Line {
    fn new(tables: f64, fixed: f64, per_cluster: f64, per_row: f64) -> Line {
        Line {
            tables,
            fixed,
            per_cluster,
            per_row,
            asked: RefCell::new(Vec::new()),
        }
    }
}

impl ReadTimes for Line {
    fn tables_ms(&self) -> f64 {
        self.tables
    }

    fn chain_ms(&self, clusters: usize, rows: usize) -> f64 {
        self.asked.borrow_mut().push((clusters, rows));
        self.fixed + self.per_cluster * clusters as f64 + self.per_row * rows as f64
    }
}

/// A 12-layer profile that ends at 16 ms and leaves 8, 5, 4 and 2.5 ms after layers 6, 8, 9 and 10.
fn profile() -> LayerProfile {
    LayerProfile::new(vec![
        (0, 0.0625),
        (3, 4.0),
        (6, 8.0),
        (8, 11.0),
        (9, 12.0),
        (10, 13.5),
        (12, 16.0),
    ])
    .unwrap()
}

fn base(nprobe: usize, rerank: usize) -> SearchParams {
    SearchParams {
        k: 10,
        nprobe,
        rerank,
        reach: None,
    }
}

fn settings(layer: usize, beta: f64, rerank_mult: u8) -> LayerSettings {
    LayerSettings {
        layer,
        beta,
        rerank_mult,
    }
}

/// The layers 6, 8, 9 and 10 with the extra clusters and the rows that each needs.
fn layers() -> Vec<LayerSettings> {
    vec![
        settings(6, 0.25, 4),
        settings(8, 0.25, 3),
        settings(9, 0.0, 2),
        settings(10, 0.25, 2),
    ]
}

#[test]
fn the_latest_layer_whose_reads_fit_fires() {
    // Layer 10 reads 11 clusters and 58 rows, 1 + 0.5 + 0.58 = 2.08 ms, and needs 2.288 of its 2.5 ms.
    let fast = Line::new(0.25, 1.0, 1.0 / 22.0, 0.01);
    let schedule = choose(&profile(), &fast, &base(9, 29), &layers(), None).unwrap();
    let fire = schedule.fire.unwrap();

    assert_eq!(fire.settings, settings(10, 0.25, 2));
    assert_eq!((fire.clusters, fire.rows), (11, 58));
    assert_eq!(fire.left_ms, 2.5);
    assert!((fire.needed_ms - 1.1 * 2.08).abs() < 1e-12, "{fire:?}");
    assert!(fire.fits());
    assert_eq!(schedule.layer0_beta, None);
    assert_eq!(fast.asked.borrow().len(), 4);

    // Now layer 10 needs 1.1·(1 + 1.1 + 0.58) = 2.948 ms, but layer 9 reads 9 clusters and fits its 4 ms.
    let slower = Line::new(0.25, 1.0, 0.1, 0.01);
    let fire = choose(&profile(), &slower, &base(9, 29), &layers(), None)
        .unwrap()
        .fire
        .unwrap();
    assert_eq!(fire.settings, settings(9, 0.0, 2));
    assert_eq!((fire.clusters, fire.rows, fire.left_ms), (9, 58, 4.0));
    assert!((fire.needed_ms - 1.1 * (1.0 + 0.9 + 0.58)).abs() < 1e-12);

    let order = choose(
        &profile(),
        &slower,
        &base(9, 29),
        &[layers()[3], layers()[1], layers()[2], layers()[0]],
        None,
    );
    assert_eq!(order.unwrap().fire.unwrap().settings.layer, 9);
}

#[test]
fn each_layer_reads_with_its_own_beta_and_m() {
    let reads = Line::new(0.0, 0.0, 0.0, 0.0);
    choose(&profile(), &reads, &base(9, 29), &layers(), None).unwrap();
    let mut asked = reads.asked.borrow().clone();
    asked.sort_unstable();

    assert_eq!(asked, [(9, 58), (11, 58), (11, 87), (11, 116)]);

    for s in layers() {
        let fire = evaluate(&profile(), &reads, &base(6, 39), &s).unwrap();
        let params = fire.params(base(6, 39));
        assert_eq!(
            (fire.clusters, fire.rows),
            (params.guess_nprobe(), params.prefetch_rows())
        );
        assert_eq!((params.beta, params.rerank_mult), (s.beta, s.rerank_mult));
        assert_eq!((params.base.nprobe, params.base.rerank, params.base.k), (6, 39, 10));
    }
}

#[test]
fn the_tables_read_delays_only_the_layers_it_outlasts() {
    let reads = Line::new(5.0, 1.0, 0.0, 0.0);
    let early = evaluate(&profile(), &reads, &base(9, 29), &settings(3, 0.0, 2)).unwrap();
    let late = evaluate(&profile(), &reads, &base(9, 29), &settings(6, 0.0, 2)).unwrap();

    assert!((early.needed_ms - 1.1 * (1.0 + 1.0)).abs() < 1e-12, "{early:?}");
    assert_eq!(early.left_ms, 12.0);
    assert!((late.needed_ms - 1.1).abs() < 1e-12, "{late:?}");
}

#[test]
fn the_reads_need_a_ten_percent_margin() {
    let profile = LayerProfile::new(vec![(0, 0.0), (9, 8.5), (12, 10.0)]).unwrap();
    let fits = |chain: f64| {
        let reads = Line::new(0.0, chain, 0.0, 0.0);
        let schedule = choose(&profile, &reads, &base(9, 29), &[settings(9, 0.0, 2)], None).unwrap();
        schedule.fire.unwrap().fits()
    };

    assert!(fits(1.3));
    assert!(!fits(1.4));
    assert!(fits(1.5 / 1.1 - 1e-9));
    assert!(!fits(1.5 / 1.1 + 1e-9));

    // Reads that need exactly the time left fit.
    let exact = LayerProfile::new(vec![(0, 0.0), (12, 1.1)]).unwrap();
    let fire = evaluate(
        &exact,
        &Line::new(0.0, 1.0, 0.0, 0.0),
        &base(9, 29),
        &settings(0, 0.0, 2),
    )
    .unwrap();
    assert_eq!(fire.needed_ms, fire.left_ms);
    assert!(fire.fits());
}

#[test]
fn without_time_for_any_layer_the_one_that_runs_over_least_fires_with_a_layer0_wave() {
    // Each chain takes 10 ms plus 1 ms per cluster, so layer 6 runs over by 1.1·21 − 8 = 15.1 ms, layer 8 by
    // 1.1·21 − 5 = 18.1, layer 9 by 1.1·19 − 4 = 16.9 and layer 10 by 1.1·21 − 2.5 = 20.6.
    let slow = Line::new(0.25, 10.0, 1.0, 0.0);
    let schedule = |layer0| choose(&profile(), &slow, &base(9, 29), &layers(), layer0).unwrap();
    let wave = |beta, coverage| Some(Layer0 { beta, coverage });
    let fire = schedule(None).fire.unwrap();

    assert_eq!(fire.settings.layer, 6);
    assert!(!fire.fits());
    assert!((fire.needed_ms - fire.left_ms - 15.1).abs() < 1e-9);
    assert_eq!(schedule(None).layer0_beta, None);
    assert_eq!(schedule(wave(0.0, 0.75)).layer0_beta, Some(1.0));
    assert_eq!(schedule(wave(0.6, 0.7)).layer0_beta, Some(1.0));
    assert_eq!(schedule(wave(1.5, 0.9)).layer0_beta, Some(1.5));
    assert_eq!(schedule(wave(1.0, 0.69)).layer0_beta, None);
    assert_eq!(schedule(wave(1.0, 0.75)).fire, schedule(None).fire);

    // A layer that fits takes no layer 0 wave.
    let fast = Line::new(0.25, 0.0, 0.0, 0.0);
    let schedule = choose(&profile(), &fast, &base(9, 29), &layers(), wave(1.0, 0.9)).unwrap();
    assert_eq!(
        (schedule.fire.unwrap().settings.layer, schedule.layer0_beta),
        (10, None)
    );
}

#[test]
fn layer0_adds_no_wave_of_its_own_when_it_fires() {
    let slow = Line::new(0.0, 30.0, 0.0, 0.0);
    let wave = Some(Layer0 {
        beta: 1.0,
        coverage: 0.9,
    });
    let schedule = choose(
        &profile(),
        &slow,
        &base(9, 29),
        &[settings(0, 1.0, 2), settings(9, 0.0, 2)],
        wave,
    )
    .unwrap();

    assert_eq!(schedule.fire.unwrap().settings.layer, 0);
    assert_eq!(schedule.layer0_beta, None);
}

#[test]
fn a_tie_goes_to_the_later_layer() {
    // Layers 8 and 9 end together and read the same, so their reads run over the model's end by the same time.
    let profile = LayerProfile::new(vec![(0, 0.0), (8, 10.0), (9, 10.0), (12, 12.0)]).unwrap();
    let reads = Line::new(0.0, 5.0, 0.0, 0.0);
    let schedule = choose(
        &profile,
        &reads,
        &base(9, 29),
        &[settings(8, 0.0, 2), settings(9, 0.0, 2)],
        None,
    );
    let fire = schedule.unwrap().fire.unwrap();

    assert_eq!(fire.settings.layer, 9);
    assert!(!fire.fits());
}

#[test]
fn no_layer_that_may_guess_means_no_guess() {
    let reads = Line::new(0.25, 1.0, 0.0, 0.0);
    let wave = Some(Layer0 {
        beta: 1.0,
        coverage: 0.9,
    });

    assert_eq!(
        choose(&profile(), &reads, &base(9, 29), &[], None).unwrap(),
        Schedule::NONE
    );
    assert_eq!(
        choose(&profile(), &reads, &base(9, 29), &[], wave).unwrap(),
        Schedule::NONE
    );
}

#[test]
fn bad_settings_or_read_times_give_invalid_input() {
    let good = Line::new(0.25, 1.0, 0.0, 0.0);
    let wave = |beta, coverage| Some(Layer0 { beta, coverage });
    let kind = |reads: &Line, base: SearchParams, layers: &[LayerSettings], layer0| {
        choose(&profile(), reads, &base, layers, layer0).unwrap_err().kind()
    };
    let cases: [(&Line, SearchParams, Vec<LayerSettings>, Option<Layer0>); 13] = [
        (
            &good,
            base(9, 29),
            vec![settings(9, 0.0, 2), settings(9, 0.25, 3)],
            None,
        ),
        (&good, base(9, 29), vec![settings(7, 0.0, 2)], None),
        (&good, base(9, 29), vec![settings(12, 0.0, 2)], None),
        (&good, base(9, 29), vec![settings(9, f64::NAN, 2)], None),
        (&good, base(9, 29), vec![settings(9, -0.25, 2)], None),
        (&good, base(9, 29), vec![settings(9, 0.0, 0)], None),
        (&good, base(0, 29), vec![settings(9, 0.0, 2)], None),
        (&good, base(9, 0), vec![settings(9, 0.0, 2)], None),
        (&good, base(9, 29), vec![settings(9, 0.0, 2)], wave(1.0, 1.5)),
        (&good, base(9, 29), vec![settings(9, 0.0, 2)], wave(f64::NAN, 0.8)),
        (&good, base(9, 29), vec![settings(9, 0.0, 2)], wave(1.0, f64::NAN)),
        (
            &Line::new(f64::NAN, 1.0, 0.0, 0.0),
            base(9, 29),
            vec![settings(9, 0.0, 2)],
            None,
        ),
        (
            &Line::new(0.25, -1.0, 0.0, 0.0),
            base(9, 29),
            vec![settings(9, 0.0, 2)],
            None,
        ),
    ];

    for (reads, base, layers, layer0) in cases {
        assert_eq!(
            kind(reads, base, &layers, layer0),
            ErrorKind::InvalidInput,
            "{layers:?} {layer0:?}"
        );
    }

    let infinite = Line::new(0.25, f64::INFINITY, 0.0, 0.0);
    assert_eq!(
        kind(&infinite, base(9, 29), &[settings(9, 0.0, 2)], None),
        ErrorKind::InvalidInput
    );
    assert!(choose(&profile(), &good, &base(9, 29), &[settings(9, 0.0, 2)], wave(0.0, 0.0)).is_ok());
    assert!(choose(&profile(), &good, &base(9, 29), &[settings(9, 0.0, 2)], wave(0.0, 1.0)).is_ok());
}

#[test]
fn a_read_model_adds_the_route_both_waves_and_the_scan() {
    let model = ReadModel {
        wave: WaveModel {
            a_ms: 0.25,
            b_ms_per_mb: 0.5,
        },
        tables_bytes: 2_000_000.0,
        route_ms: 0.04,
        cluster_bytes: 20_000.0,
        row_bytes: 16_384.0,
        scan_ms_per_cluster: 0.003,
    };

    assert!((model.tables_ms() - (0.25 + 1.0)).abs() < 1e-12);
    let expected = 0.04 + (0.25 + 0.5 * 9.0 * 0.02) + 9.0 * 0.003 + (0.25 + 0.5 * 58.0 * 0.016384);
    assert!((model.chain_ms(9, 58) - expected).abs() < 1e-12);
    assert!((model.chain_ms(0, 58) - (0.04 + 0.25 + 0.5 * 58.0 * 0.016384)).abs() < 1e-12);
}

#[test]
fn a_schedules_limits_cover_the_search_and_each_of_its_guesses() {
    let base = SearchParams {
        k: 10,
        nprobe: 9,
        rerank: 29,
        reach: None,
    };
    let fire = |beta: f64, rerank_mult: u8| Fire {
        settings: LayerSettings {
            layer: 10,
            beta,
            rerank_mult,
        },
        clusters: 0,
        rows: 0,
        left_ms: 2.0,
        needed_ms: 1.0,
    };
    let limits = |s: Schedule| {
        let l = s.limits(&base);
        (l.max_k, l.max_nprobe, l.max_rerank)
    };
    assert_eq!(limits(Schedule::NONE), (10, 9, 29));
    let guess = Schedule {
        fire: Some(fire(0.5, 3)),
        layer0_beta: None,
    };
    assert_eq!(limits(guess), (10, 13, 87));
    let with_layer0 = Schedule {
        fire: Some(fire(0.25, 2)),
        layer0_beta: Some(1.5),
    };
    assert_eq!(limits(with_layer0), (10, 22, 58));
    let layer0_narrower = Schedule {
        fire: Some(fire(1.5, 2)),
        layer0_beta: Some(1.0),
    };
    assert_eq!(limits(layer0_narrower), (10, 22, 58));
    // A guess that prefetches no rows still leaves room for the rows the search rescores.
    let no_rows = Schedule {
        fire: Some(fire(0.0, 0)),
        layer0_beta: None,
    };
    assert_eq!(limits(no_rows), (10, 9, 29));
}

//! Unit tests for `SpecWave::search`, with an encoder that runs no model and predictors that pass its guesses through,
//! on small index files of random vectors in clusters.

use std::collections::HashMap;
use std::time::Duration;

use coldivf::Limits;

use super::*;
use crate::predictor::Ridge;
use crate::scheduler::{Fire, LayerSettings};
use crate::speculate::{Step, Watcher};
use crate::testutil::{DIM, Indexed, Rng, cosine, indexed, same, searched, unit, within};

/// The model's layers.
const LAYERS: usize = 6;

/// An encoder that runs no model. Each text names a query of the test index by its number, and the final embedding is
/// that query's vector. At each layer that the hook wants, it hands over the guess followed by d zeros, which the
/// pass-through predictors turn back into the guess. The guess is the final embedding unless the test set another.
struct Fake {
    embeddings: Vec<Vec<f32>>,
    /// The guess at a layer of a query, when it isn't the final embedding.
    guesses: HashMap<(usize, usize), Vec<f32>>,
    /// The query and the layer after which the model fails.
    fail: Option<(usize, usize)>,
    /// The layers that the hook wanted, over every call.
    asked: Vec<usize>,
    calls: usize,
    /// Whether it says that its device queues the model's work.
    queues: bool,
    /// A handle on the worker, and what to do with it after each layer named.
    watcher: Option<Watcher>,
    acts: Vec<(usize, Act)>,
}

/// What the encoder does with the worker after a layer.
#[derive(Clone, Copy, PartialEq)]
enum Act {
    /// Waits until the worker has taken the guess just posted.
    WaitTaken,
    /// Lets a held worker go on.
    Release,
}

impl Fake {
    fn new(data: &Indexed) -> Fake {
        Fake {
            embeddings: (0..data.len()).map(|q| data.query(q).to_vec()).collect(),
            guesses: HashMap::new(),
            fail: None,
            asked: Vec::new(),
            calls: 0,
            queues: false,
            watcher: None,
            acts: Vec::new(),
        }
    }
}

impl LayerEncoder for Fake {
    fn layers(&self) -> usize {
        LAYERS
    }

    fn dim(&self) -> usize {
        DIM
    }

    fn encode(&mut self, text: &str, hook: &mut dyn LayerHook) -> Result<Vec<f32>, Error> {
        self.calls += 1;
        let q: usize = text.parse().unwrap();
        let mut pooled = vec![0.0; 2 * DIM];

        for layer in 0..=LAYERS {
            if hook.wants(layer) {
                self.asked.push(layer);
                let guess = self.guesses.get(&(q, layer)).unwrap_or(&self.embeddings[q]);
                pooled[..DIM].copy_from_slice(guess);
                hook.on_layer(layer, &pooled);
            }
            for &(_, act) in self.acts.iter().filter(|(at, _)| *at == layer) {
                let watcher = self.watcher.as_ref().unwrap();
                match act {
                    Act::WaitTaken => watcher.wait_taken(),
                    Act::Release => watcher.release(),
                }
            }
            if self.fail == Some((q, layer)) {
                return Err(Error::other("the model failed"));
            }
        }

        Ok(self.embeddings[q].clone())
    }

    fn queues_work(&self) -> bool {
        self.queues
    }
}

/// Predictors at `layers` that give back the first d values of a summary, scaled to length 1.
fn pass_through(layers: &[usize]) -> Predictors {
    let ridges = layers.iter().map(|&layer| {
        let mut weights = vec![0.0; 2 * DIM * DIM];
        (0..DIM).for_each(|i| weights[i * 2 * DIM + i] = 1.0);
        Ridge::new(layer, DIM, weights, vec![0.0; DIM], 0.0).unwrap()
    });
    Predictors::new("test/pass-through", ridges.collect()).unwrap()
}

/// A schedule that guesses at `layer` with β and m, plus a wave from layer 0 with β `layer0` if given.
fn schedule(layer: usize, beta: f64, rerank_mult: u8, layer0: Option<f64>) -> Schedule {
    let settings = LayerSettings {
        layer,
        beta,
        rerank_mult,
    };
    Schedule {
        fire: Some(Fire {
            settings,
            clusters: 0,
            rows: 0,
            left_ms: 2.0,
            needed_ms: 1.0,
        }),
        layer0_beta: layer0,
    }
}

const BASE: SearchParams = SearchParams {
    k: 10,
    nprobe: 4,
    rerank: 20,
    reach: None,
};

/// SpecWave over `fake` with pass-through predictors at layers 0 and 3, and a worker sized for `schedule`.
fn specwave(fake: Fake, schedule: Schedule) -> SpecWave<Fake> {
    watched(fake, schedule, &[])
}

/// The same, with the encoder holding a handle on the worker and doing `acts` with it.
fn watched(mut fake: Fake, schedule: Schedule, acts: &[(usize, Act)]) -> SpecWave<Fake> {
    let worker = Worker::new(schedule.limits(&BASE)).unwrap();
    fake.watcher = Some(worker.watcher());
    fake.acts = acts.to_vec();
    SpecWave::new(fake, pass_through(&[0, 3]), schedule, worker).unwrap()
}

/// The search with the fire layer's clusters, ⌊(1 + β)·nprobe⌋.
fn wide(beta: f64) -> SearchParams {
    SearchParams {
        nprobe: speculate::wide_nprobe(BASE.nprobe, beta),
        ..BASE
    }
}

#[test]
fn with_perfect_guesses_specwave_gives_the_hits_of_a_search_at_the_wider_nprobe() {
    let data = indexed("lib-perfect", 21);
    let mut sw = specwave(Fake::new(&data), schedule(3, 0.5, 2, None));
    let (mut out, mut wider_differs) = (Vec::new(), 0);

    for q in 0..data.len() {
        sw.search(&q.to_string(), &data.index, &BASE, &mut out).unwrap();
        let e = data.query(q);
        assert!(same(&out, &searched(&data.index, e, &wide(0.5))), "query {q}");
        let report = sw.last();
        assert_eq!(
            (report.ran, report.fire_layer, report.layer0),
            (Ran::Guessed, Some(3), false)
        );
        assert_eq!(report.served.ranked, Ranked::Guess);
        wider_differs += usize::from(!same(&out, &searched(&data.index, e, &BASE)));
        // The search keeps its guess, the embedding at length 1, and the embedding itself.
        assert!(cosine(sw.guess().unwrap(), e) > 1.0 - 1e-6, "query {q}");
        assert_eq!(sw.embedding(), e);
    }

    assert!(wider_differs > 0);
    assert_eq!(sw.encoder().asked, vec![3; data.len()]);
    // Each search had the worker prepare the index with the size of the encoder's embeddings.
    assert_eq!(sw.worker().watcher().prepared(), Some(DIM));
}

#[test]
fn a_warm_index_skips_the_guess_and_gives_a_normal_search() {
    let data = indexed("lib-warm", 22);
    data.index.set_warm(true);
    let mut sw = specwave(Fake::new(&data), schedule(3, 0.5, 2, Some(1.0)));
    let mut out = Vec::new();

    for q in 0..data.len() {
        sw.search(&q.to_string(), &data.index, &BASE, &mut out).unwrap();
        assert!(same(&out, &searched(&data.index, data.query(q), &BASE)), "query {q}");
        let report = sw.last();
        assert_eq!((report.ran, report.fire_layer, report.layer0), (Ran::Warm, None, false));
    }

    assert!(sw.encoder().asked.is_empty());
}

#[test]
fn without_a_fire_layer_or_a_guess_there_the_hits_equal_a_normal_search() {
    let data = indexed("lib-unguessed", 23);
    let mut out = Vec::new();
    let mut never = specwave(Fake::new(&data), Schedule::NONE);
    // A summary of zeros gives no direction, so the predictor gives no guess.
    let mut fake = Fake::new(&data);
    (0..data.len()).for_each(|q| _ = fake.guesses.insert((q, 3), vec![0.0; DIM]));
    let mut no_guess = specwave(fake, schedule(3, 0.5, 2, None));

    for sw in [&mut never, &mut no_guess] {
        for q in 0..data.len() {
            sw.search(&q.to_string(), &data.index, &BASE, &mut out).unwrap();
            assert!(same(&out, &searched(&data.index, data.query(q), &BASE)), "query {q}");
            let report = sw.last();
            assert_eq!((report.ran, report.fire_layer), (Ran::Unguessed, None));
            assert_eq!(report.served.ranked, Ranked::Normal);
            assert!(sw.guess().is_none());
            assert_eq!(sw.embedding(), data.query(q));
        }
    }

    assert!(never.encoder().asked.is_empty());
    assert_eq!(no_guess.encoder().asked.len(), data.len());
}

#[test]
fn a_wave_from_layer0_reads_first_and_the_fire_layers_guess_decides_the_hits() {
    let data = indexed("lib-layer0", 24);
    let mut rng = Rng(25);
    let mut fake = Fake::new(&data);
    let mut no_guess = Fake::new(&data);

    for q in 0..data.len() {
        let far = unit(rng.normals(DIM));
        fake.guesses.insert((q, 0), far.clone());
        no_guess.guesses.insert((q, 0), far);
        no_guess.guesses.insert((q, 3), vec![0.0; DIM]);
    }

    // The encoder waits until the worker has taken the guess from layer 0, so it reads before the next request.
    let mut out = Vec::new();
    let taken = [(0, Act::WaitTaken)];
    let mut both = watched(fake, schedule(3, 0.5, 2, Some(1.0)), &taken);
    let mut layer0_alone = watched(no_guess, schedule(3, 0.5, 2, Some(1.0)), &taken);

    for q in 0..data.len() {
        let e = data.query(q);
        both.search(&q.to_string(), &data.index, &BASE, &mut out).unwrap();
        assert!(same(&out, &searched(&data.index, e, &wide(0.5))), "query {q}");
        let report = both.last();
        assert_eq!(
            (report.ran, report.fire_layer, report.layer0),
            (Ran::Guessed, Some(3), true)
        );
        // Without the fire layer's guess, the guess from layer 0 only read ahead, and a normal search gives the hits.
        layer0_alone
            .search(&q.to_string(), &data.index, &BASE, &mut out)
            .unwrap();
        assert!(same(&out, &searched(&data.index, e, &BASE)), "query {q}");
        let report = layer0_alone.last();
        assert_eq!(
            (report.ran, report.fire_layer, report.layer0),
            (Ran::Unguessed, None, true)
        );
    }

    assert_eq!(both.encoder().asked, [0, 3].repeat(data.len()));

    // A schedule that fires at layer 0 makes no second wave there, so its worker needs no room for one.
    let worker = Worker::new(schedule(0, 0.5, 2, None).limits(&BASE)).unwrap();
    let mut at0 = SpecWave::new(
        Fake::new(&data),
        pass_through(&[0, 3]),
        schedule(0, 0.5, 2, Some(1.0)),
        worker,
    )
    .unwrap();
    at0.search("0", &data.index, &BASE, &mut out).unwrap();
    assert!(same(&out, &searched(&data.index, data.query(0), &wide(0.5))));
    let report = at0.last();
    assert_eq!(
        (report.ran, report.fire_layer, report.layer0),
        (Ran::Guessed, Some(0), false)
    );
}

#[test]
fn a_model_failure_after_the_guess_drops_it_and_later_searches_work() {
    let data = indexed("lib-failure", 26);
    let mut fake = Fake::new(&data);
    fake.fail = Some((1, 4));
    let mut sw = specwave(fake, schedule(3, 0.5, 2, None));
    let mut out = Vec::new();

    for q in 0..data.len() {
        let result = sw.search(&q.to_string(), &data.index, &BASE, &mut out);
        match q {
            1 => {
                assert_eq!(result.unwrap_err().kind(), ErrorKind::Other);
                // The failed search dropped its guess, so the worker has none to finish.
                let error = sw.worker.finish(data.query(q), &mut out).unwrap_err();
                assert_eq!(error.kind(), ErrorKind::InvalidInput);
            }
            _ => {
                result.unwrap();
                assert!(
                    same(&out, &searched(&data.index, data.query(q), &wide(0.5))),
                    "query {q}"
                );
            }
        }
    }
}

#[test]
fn settings_past_the_workers_limits_fail_before_the_model_runs() {
    let data = indexed("lib-limits", 27);
    let small = Limits {
        max_k: BASE.k,
        max_nprobe: BASE.nprobe,
        max_rerank: BASE.rerank * 2,
    };
    let worker = Worker::new(small).unwrap();
    let mut sw = SpecWave::new(Fake::new(&data), pass_through(&[3]), schedule(3, 0.5, 2, None), worker).unwrap();
    let error = sw.search("0", &data.index, &BASE, &mut Vec::new()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(sw.encoder().calls, 0);
}

#[test]
fn new_checks_the_predictors_against_the_schedule_and_the_encoder() {
    let data = indexed("lib-new", 28);
    let limits = schedule(3, 0.5, 2, Some(1.0)).limits(&BASE);
    let make = |predictors: Predictors, schedule: Schedule| {
        SpecWave::new(Fake::new(&data), predictors, schedule, Worker::new(limits).unwrap()).map(|_| ())
    };
    let other_size = Predictors::new(
        "test/small",
        vec![Ridge::new(3, 8, vec![0.0; 128], vec![0.0; 8], 0.0).unwrap()],
    )
    .unwrap();
    let refused = [
        (pass_through(&[0, 2]), schedule(3, 0.5, 2, None)),
        (pass_through(&[3]), schedule(3, 0.5, 2, Some(1.0))),
        (other_size, schedule(3, 0.5, 2, None)),
        (pass_through(&[0, 7]), schedule(7, 0.5, 2, None)),
        (pass_through(&[0, 3]), schedule(3, f64::NAN, 2, None)),
        (pass_through(&[0, 3]), schedule(3, 0.5, 2, Some(-1.0))),
    ];

    for (i, (predictors, schedule)) in refused.into_iter().enumerate() {
        assert_eq!(
            make(predictors, schedule).unwrap_err().kind(),
            ErrorKind::InvalidInput,
            "case {i}"
        );
    }

    make(pass_through(&[0, 3]), schedule(3, 0.5, 2, Some(1.0))).unwrap();
    make(pass_through(&[3]), Schedule::NONE).unwrap();
}

#[test]
fn the_report_orders_each_step_in_time() {
    let data = indexed("lib-report", 29);
    let mut sw = specwave(Fake::new(&data), schedule(3, 0.5, 2, None));
    sw.search("2", &data.index, &BASE, &mut Vec::new()).unwrap();
    let report = *sw.last();
    let times = [report.started, report.fired, report.encoded, report.done];
    assert!(times.iter().all(Option::is_some), "{report:?}");
    assert!(times.windows(2).all(|w| w[0] <= w[1]), "{report:?}");
    let served = report.served;
    assert!(
        report.fired <= served.guess_taken && served.taken <= report.done,
        "{report:?}"
    );
}

#[test]
fn a_search_ran_as_its_index_its_guess_and_the_workers_answer_say() {
    let cases = [
        (true, false, Ranked::Normal, Ran::Warm),
        (true, true, Ranked::Guess, Ran::Warm),
        (false, false, Ranked::Normal, Ran::Unguessed),
        (false, false, Ranked::Fallback, Ran::Unguessed),
        (false, true, Ranked::Guess, Ran::Guessed),
        (false, true, Ranked::Fallback, Ran::Fallback),
    ];

    for (warm, fired, ranked, expected) in cases {
        assert_eq!(ran(warm, fired, ranked), expected, "{warm} {fired} {ranked:?}");
    }
}

#[test]
fn the_wave_from_layer0_reads_only_clusters_with_the_schedules_beta() {
    let data = indexed("lib-layer0-params", 30);
    let mut sw = watched(Fake::new(&data), schedule(3, 0.5, 2, Some(1.5)), &[(0, Act::WaitTaken)]);
    let mut out = Vec::new();

    for q in 0..4 {
        sw.search(&q.to_string(), &data.index, &BASE, &mut out).unwrap();
        assert!(
            same(&out, &searched(&data.index, data.query(q), &wide(0.5))),
            "query {q}"
        );
        assert!(sw.last().layer0);
    }

    // Each search sent the guess from layer 0, then the fire layer's, with the search's own settings.
    let sent: Vec<(usize, usize, f64)> = sw
        .worker()
        .guesses()
        .iter()
        .inspect(|p| assert_eq!((p.base.k, p.base.nprobe, p.base.rerank), (10, 4, 20)))
        .map(|p| (p.guess_nprobe(), p.prefetch_rows(), p.beta))
        .collect();
    assert_eq!(sent, [(10, 0, 1.5), (6, 40, 0.5)].repeat(4));
}

#[test]
fn the_report_says_whether_the_worker_took_the_guess_from_layer0_before_the_next_request() {
    within(Duration::from_secs(60), || {
        let data = indexed("lib-layer0-taken", 31);
        let mut out = Vec::new();

        // The worker is held until the fire layer, whose guess replaces the one from layer 0 untaken.
        let mut held = watched(Fake::new(&data), schedule(3, 0.5, 2, Some(1.0)), &[(3, Act::Release)]);
        held.worker().watcher().hold(Step::Take);
        held.search("0", &data.index, &BASE, &mut out).unwrap();
        assert!(same(&out, &searched(&data.index, data.query(0), &wide(0.5))));
        assert_eq!((held.last().ran, held.last().layer0), (Ran::Guessed, false));

        // Without a guess at the fire layer, the search with the final embedding replaces it, here while the worker
        // is held until that search is posted.
        let mut fake = Fake::new(&data);
        fake.guesses.insert((1, 3), vec![0.0; DIM]);
        let mut unfired = watched(fake, schedule(3, 0.5, 2, Some(1.0)), &[]);
        let watcher = unfired.worker().watcher();
        watcher.hold(Step::Take);
        std::thread::scope(|s| {
            s.spawn(|| {
                watcher.wait_search();
                watcher.release();
            });
            unfired.search("1", &data.index, &BASE, &mut out).unwrap();
        });
        assert!(same(&out, &searched(&data.index, data.query(1), &BASE)));
        assert_eq!((unfired.last().ran, unfired.last().layer0), (Ran::Unguessed, false));

        // Taken before the next request, it read first, with a fire layer and without.
        let mut fake = Fake::new(&data);
        fake.guesses.insert((3, 3), vec![0.0; DIM]);
        let mut taken = watched(fake, schedule(3, 0.5, 2, Some(1.0)), &[(0, Act::WaitTaken)]);

        for (q, ran) in [(2, Ran::Guessed), (3, Ran::Unguessed)] {
            taken.search(&q.to_string(), &data.index, &BASE, &mut out).unwrap();
            assert_eq!((taken.last().ran, taken.last().layer0), (ran, true), "query {q}");
        }
    });
}

#[test]
fn an_encoder_that_queues_its_work_may_not_guess() {
    let data = indexed("lib-queued", 32);
    let queued = || {
        let mut fake = Fake::new(&data);
        fake.queues = true;
        fake
    };
    let make = |schedule: Schedule| {
        let worker = Worker::new(schedule.limits(&BASE)).unwrap();
        SpecWave::new(queued(), pass_through(&[0, 3]), schedule, worker)
    };
    let layer0_alone = Schedule {
        fire: None,
        layer0_beta: Some(1.0),
    };

    for (i, schedule) in [schedule(3, 0.5, 2, None), layer0_alone].into_iter().enumerate() {
        assert_eq!(make(schedule).err().unwrap().kind(), ErrorKind::Unsupported, "case {i}");
    }

    // Without guesses it searches after the model, as usual.
    let mut sw = make(Schedule::NONE).unwrap();
    let mut out = Vec::new();
    sw.search("0", &data.index, &BASE, &mut out).unwrap();
    assert!(same(&out, &searched(&data.index, data.query(0), &BASE)));
    assert_eq!(sw.last().ran, Ran::Unguessed);
}

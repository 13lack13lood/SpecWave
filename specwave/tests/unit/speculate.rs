//! Unit tests for the clusters and rows that a guess reads, and for the worker, on small index files of random vectors
//! in clusters.

use std::time::Duration;

use super::*;
use crate::testutil::{DIM, Indexed, Rng, indexed, same, searched, unit, within};

fn params(nprobe: usize, rerank: usize, beta: f64, rerank_mult: u8) -> SpecParams {
    SpecParams {
        base: SearchParams {
            k: 10,
            nprobe,
            rerank,
            reach: None,
        },
        beta,
        rerank_mult,
    }
}

/// Limits that cover every setting of the worker tests.
const LIMITS: Limits = Limits {
    max_k: 20,
    max_nprobe: 24,
    max_rerank: 160,
};

/// The search with the guess's clusters, ⌊(1 + beta)·nprobe⌋, and otherwise the guess's settings.
fn wide(p: &SpecParams) -> SearchParams {
    SearchParams {
        nprobe: p.guess_nprobe(),
        ..p.base
    }
}

/// The hits of ranking with `embedding` the clusters that `guess` reads with `p`, through ColdIVF alone.
fn ranked(index: &Index, guess: &[f32], embedding: &[f32], p: &SpecParams) -> Vec<Hit> {
    let mut ctx = SearchCtx::new(LIMITS);
    let mut leaves = Vec::new();
    index
        .route(guess, p.guess_nprobe(), p.base.reach, &mut ctx, &mut leaves)
        .unwrap();
    let fetched = index.fetch(&leaves, &mut ctx).unwrap();
    fetched.search(embedding, p.base.k, p.base.rerank).unwrap().to_vec()
}

/// A random direction, which as a guess reads clusters that mostly miss the query's.
fn far(rng: &mut Rng) -> Vec<f32> {
    unit(rng.normals(DIM))
}

#[test]
fn a_guess_reads_the_extra_clusters_rounded_down_and_m_times_the_rows() {
    let cases = [
        (6, 0.25, 7),
        (9, 0.0, 9),
        (9, 0.5, 13),
        (11, 0.25, 13),
        (11, 0.5, 16),
        (20, 0.15, 23),
        (56, 0.25, 70),
        (100, 1.0, 200),
        (128, 0.7, 217),
    ];

    for (nprobe, beta, clusters) in cases {
        assert_eq!(wide_nprobe(nprobe, beta), clusters, "nprobe {nprobe}, beta {beta}");
        assert_eq!(params(nprobe, 29, beta, 2).guess_nprobe(), clusters);
    }

    assert_eq!(params(9, 29, 0.0, 2).prefetch_rows(), 58);
    assert_eq!(params(9, 39, 0.0, 3).prefetch_rows(), 117);
    assert_eq!(params(9, 39, 0.0, 1).prefetch_rows(), 39);
}

#[test]
fn the_extra_clusters_never_pass_beta() {
    for nprobe in 1..300 {
        for beta in [0.0, 0.1, 0.15, 0.25, 0.3, 0.5, 0.7, 1.0, 1.5] {
            let wide = wide_nprobe(nprobe, beta);
            assert!(
                wide as f64 <= (1.0 + beta) * nprobe as f64 + 1e-6,
                "nprobe {nprobe}, beta {beta}"
            );
            assert!(
                (wide + 1) as f64 > (1.0 + beta) * nprobe as f64 + 1e-6,
                "nprobe {nprobe}, beta {beta}"
            );
        }
    }
}

#[test]
fn with_the_true_embedding_as_its_guess_a_finish_equals_a_search_at_the_wider_nprobe() {
    let data = indexed("worker-perfect", 1);
    let mut worker = Worker::new(LIMITS).unwrap();
    let reach = SpecParams {
        base: SearchParams {
            reach: Some(0.1),
            ..params(8, 20, 0.5, 2).base
        },
        ..params(8, 20, 0.5, 2)
    };
    let settings = [
        params(3, 20, 0.5, 2),
        params(4, 15, 0.25, 3),
        params(6, 10, 0.0, 2),
        params(5, 25, 1.0, 4),
        reach,
    ];
    let (mut out, mut wider_differs) = (Vec::new(), 0);

    for p in &settings {
        for q in 0..data.len() {
            let e = data.query(q);
            worker.speculate(&data.index, e, p).unwrap();
            worker.finish(e, &mut out).unwrap();
            assert!(same(&out, &searched(&data.index, e, &wide(p))), "query {q}, {p:?}");
            assert_eq!(worker.served().ranked, Ranked::Guess);
            wider_differs += usize::from(!same(&out, &searched(&data.index, e, &p.base)));
        }
    }

    // The wider read changes some answers, so the check above tells the two searches apart, and so does the reach.
    assert!(wider_differs > 0);
    let unreached = SearchParams {
        reach: None,
        ..wide(&reach)
    };
    let reach_cuts = (0..data.len()).filter(|&q| {
        let e = data.query(q);
        !same(
            &searched(&data.index, e, &wide(&reach)),
            &searched(&data.index, e, &unreached),
        )
    });
    assert!(reach_cuts.count() > 0);
}

#[test]
fn a_far_guess_still_ranks_its_own_clusters_correctly() {
    let data = indexed("worker-far", 2);
    let mut worker = Worker::new(LIMITS).unwrap();
    let (mut rng, mut out, mut normal_differs) = (Rng(9), Vec::new(), 0);
    let p = params(3, 20, 0.5, 2);

    for q in 0..data.len() {
        let (e, guess) = (data.query(q), far(&mut rng));
        worker.speculate(&data.index, &guess, &p).unwrap();
        worker.finish(e, &mut out).unwrap();
        assert!(same(&out, &ranked(&data.index, &guess, e, &p)), "query {q}");
        assert_eq!(worker.served().ranked, Ranked::Guess);
        normal_differs += usize::from(!same(&out, &searched(&data.index, e, &p.base)));
    }

    assert!(
        normal_differs > data.len() / 2,
        "{normal_differs} of {} answers differ",
        data.len()
    );
}

#[test]
fn a_guess_the_index_cant_take_falls_back_to_a_normal_search() {
    let data = indexed("worker-fallback", 3);
    let mut worker = Worker::new(LIMITS).unwrap();
    let (p, mut out, mut wider_differs) = (params(4, 20, 0.5, 2), Vec::new(), 0);

    for q in 0..data.len() {
        let e = data.query(q);
        let mut nan = e.to_vec();
        nan[5] = f32::NAN;
        for guess in [&e[..DIM - 1], &nan[..]] {
            worker.speculate(&data.index, guess, &p).unwrap();
            worker.finish(e, &mut out).unwrap();
            assert!(same(&out, &searched(&data.index, e, &p.base)), "query {q}");
            assert_eq!(worker.served().ranked, Ranked::Fallback);
        }
        wider_differs += usize::from(!same(&out, &searched(&data.index, e, &wide(&p))));
    }

    // The guess's wider read changes some answers, so the checks above tell a normal search from it.
    assert!(wider_differs > 0);
    let e = data.query(0);

    // A finish with an embedding that the index can't take gives the index's error.
    worker.speculate(&data.index, e, &p).unwrap();
    let error = worker.finish(&e[..DIM - 1], &mut out).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(out.is_empty());

    // The worker goes on as before.
    worker.speculate(&data.index, e, &p).unwrap();
    worker.finish(e, &mut out).unwrap();
    assert!(same(&out, &searched(&data.index, e, &wide(&p))));
}

#[test]
fn a_newer_guess_replaces_one_that_no_finish_claimed() {
    let data = indexed("worker-newer", 4);
    let mut worker = Worker::new(LIMITS).unwrap();
    let (mut rng, mut out, mut first_differs) = (Rng(5), Vec::new(), 0);
    let p = params(3, 20, 0.5, 2);

    for round in 0..3 * data.len() {
        let (e, newer) = (data.query(round % data.len()), far(&mut rng));
        worker.speculate(&data.index, e, &p).unwrap();
        // The newer guess comes at once, a moment later or once the first is read, by turns.
        match round % 3 {
            0 => {}
            1 => std::thread::sleep(Duration::from_micros(200)),
            _ => std::thread::sleep(Duration::from_millis(5)),
        }
        worker.speculate(&data.index, &newer, &p).unwrap();
        worker.finish(e, &mut out).unwrap();
        assert!(same(&out, &ranked(&data.index, &newer, e, &p)), "round {round}");
        first_differs += usize::from(!same(&out, &searched(&data.index, e, &wide(&p))));
    }

    assert!(
        first_differs > data.len(),
        "{first_differs} answers differ from the first guess's"
    );
}

#[test]
fn a_search_a_cancel_or_a_prepare_drops_the_open_guess() {
    let data = indexed("worker-drop", 5);
    let mut worker = Worker::new(LIMITS).unwrap();
    let (p, e, mut out) = (params(3, 20, 1.0, 2), data.query(1), Vec::new());
    let guess = far(&mut Rng(6));

    // A search after a guess ranks its own clusters.
    worker.speculate(&data.index, &guess, &p).unwrap();
    worker.search(&data.index, e, &p.base, &mut out).unwrap();
    assert!(same(&out, &searched(&data.index, e, &p.base)));
    assert_eq!(worker.served().ranked, Ranked::Normal);
    // A finish without its guess never reaches the worker, so the last answer stays the search's.
    let refused = |worker: &mut Worker, out: &mut Vec<Hit>| {
        let before = worker.served();
        let kind = worker.finish(e, out).unwrap_err().kind();
        assert_eq!(worker.served(), before);
        kind
    };
    assert_eq!(refused(&mut worker, &mut out), ErrorKind::InvalidInput);

    // A cancel drops the guess, and so does a prepare, which starts the next query, also once the worker holds the
    // guess's reads.
    for hold in [Duration::ZERO, Duration::from_millis(20)] {
        worker.speculate(&data.index, &guess, &p).unwrap();
        std::thread::sleep(hold);
        worker.cancel();
        assert_eq!(refused(&mut worker, &mut out), ErrorKind::InvalidInput);
        worker.speculate(&data.index, e, &p).unwrap();
        worker.finish(e, &mut out).unwrap();
        assert!(same(&out, &searched(&data.index, e, &wide(&p))));
        worker.speculate(&data.index, &guess, &p).unwrap();
        std::thread::sleep(hold);
        worker.prepare(&data.index, DIM).unwrap();
        assert_eq!(refused(&mut worker, &mut out), ErrorKind::InvalidInput);
        worker.speculate(&data.index, e, &p).unwrap();
        worker.finish(e, &mut out).unwrap();
        assert!(same(&out, &searched(&data.index, e, &wide(&p))));
    }

    // The next guess is served as usual, and a worker that has seen no guess has none to finish.
    worker.speculate(&data.index, e, &p).unwrap();
    worker.finish(e, &mut out).unwrap();
    assert!(same(&out, &searched(&data.index, e, &wide(&p))));
    assert_eq!(
        refused(&mut Worker::new(LIMITS).unwrap(), &mut out),
        ErrorKind::InvalidInput
    );
}

/// Drops guesses in every way, at moments that vary with the round, and checks that the next answer is still right.
fn drop_guesses_mid_wave(data: &Indexed, index: &Index, rounds: usize) {
    let p = params(6, 20, 1.0, 4);
    let (mut rng, mut out) = (Rng(7), Vec::new());
    let mut worker = Worker::new(LIMITS).unwrap();

    for round in 0..rounds {
        let e = data.query(round % data.len());
        // The worker's handle keeps the index open after the caller's goes.
        let held = index.clone();
        worker.speculate(&held, &far(&mut rng), &p).unwrap();
        drop(held);
        if round % 4 == 1 {
            std::thread::sleep(Duration::from_micros(50 * (round as u64 % 7)));
        }
        match round % 5 {
            0 => worker.cancel(),
            1 => worker.prepare(index, DIM).unwrap(),
            2 => worker.speculate(index, e, &p).unwrap(),
            3 => worker = Worker::new(LIMITS).unwrap(),
            _ => {}
        }
        match round % 5 {
            2 => {
                worker.finish(e, &mut out).unwrap();
                assert!(same(&out, &searched(index, e, &wide(&p))), "round {round}");
            }
            _ => {
                worker.search(index, e, &p.base, &mut out).unwrap();
                assert!(same(&out, &searched(index, e, &p.base)), "round {round}");
            }
        }
    }
}

#[test]
fn dropping_a_guess_mid_wave_neither_hangs_nor_changes_later_answers() {
    within(Duration::from_secs(120), || {
        let data = indexed("worker-mid-wave", 6);
        drop_guesses_mid_wave(&data, &data.index, 200);
    });
}

#[cfg(feature = "bench")]
#[test]
fn dropping_a_guess_mid_wave_is_safe_with_every_read_backend() {
    within(Duration::from_secs(120), || {
        let data = indexed("worker-backends", 7);
        for backend in [coldivf::Backend::Pool, coldivf::Backend::Portable] {
            let index = Index::open_with(&data.path(), backend).unwrap();
            drop_guesses_mid_wave(&data, &index, 60);
        }
    });
}

#[test]
fn settings_past_the_limits_are_refused_before_they_reach_the_worker() {
    let data = indexed("worker-limits", 8);
    let limits = Limits {
        max_k: 10,
        max_nprobe: 6,
        max_rerank: 40,
    };
    let mut worker = Worker::new(limits).unwrap();
    let (e, mut out) = (data.query(0), Vec::new());
    let fits = params(5, 20, 0.25, 2);
    let many = SpecParams {
        base: SearchParams { k: 11, ..fits.base },
        ..fits
    };
    let refused = [
        params(5, 20, 0.5, 2),
        params(5, 20, 0.25, 3),
        params(5, 20, f64::NAN, 2),
        params(5, 20, -0.5, 2),
        params(7, 20, 0.0, 1),
        params(5, 41, 0.0, 0),
        many,
    ];

    for p in &refused {
        let error = worker.speculate(&data.index, e, p).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput, "{p:?}");
    }

    // None reached the worker, so it has no guess to finish and has answered nothing.
    assert_eq!(worker.finish(e, &mut out).unwrap_err().kind(), ErrorKind::InvalidInput);
    assert_eq!(worker.served(), Served::default());

    // A refused guess leaves the open one alone.
    worker.speculate(&data.index, e, &fits).unwrap();
    assert!(worker.speculate(&data.index, e, &refused[0]).is_err());
    worker.finish(e, &mut out).unwrap();
    assert!(same(&out, &searched(&data.index, e, &wide(&fits))));
}

#[test]
fn the_rows_are_skipped_without_a_multiple_or_once_the_embedding_is_there() {
    let data = indexed("worker-rows", 9);
    let mut worker = Worker::new(LIMITS).unwrap();
    let (e, mut out) = (data.query(2), Vec::new());

    worker.speculate(&data.index, e, &params(4, 20, 0.5, 0)).unwrap();
    worker.finish(e, &mut out).unwrap();
    assert!(worker.served().skipped_rows);
    assert!(same(&out, &searched(&data.index, e, &wide(&params(4, 20, 0.5, 0)))));

    // A finish once the rows are chosen finds them prefetched. The guess is ready only once its reads started, which
    // here is at least 5 ms after the worker took it.
    let (p, watcher) = (params(4, 20, 0.5, 2), worker.watcher());
    watcher.hold(Step::Fetched);
    worker.speculate(&data.index, e, &p).unwrap();
    watcher.wait_held(Step::Fetched);
    std::thread::sleep(Duration::from_millis(5));
    watcher.hold(Step::Ready);
    watcher.wait_held(Step::Ready);
    watcher.release();
    worker.finish(e, &mut out).unwrap();
    let served = worker.served();
    assert!(!served.skipped_rows);
    let times = [served.guess_taken, served.guess_ready, served.taken, served.done];
    assert!(times.iter().all(Option::is_some), "{served:?}");
    assert!(times.windows(2).all(|w| w[0] <= w[1]), "{served:?}");
    let chain = served.guess_ready.unwrap() - served.guess_taken.unwrap();
    assert!(chain >= Duration::from_millis(5), "{served:?}");

    // A finish right after its guess mostly comes before the rows are chosen, which the worker then skips.
    let mut skipped = 0;

    for _ in 0..50 {
        worker.speculate(&data.index, e, &p).unwrap();
        worker.finish(e, &mut out).unwrap();
        assert!(same(&out, &searched(&data.index, e, &wide(&p))));
        skipped += usize::from(worker.served().skipped_rows);
    }

    assert!(skipped > 0);

    // A normal search has no guess.
    worker.search(&data.index, e, &p.base, &mut out).unwrap();
    let served = worker.served();
    assert_eq!(
        (served.ranked, served.guess_taken, served.guess_ready),
        (Ranked::Normal, None, None)
    );
    assert!(served.taken.unwrap() <= served.done.unwrap());
}

#[cfg(feature = "bench")]
#[test]
fn a_perfect_guess_prefetches_every_row_that_its_finish_rescores() {
    let data = indexed("worker-prefetch", 12);
    let mut worker = Worker::new(LIMITS).unwrap();
    let mut out = Vec::new();

    for q in 0..data.len() {
        let e = data.query(q);
        for (mult, prefetched) in [(2, true), (0, false)] {
            worker.speculate(&data.index, e, &params(4, 20, 0.5, mult)).unwrap();
            std::thread::sleep(Duration::from_millis(20));
            worker.finish(e, &mut out).unwrap();
            // The guess ranks its clusters as the embedding does, so its best m·R hold the R that the finish rescores.
            let rows = worker.trace().row_bytes;
            assert_eq!(
                rows == 0,
                prefetched,
                "query {q}, m {mult}, {rows} row bytes read at the finish"
            );
        }
    }
}

#[test]
fn callers_get_an_error_rather_than_wait_once_the_worker_thread_is_gone() {
    within(Duration::from_secs(60), || {
        let data = indexed("worker-gone", 10);
        let (p, e, mut out) = (params(4, 20, 0.5, 2), data.query(0), Vec::new());

        // The worker dies serving a search that a caller waits for.
        let mut worker = Worker::new(LIMITS).unwrap();
        worker.search(&data.index, e, &p.base, &mut out).unwrap();
        worker.panic_on_next();
        assert_eq!(
            worker.search(&data.index, e, &p.base, &mut out).unwrap_err().kind(),
            ErrorKind::Other
        );
        assert!(worker.gone());
        assert!(worker.prepare(&data.index, DIM).is_err());
        assert!(worker.speculate(&data.index, e, &p).is_err());
        assert!(worker.search(&data.index, e, &p.base, &mut out).is_err());

        // It dies taking the finish of an open guess.
        let mut worker = Worker::new(LIMITS).unwrap();
        worker.speculate(&data.index, e, &p).unwrap();
        worker.finish(e, &mut out).unwrap();
        worker.speculate(&data.index, e, &p).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        worker.panic_on_next();
        assert_eq!(worker.finish(e, &mut out).unwrap_err().kind(), ErrorKind::Other);
    });
}

#[cfg(target_os = "macos")]
#[test]
fn the_worker_thread_runs_at_the_user_interactive_class() {
    let data = indexed("worker-qos", 11);
    let mut worker = Worker::new(LIMITS).unwrap();
    // An answer shows that the thread has started.
    let p = params(4, 20, 0.5, 2);
    worker
        .search(&data.index, data.query(0), &p.base, &mut Vec::new())
        .unwrap();
    assert_eq!(worker.qos(), Some(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE as u32));
}

#[test]
fn a_newer_request_ends_a_guess_after_its_route_or_after_its_fetch() {
    within(Duration::from_secs(120), || {
        let data = indexed("worker-steps", 13);
        let mut worker = Worker::new(LIMITS).unwrap();
        let watcher = worker.watcher();
        let (e, guess, mut out) = (data.query(3), far(&mut Rng(14)), Vec::new());
        // The first guess would prefetch rows, and the newer one reads only clusters.
        let (p, newer) = (params(4, 20, 0.5, 2), params(4, 20, 0.5, 0));

        for at in [Step::Routed, Step::Fetched] {
            for request in ["prepare", "cancel", "guess", "search"] {
                let (fetched, prefetched) = (watcher.count(Step::Fetched), watcher.count(Step::Prefetched));
                watcher.hold(at);
                worker.speculate(&data.index, &guess, &p).unwrap();
                watcher.wait_held(at);
                std::thread::scope(|s| {
                    // A search waits for its answer, so the worker goes on once the request was posted.
                    s.spawn(|| {
                        watcher.wait_posted();
                        watcher.release();
                    });
                    match request {
                        "prepare" => worker.prepare(&data.index, DIM).unwrap(),
                        "cancel" => worker.cancel(),
                        "guess" => worker.speculate(&data.index, e, &newer).unwrap(),
                        _ => worker.search(&data.index, e, &p.base, &mut out).unwrap(),
                    }
                });
                match request {
                    "guess" => {
                        worker.finish(e, &mut out).unwrap();
                        assert!(
                            same(&out, &searched(&data.index, e, &wide(&newer))),
                            "{at:?}, {request}"
                        );
                    }
                    _ => {
                        worker.search(&data.index, e, &p.base, &mut out).unwrap();
                        assert!(same(&out, &searched(&data.index, e, &p.base)), "{at:?}, {request}");
                    }
                }
                // The first guess fetched only if it was held after its fetch, and it read no rows. A newer guess
                // fetched once.
                let fetches = usize::from(at == Step::Fetched) + usize::from(request == "guess");
                assert_eq!(watcher.count(Step::Fetched) - fetched, fetches, "{at:?}, {request}");
                assert_eq!(watcher.count(Step::Prefetched), prefetched, "{at:?}, {request}");
            }
        }
    });
}

#[test]
fn a_prepare_drops_a_guess_that_the_worker_has_not_taken() {
    within(Duration::from_secs(60), || {
        let data = indexed("worker-stale", 15);
        let mut worker = Worker::new(LIMITS).unwrap();
        let watcher = worker.watcher();
        let (e, guess, mut out) = (data.query(4), far(&mut Rng(16)), Vec::new());
        let p = params(4, 20, 0.5, 2);
        // The worker takes nothing until both are posted. Then it runs the prepare, and a guess left behind next.
        watcher.hold(Step::Take);
        worker.speculate(&data.index, &guess, &p).unwrap();
        worker.prepare(&data.index, DIM).unwrap();
        watcher.release();
        watcher.wait_taken();
        worker.search(&data.index, e, &p.base, &mut out).unwrap();
        assert!(same(&out, &searched(&data.index, e, &p.base)));
        assert_eq!(watcher.count(Step::Routed), 0);
    });
}

#[test]
fn a_guess_that_replaces_one_the_worker_has_not_taken_says_so() {
    within(Duration::from_secs(60), || {
        let data = indexed("worker-replaced", 17);
        let mut worker = Worker::new(LIMITS).unwrap();
        let watcher = worker.watcher();
        let (e, mut out) = (data.query(5), Vec::new());
        let p = params(4, 20, 0.5, 2);
        watcher.hold(Step::Take);
        worker.speculate(&data.index, &far(&mut Rng(18)), &p).unwrap();
        assert!(!worker.replaced_guess());
        worker.speculate(&data.index, e, &p).unwrap();
        assert!(worker.replaced_guess());
        watcher.release();
        worker.finish(e, &mut out).unwrap();
        assert!(same(&out, &searched(&data.index, e, &wide(&p))));
        assert_eq!(watcher.count(Step::Routed), 1);

        // Once the worker has taken a guess, the next request replaces nothing.
        worker.speculate(&data.index, e, &p).unwrap();
        watcher.wait_taken();
        worker.search(&data.index, e, &p.base, &mut out).unwrap();
        assert!(!worker.replaced_guess());
    });
}

#[test]
fn a_finish_whose_reads_fail_gives_the_hits_of_a_normal_search() {
    let data = indexed("worker-finish-fails", 19);
    let mut worker = Worker::new(LIMITS).unwrap();
    let watcher = worker.watcher();
    let (p, mut out, mut wider_differs) = (params(4, 20, 0.5, 2), Vec::new(), 0);

    for q in 0..data.len() {
        let e = data.query(q);
        worker.speculate(&data.index, e, &p).unwrap();
        watcher.fail_next_finish();
        worker.finish(e, &mut out).unwrap();
        assert!(same(&out, &searched(&data.index, e, &p.base)), "query {q}");
        assert_eq!(worker.served().ranked, Ranked::Fallback);
        wider_differs += usize::from(!same(&out, &searched(&data.index, e, &wide(&p))));
    }

    // The guess's wider read changes some answers, so the checks above tell a normal search from it.
    assert!(wider_differs > 0);
}

#[test]
fn a_prepare_with_the_wrong_size_still_leaves_the_guess_right() {
    let data = indexed("worker-wrong-size", 20);
    let mut worker = Worker::new(LIMITS).unwrap();
    let (p, e, mut out) = (params(4, 20, 0.5, 2), data.query(6), Vec::new());

    for dim in [DIM + 1, 0, DIM] {
        worker.prepare(&data.index, dim).unwrap();
        worker.speculate(&data.index, e, &p).unwrap();
        worker.finish(e, &mut out).unwrap();
        assert!(same(&out, &searched(&data.index, e, &wide(&p))), "{dim}");
    }
}

#[test]
fn callers_get_an_error_rather_than_wait_when_the_worker_thread_dies_as_it_starts() {
    within(Duration::from_secs(20), || {
        let data = indexed("worker-start", 21);
        // A context for this many clusters can't be made, so the thread panics before it serves anything.
        let limits = Limits {
            max_k: 10,
            max_nprobe: usize::MAX / 2,
            max_rerank: 40,
        };
        let mut worker = Worker::new(limits).unwrap();
        let p = params(4, 20, 0.5, 2);
        let error = worker
            .search(&data.index, data.query(0), &p.base, &mut Vec::new())
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Other);
        assert!(worker.gone());
    });
}

#[cfg(feature = "bench")]
#[test]
fn a_guess_prefetches_the_rows_of_its_best_m_times_r_candidates() {
    let data = indexed("worker-prefetch-m", 22);
    let mut worker = Worker::new(LIMITS).unwrap();
    let watcher = worker.watcher();
    let (mut rng, mut out, mut deeper_helps) = (Rng(23), Vec::new(), 0);

    for q in 0..data.len() {
        let e = data.query(q);
        // A guess near the embedding ranks the candidates a little differently, so its best R rows miss some of the
        // embedding's.
        let guess = unit(e.iter().map(|&x| x + (NOISE * rng.normal()) as f32).collect());
        let mut rows = [0; 2];
        for (m, read) in [1, 2].into_iter().zip(&mut rows) {
            watcher.hold(Step::Ready);
            worker.speculate(&data.index, &guess, &params(4, 20, 0.5, m)).unwrap();
            watcher.wait_held(Step::Ready);
            watcher.release();
            worker.finish(e, &mut out).unwrap();
            assert!(!worker.served().skipped_rows, "query {q}, m {m}");
            *read = worker.trace().row_bytes;
        }
        // The best 2R candidates hold the best R, so the deeper prefetch misses no row that the shallower one has.
        assert!(rows[1] <= rows[0], "query {q}, {rows:?}");
        deeper_helps += usize::from(rows[0] > 0 && rows[1] == 0);
    }

    assert!(
        deeper_helps >= data.len() / 4,
        "{deeper_helps} of {} queries",
        data.len()
    );
}

/// The noise of each value of the guesses near the embeddings, which leaves a cosine of ~0.99 with the embedding.
#[cfg(feature = "bench")]
const NOISE: f64 = 0.015;

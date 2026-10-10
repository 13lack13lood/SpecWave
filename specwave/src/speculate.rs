//! Speculation, built on ColdIVF's generic `route`, `fetch` and `Fetched`. ColdIVF knows nothing about SpecWave.
//!
//! ```text
//! text arrives:  index.prepare(), index.route(ones, 1), index.fetch(no clusters)                       (on the worker)
//! fire layer:    guess → index.route(guess, ⌊(1 + beta)·nprobe⌋) → index.fetch(clusters)               (on the worker)
//!                fetched.prefetch_rows(guess, rerank_mult·rerank)                          (the likely rescoring rows)
//! model ends:    fetched.search(embedding, k, rerank) → hits                                           (on the worker)
//! ```
//!
//! The worker never reads a cluster the guess missed, because the extra clusters already make up for those. It is one
//! thread with a search context of its own, parked between queries at the user-interactive Quality of Service (QoS)
//! class, so the model's thread never waits on the disk. The model's thread hands it each request through one slot,
//! a mutex with a condition variable for each direction. A request copies its vector into the slot, so the caller's
//! buffers are free as soon as it returns, and once every buffer has grown to size a query allocates nothing.
//!
//! When the text arrives, a prepare readies the index while the model runs. It loads the cluster tables and makes the
//! buffers that the guess's reads need, so the guess at the fire layer only routes and reads.
//!
//! The worker serves one request at a time, in order. Any newer request but a finish ends a guess. The worker looks
//! for one after the guess's route, after its fetch and once its rows are chosen, and drops the guess's reads when it
//! finds one. One that comes while the guess's clusters arrive and are scanned waits until the rows are chosen. A
//! dropped [`coldivf::Fetched`] cancels or finishes the reads still under way. When the final embedding is there before
//! the guess's rows are chosen, the worker skips them, since the final search reads its own rows just as soon. When a
//! guess's reads fail, eg. because the index moved to a newer generation between the route and the fetch, or the
//! finish's reads of its clusters and rows fail, the finish runs a normal search instead, so the hits are still right.

use std::io::{Error, ErrorKind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Instant;

use coldivf::{Hit, Index, LeafId, Limits, SearchCtx, SearchParams};

#[derive(Clone, Copy, Debug)]
pub struct SpecParams {
    pub base: SearchParams,
    /// Extra clusters to read, as a fraction of nprobe, eg. 0 to 0.25 for arctic-embed-m at layers 8 to 10. The
    /// schedule picks it for each layer that may guess.
    pub beta: f64,
    /// How many candidates' precise copies to prefetch, as a multiple of `rerank`, 2 to 4. 0 prefetches none, eg. for
    /// a guess that only starts reading clusters.
    pub rerank_mult: u8,
}

impl SpecParams {
    /// The clusters that a guess reads, ⌊(1 + beta)·nprobe⌋.
    pub fn guess_nprobe(&self) -> usize {
        wide_nprobe(self.base.nprobe, self.beta)
    }

    /// The candidates whose precise copies a guess prefetches, rerank_mult·rerank.
    pub fn prefetch_rows(&self) -> usize {
        usize::from(self.rerank_mult) * self.base.rerank
    }
}

/// ⌊(1 + beta)·nprobe⌋, the clusters that a guess reads for a base `nprobe`. Rounding down keeps the share of extra
/// clusters within `beta`. The floor allows for the rounding error of the product, so eg. 1.15·20 gives 23.
pub fn wide_nprobe(nprobe: usize, beta: f64) -> usize {
    ((1.0 + beta) * nprobe as f64 + 1e-9).floor() as usize
}

/// The clusters that the worker's last answer ranked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Ranked {
    /// Those of a normal search, asked for with [`Worker::search`].
    #[default]
    Normal,
    /// Those that the guess read.
    Guess,
    /// Those of a normal search, since the guess's reads failed.
    Fallback,
}

/// What the worker did for its last answer, and when.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Served {
    pub ranked: Ranked,
    /// Whether the guess prefetched no rows, since its multiple was 0 or the final embedding came first.
    pub skipped_rows: bool,
    /// When the worker took the guess, and when it had started the guess's reads. That is once the clusters arrived and
    /// their scan started the row prefetch, or once the cluster reads started when the rows were skipped.
    pub guess_taken: Option<Instant>,
    pub guess_ready: Option<Instant>,
    /// When the worker took the final embedding, and when the hits were ready.
    pub taken: Option<Instant>,
    pub done: Option<Instant>,
}

/// The thread that runs the guessed half of a query. It has its own `SearchCtx`, sized for (1 + beta)·nprobe
/// clusters.
pub struct Worker {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
    limits: Limits,
    /// The number of the last request posted.
    posted: u64,
    /// Whether a guess is open, i.e. `speculate` ran and nothing has finished or dropped the guess since.
    open: bool,
    /// Whether the last request posted replaced a guess that the worker hadn't taken, so that guess read nothing.
    replaced: bool,
    /// What the last answer did, and the trace of its search.
    served: Served,
    #[cfg(feature = "bench")]
    trace: coldivf::Trace,
    /// The settings of every guess posted, for the tests.
    #[cfg(test)]
    guesses: Vec<SpecParams>,
}

impl Worker {
    /// Starts the worker thread, with a search context sized for `limits`. They must cover each guess, i.e. the
    /// clusters it reads and the rows it prefetches, and the searches that finish it. A thread that can't start gives
    /// its error. A thread that ends early, eg. in a panic while it makes its search context, leaves callers an error
    /// rather than a wait.
    pub fn new(limits: Limits) -> Result<Worker, Error> {
        let shared = Arc::new(Shared {
            slot: Mutex::new(Slot::new(limits)),
            requests: Condvar::new(),
            answers: Condvar::new(),
        });
        let theirs = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("specwave-worker".to_string())
            .spawn(move || {
                let _gone = Gone(Arc::clone(&theirs));
                Thread::new(theirs, limits).run();
            })?;
        Ok(Worker {
            shared,
            thread: Some(thread),
            limits,
            posted: 0,
            open: false,
            replaced: false,
            served: Served::default(),
            #[cfg(feature = "bench")]
            trace: coldivf::Trace::default(),
            #[cfg(test)]
            guesses: Vec::new(),
        })
    }

    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// Asks the worker to ready `index` for the next query and returns right away, so the worker does it while the
    /// model runs. On its context the worker calls `index.prepare`, which starts reading the cluster tables. Then it
    /// routes a vector of `dim` ones to one cluster and fetches no cluster, where `dim` is the size of the index's
    /// vectors. That loads the tables and makes the buffer for the clusters, which is sized from them, so the guess
    /// that follows only routes and reads. It starts the next query, so it drops any open guess. A step that fails
    /// only loses the head start, since a search reads what it needs itself.
    pub fn prepare(&mut self, index: &Index, dim: usize) -> Result<(), Error> {
        let mut slot = self.shared.lock();

        if slot.gone {
            return Err(stopped());
        }

        slot.prepare = Some((index.clone(), dim));

        // A guess or a cancel that the worker hasn't taken belongs to the query before.
        if matches!(slot.request, Some((_, Request::Speculate { .. } | Request::Cancel))) {
            slot.request = None;
        }

        drop(slot);
        self.open = false;
        self.shared.requests.notify_one();
        Ok(())
    }

    /// Hands a guessed embedding to the worker and returns right away. The worker picks the clusters, starts
    /// reading them and prefetches the likely rescoring rows. A guess that the worker hasn't finished replaces it.
    /// Settings beyond the worker's limits, or a β that isn't finite or is below 0, give `InvalidInput`.
    pub fn speculate(&mut self, index: &Index, guess: &[f32], params: &SpecParams) -> Result<(), Error> {
        self.check(params)?;
        let request = Request::Speculate {
            index: index.clone(),
            params: *params,
        };
        self.post(request, Some(guess))?;
        self.open = true;
        #[cfg(test)]
        self.guesses.push(*params);
        Ok(())
    }

    /// Finishes with the true embedding. It ranks only the clusters already read, and copies the hits into `out`.
    /// Without an open guess it gives `InvalidInput`.
    pub fn finish(&mut self, embedding: &[f32], out: &mut Vec<Hit>) -> Result<(), Error> {
        if !self.open {
            return Err(invalid(
                "there is no guess to finish, since none was made or it was dropped",
            ));
        }

        self.open = false;
        let n = self.post(Request::Finish, Some(embedding))?;
        self.answer(n, out)
    }

    /// Drops the open guess, if any, and returns right away, eg. after the model failed. The worker drops the guess's
    /// reads once it is between steps.
    pub fn cancel(&mut self) {
        if self.open {
            self.open = false;
            let _ = self.post(Request::Cancel, None);
        }
    }

    /// Runs a normal `index.search` on the worker's context, for when no layer fired. It drops any open guess.
    pub fn search(
        &mut self,
        index: &Index,
        embedding: &[f32],
        params: &SearchParams,
        out: &mut Vec<Hit>,
    ) -> Result<(), Error> {
        self.open = false;
        let request = Request::Search {
            index: index.clone(),
            params: *params,
        };
        let n = self.post(request, Some(embedding))?;
        self.answer(n, out)
    }

    /// What the worker did for its last answer.
    pub fn served(&self) -> Served {
        self.served
    }

    /// Where the search of the worker's last answer spent its time, and what it read. A finish's trace starts at the
    /// guess's route and holds its cluster wave.
    #[cfg(feature = "bench")]
    pub fn trace(&self) -> coldivf::Trace {
        self.trace
    }

    /// Checks a guess's settings against the worker's limits.
    pub fn check(&self, params: &SpecParams) -> Result<(), Error> {
        let (base, limits) = (&params.base, &self.limits);

        if !(params.beta.is_finite() && params.beta >= 0.0) {
            return Err(invalid("beta must be finite and at least 0"));
        }

        // β is at least 0, so a guess reads at least nprobe clusters, and the limit on its clusters covers nprobe.
        if params.guess_nprobe() > limits.max_nprobe
            || params.prefetch_rows() > limits.max_rerank
            || base.k > limits.max_k
            || base.rerank > limits.max_rerank
        {
            return Err(invalid("the guess's settings exceed the worker's limits"));
        }

        Ok(())
    }

    /// Whether the last request posted replaced a guess that the worker hadn't taken, so that guess read nothing.
    pub(crate) fn replaced_guess(&self) -> bool {
        self.replaced
    }

    /// Puts `request` in the slot, with `vector` copied in, and wakes the worker. It replaces a guess or a cancel that
    /// the worker hasn't taken, but a finish waits until the worker has taken the guess it finishes. Returns the
    /// request's number.
    fn post(&mut self, request: Request, vector: Option<&[f32]>) -> Result<u64, Error> {
        let mut slot = self.shared.lock();

        if matches!(request, Request::Finish) {
            while slot.request.is_some() && !slot.gone {
                slot = self.shared.answers.wait(slot).unwrap_or_else(PoisonError::into_inner);
            }
        }

        if slot.gone {
            return Err(stopped());
        }

        if let Some(vector) = vector {
            slot.vector.clear();
            slot.vector.extend_from_slice(vector);
        }

        self.posted += 1;
        self.replaced = matches!(slot.request, Some((_, Request::Speculate { .. })));
        slot.request = Some((self.posted, request));
        drop(slot);
        self.shared.requests.notify_one();
        Ok(self.posted)
    }

    /// Waits for the answer to request `n` and copies its hits into `out`.
    fn answer(&mut self, n: u64, out: &mut Vec<Hit>) -> Result<(), Error> {
        let mut slot = self.shared.lock();

        while slot.answered < n && !slot.gone {
            slot = self.shared.answers.wait(slot).unwrap_or_else(PoisonError::into_inner);
        }

        if slot.answered < n {
            return Err(stopped());
        }

        self.served = slot.served;
        out.clear();

        #[cfg(feature = "bench")]
        {
            self.trace = slot.trace;
        }

        match slot.error.take() {
            Some(e) => Err(e),
            None => {
                out.extend_from_slice(&slot.hits);
                Ok(())
            }
        }
    }

    /// The tests' handle on the worker thread.
    #[cfg(test)]
    pub(crate) fn watcher(&self) -> Watcher {
        Watcher(Arc::clone(&self.shared))
    }

    /// The settings of every guess posted so far.
    #[cfg(test)]
    pub(crate) fn guesses(&self) -> &[SpecParams] {
        &self.guesses
    }

    /// Makes the worker thread panic as it takes its next request, so tests can check that callers get an error
    /// rather than wait forever.
    #[cfg(test)]
    fn panic_on_next(&mut self) {
        self.shared.lock().watch.panic_on_next = true;
    }

    /// Whether the worker thread has ended.
    #[cfg(test)]
    fn gone(&self) -> bool {
        self.shared.lock().gone
    }

    /// The QoS class that the worker thread runs at, once it has started.
    #[cfg(all(test, target_os = "macos"))]
    fn qos(&self) -> Option<u32> {
        self.shared.lock().watch.qos
    }
}

impl Drop for Worker {
    /// Stops the worker thread and waits for it. The thread first ends the step it is in, eg. a scan waiting for
    /// its clusters, then drops the guess's reads.
    fn drop(&mut self) {
        let mut slot = self.shared.lock();
        slot.prepare = None;
        slot.request = Some((u64::MAX, Request::Stop));
        drop(slot);
        self.shared.requests.notify_one();

        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// What the worker is asked to do next. The guess and the embedding travel in the slot's vector.
enum Request {
    /// Read the clusters of the guess, and prefetch its rows.
    Speculate { index: Index, params: SpecParams },
    /// Rank the guess's clusters with the embedding.
    Finish,
    /// A normal search with the embedding.
    Search { index: Index, params: SearchParams },
    /// Drop the guess.
    Cancel,
    /// End the thread.
    Stop,
}

impl Request {
    /// Whether the request brings a vector in the slot.
    fn has_vector(&self) -> bool {
        matches!(
            self,
            Request::Speculate { .. } | Request::Finish | Request::Search { .. }
        )
    }
}

/// What the caller and the worker share.
struct Shared {
    slot: Mutex<Slot>,
    /// The worker waits on it for a request or a prepare.
    requests: Condvar,
    /// The caller waits on it for the worker to take a request or to answer one.
    answers: Condvar,
}

impl Shared {
    /// The slot, also after a panic of the other side, since every update of it is whole.
    fn lock(&self) -> MutexGuard<'_, Slot> {
        self.slot.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Whether a request other than a finish, or a prepare, waits, either of which ends a guess.
    fn guess_ended(&self) -> bool {
        let slot = self.lock();
        slot.prepare.is_some() || matches!(slot.request, Some((_, ref r)) if !matches!(r, Request::Finish))
    }

    /// Whether a finish waits.
    fn finish_waits(&self) -> bool {
        matches!(self.lock().request, Some((_, Request::Finish)))
    }

    /// Whether a test asked the next finish's ranking of the guess's clusters to fail, which it then does only once.
    #[cfg(test)]
    fn fails_finish(&self) -> bool {
        std::mem::take(&mut self.lock().watch.fail_finish)
    }
}

/// The handoff between the caller and the worker.
struct Slot {
    /// A prepare that the worker hasn't run yet, with the size of the index's vectors.
    prepare: Option<(Index, usize)>,
    /// The request that the worker hasn't taken yet, with its number.
    request: Option<(u64, Request)>,
    /// The guess or the embedding of the request.
    vector: Vec<f32>,
    /// The number of the last request answered, and its hits or its error.
    answered: u64,
    hits: Vec<Hit>,
    error: Option<Error>,
    served: Served,
    #[cfg(feature = "bench")]
    trace: coldivf::Trace,
    /// Whether the worker thread has ended, eg. after a panic.
    gone: bool,
    #[cfg(test)]
    watch: Watch,
}

impl Slot {
    fn new(limits: Limits) -> Slot {
        Slot {
            prepare: None,
            request: None,
            vector: Vec::new(),
            answered: 0,
            hits: Vec::with_capacity(limits.max_k),
            error: None,
            served: Served::default(),
            #[cfg(feature = "bench")]
            trace: coldivf::Trace::default(),
            gone: false,
            #[cfg(test)]
            watch: Watch::default(),
        }
    }
}

/// Takes the slot's request, if any, and swaps its vector with `vector`, so neither side copies it twice. It wakes a
/// finish that waits for the worker to take the guess.
fn take(shared: &Shared, slot: &mut Slot, vector: &mut Vec<f32>) -> Option<(u64, Request)> {
    let (n, request) = slot.request.take()?;

    #[cfg(test)]
    if slot.watch.panic_on_next {
        panic!("the test asked the worker to panic");
    }

    if request.has_vector() {
        std::mem::swap(&mut slot.vector, vector);
    }

    shared.answers.notify_all();
    Some((n, request))
}

/// The worker thread's own state.
struct Thread {
    shared: Arc<Shared>,
    ctx: SearchCtx,
    leaves: Vec<LeafId>,
    /// The guess or the embedding of the request being served, which trades places with the slot's.
    vector: Vec<f32>,
    /// The hits of an answer, which trade places with the slot's.
    hits: Vec<Hit>,
    /// The vector of ones that a prepare routes, as long as the index's vectors.
    probe: Vec<f32>,
}

impl Thread {
    fn new(shared: Arc<Shared>, limits: Limits) -> Thread {
        Thread {
            shared,
            ctx: SearchCtx::new(limits),
            leaves: Vec::with_capacity(limits.max_nprobe),
            vector: Vec::new(),
            hits: Vec::with_capacity(limits.max_k),
            probe: Vec::new(),
        }
    }

    /// Serves requests until a stop.
    fn run(mut self) {
        set_user_interactive();

        #[cfg(all(test, target_os = "macos"))]
        {
            self.shared.lock().watch.qos = qos_class();
        }

        let mut next = None;

        loop {
            let (n, request) = match next.take() {
                Some(request) => request,
                None => self.wait(),
            };
            next = match request {
                Request::Stop => return,
                Request::Cancel => None,
                Request::Finish => {
                    let error = invalid("the guess was dropped before its finish");
                    self.answer(n, Err(error), Served::default());
                    None
                }
                Request::Search { index, params } => {
                    let taken = Some(Instant::now());
                    let result = index.search(&self.vector, &params, &mut self.ctx);
                    let result = keep(&mut self.hits, result);
                    let served = Served {
                        taken,
                        ..Served::default()
                    };
                    self.answer(n, result, served);
                    None
                }
                Request::Speculate { index, params } => match self.speculate(&index, &params) {
                    Ok(next) => next,
                    Err(()) => self.fall_back(&index, &params),
                },
            };
        }
    }

    /// Runs any prepare that waits, then takes the next request, parking while there is none.
    fn wait(&mut self) -> (u64, Request) {
        let Thread {
            shared,
            ctx,
            leaves,
            vector,
            probe,
            ..
        } = self;
        let mut slot = shared.lock();

        loop {
            #[cfg(test)]
            {
                slot = hold_at(shared, slot, Step::Take);
            }
            if let Some((index, dim)) = slot.prepare.take() {
                drop(slot);
                ready(&index, dim, ctx, probe, leaves);
                drop(index);
                slot = shared.lock();
                #[cfg(test)]
                {
                    slot.watch.prepared = Some(dim);
                }
                continue;
            }
            if let Some(request) = take(shared, &mut slot, vector) {
                return request;
            }
            slot = shared.requests.wait(slot).unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Waits for the request that ends a guess. A prepare ends it too, which gives `None`, and the main loop then runs
    /// the prepare.
    fn after_guess(shared: &Shared, vector: &mut Vec<f32>) -> Option<(u64, Request)> {
        let mut slot = shared.lock();

        loop {
            if slot.prepare.is_some() {
                return None;
            }
            if let Some(request) = take(shared, &mut slot, vector) {
                return Some(request);
            }
            slot = shared.requests.wait(slot).unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Reads the clusters of the guess in `vector` and prefetches its rows, then waits for the request that ends the
    /// guess. A finish ranks the clusters with the embedding, or runs a normal search when that ranking's reads fail.
    /// Any other request goes back to the main loop, and so does one that came between the steps. A step that fails
    /// gives `Err`, and the caller then waits for the finish.
    fn speculate(&mut self, index: &Index, params: &SpecParams) -> Result<Option<(u64, Request)>, ()> {
        let guess_taken = Some(Instant::now());
        let Thread {
            shared,
            ctx,
            leaves,
            vector,
            hits,
            ..
        } = self;
        let (clusters, reach) = (params.guess_nprobe(), params.base.reach);
        index.route(vector, clusters, reach, ctx, leaves).map_err(drop)?;
        #[cfg(test)]
        reached(shared, Step::Routed);

        if shared.guess_ended() {
            return Ok(None);
        }

        let mut fetched = index.fetch(leaves, ctx).map_err(drop)?;
        #[cfg(test)]
        reached(shared, Step::Fetched);

        if shared.guess_ended() {
            return Ok(None);
        }

        let skipped_rows = params.rerank_mult == 0 || shared.finish_waits();

        if !skipped_rows {
            fetched.prefetch_rows(vector, params.prefetch_rows()).map_err(drop)?;
            #[cfg(test)]
            reached(shared, Step::Prefetched);
        }

        let guess_ready = Some(Instant::now());
        #[cfg(test)]
        reached(shared, Step::Ready);
        let Some((n, request)) = Thread::after_guess(shared, vector) else {
            return Ok(None);
        };
        let Request::Finish = request else {
            return Ok(Some((n, request)));
        };
        let taken = Some(Instant::now());
        let base = &params.base;
        let result = fetched.search(vector, base.k, base.rerank);
        #[cfg(test)]
        let result = match shared.fails_finish() {
            true => Err(Error::other("the test failed the reads of the guess's finish")),
            false => result,
        };
        // When the reads of the guess's clusters or rows failed, a normal search reads what it needs again. An
        // embedding that the index can't take fails that search too.
        let (ranked, result) = match keep(hits, result) {
            Ok(()) => (Ranked::Guess, Ok(())),
            Err(_) => (Ranked::Fallback, keep(hits, index.search(vector, base, ctx))),
        };
        let served = Served {
            ranked,
            skipped_rows,
            guess_taken,
            guess_ready,
            taken,
            done: None,
        };
        self.answer(n, result, served);
        Ok(None)
    }

    /// Waits for the request that ends a guess whose reads failed. A finish runs a normal search with the embedding.
    fn fall_back(&mut self, index: &Index, params: &SpecParams) -> Option<(u64, Request)> {
        let (n, request) = Thread::after_guess(&self.shared, &mut self.vector)?;
        let Request::Finish = request else {
            return Some((n, request));
        };
        let taken = Some(Instant::now());
        let result = index.search(&self.vector, &params.base, &mut self.ctx);
        let result = keep(&mut self.hits, result);
        let served = Served {
            ranked: Ranked::Fallback,
            taken,
            ..Served::default()
        };
        self.answer(n, result, served);
        None
    }

    /// Hands the hits in `self.hits`, or the error, to the caller as the answer to request `n`.
    fn answer(&mut self, n: u64, result: Result<(), Error>, served: Served) {
        let mut slot = self.shared.lock();

        match result {
            Ok(()) => {
                std::mem::swap(&mut slot.hits, &mut self.hits);
                slot.error = None;
            }
            Err(e) => {
                slot.hits.clear();
                slot.error = Some(e);
            }
        }

        slot.served = Served {
            done: Some(Instant::now()),
            ..served
        };
        slot.answered = n;

        #[cfg(feature = "bench")]
        {
            slot.trace = self.ctx.trace();
        }

        drop(slot);
        self.shared.answers.notify_all();
    }
}

/// Readies `index` on `ctx` for the next query. `index.prepare` starts reading the cluster tables and makes the
/// context's buffers for the index. A route of `probe`, made `dim` ones, to one cluster then loads the tables, and a
/// fetch of no cluster makes the buffer for the clusters, which is sized from them. A step that fails is left to the
/// search, which does it again.
fn ready(index: &Index, dim: usize, ctx: &mut SearchCtx, probe: &mut Vec<f32>, leaves: &mut Vec<LeafId>) {
    let _ = index.prepare(ctx);

    if probe.len() != dim {
        probe.clear();
        probe.resize(dim, 1.0);
    }

    let _ = index.route(probe, 1, None, ctx, leaves);
    let _ = index.fetch(&[], ctx);
}

/// Copies the hits of `result` into `hits`, so the search's borrow of the context ends.
fn keep(hits: &mut Vec<Hit>, result: Result<&[Hit], Error>) -> Result<(), Error> {
    let found = result?;
    hits.clear();
    hits.extend_from_slice(found);
    Ok(())
}

/// Marks the worker as gone when its thread ends, eg. in a panic, and wakes every caller that waits for it.
struct Gone(Arc<Shared>);

impl Drop for Gone {
    fn drop(&mut self) {
        let mut slot = self.0.lock();
        slot.gone = true;
        (slot.prepare, slot.request) = (None, None);
        drop(slot);
        self.0.answers.notify_all();
    }
}

/// Runs the calling thread at the user-interactive QoS class on macOS, the class of work that the user waits on.
/// Elsewhere it does nothing.
#[cfg(target_os = "macos")]
fn set_user_interactive() {
    // SAFETY: the call changes only the calling thread's scheduling class, and it takes no pointers.
    let _ = unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0) };
}

#[cfg(not(target_os = "macos"))]
fn set_user_interactive() {}

/// The QoS class number of the calling thread.
#[cfg(all(test, target_os = "macos"))]
fn qos_class() -> Option<u32> {
    let (mut class, mut relative) = (0u32, 0);
    // SAFETY: the call writes a class number into `class` and a priority into `relative`. The class is read as the
    // integer it is, never as the enum.
    let rc = unsafe { libc::pthread_get_qos_class_np(libc::pthread_self(), (&raw mut class).cast(), &mut relative) };
    (rc == 0).then_some(class)
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message.to_string())
}

fn stopped() -> Error {
    Error::other("the worker thread stopped")
}

/// The points where the tests watch the worker, and may hold it.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// Before the idle worker takes a request or a prepare.
    Take,
    /// After a guess's route, after its fetch, and after its rows' prefetch, when it ran.
    Routed,
    Fetched,
    Prefetched,
    /// Once the guess's rows are chosen or skipped, before it waits for the request that ends it.
    Ready,
}

/// What the tests watch and steer on the worker thread.
#[cfg(test)]
#[derive(Default)]
struct Watch {
    /// Makes the thread panic as it takes its next request.
    panic_on_next: bool,
    /// Makes the next finish's ranking of the guess's clusters fail, as a failed read would.
    fail_finish: bool,
    /// The step where the thread is to wait, and the one where it waits now.
    hold: Option<Step>,
    held: Option<Step>,
    /// How many times the thread reached each step, by the step's place in `Step`.
    counts: [usize; 5],
    /// The size of the vector that the last prepare routed.
    prepared: Option<usize>,
    #[cfg(target_os = "macos")]
    qos: Option<u32>,
}

/// Notes that the worker reached `step`, and waits there while a test holds it.
#[cfg(test)]
fn reached(shared: &Shared, step: Step) {
    drop(hold_at(shared, shared.lock(), step));
}

/// Counts `step`, then waits while a test holds the worker there, unless a stop waits. It takes the slot's lock and
/// gives it back.
#[cfg(test)]
fn hold_at<'a>(shared: &'a Shared, mut slot: MutexGuard<'a, Slot>, step: Step) -> MutexGuard<'a, Slot> {
    if step != Step::Take {
        slot.watch.counts[step as usize] += 1;
    }

    while slot.watch.hold == Some(step) && !matches!(slot.request, Some((_, Request::Stop))) {
        slot.watch.held = Some(step);
        shared.answers.notify_all();
        slot = shared.requests.wait(slot).unwrap_or_else(PoisonError::into_inner);
    }

    slot.watch.held = None;
    slot
}

/// The tests' handle on a worker thread. It holds the thread at a step, waits for it to get somewhere, and counts the
/// steps it reached.
#[cfg(test)]
#[derive(Clone)]
pub(crate) struct Watcher(Arc<Shared>);

#[cfg(test)]
impl Watcher {
    /// Holds the worker the next time it reaches `step`, until `release` or a hold at another step. A stop still goes
    /// through.
    pub(crate) fn hold(&self, step: Step) {
        self.0.lock().watch.hold = Some(step);
        self.0.requests.notify_all();
    }

    /// Lets the worker go on.
    pub(crate) fn release(&self) {
        self.0.lock().watch.hold = None;
        self.0.requests.notify_all();
    }

    /// Waits until the worker is held at `step`.
    pub(crate) fn wait_held(&self, step: Step) {
        self.wait_for(
            |slot| slot.watch.held == Some(step),
            "the worker never got to the step it is held at",
        );
    }

    /// Waits until the worker has taken the last request posted.
    pub(crate) fn wait_taken(&self) {
        self.wait_for(|slot| slot.request.is_none(), "the worker never took the request");
    }

    /// Waits until a request or a prepare waits for the worker.
    pub(crate) fn wait_posted(&self) {
        self.wait_for(
            |slot| slot.request.is_some() || slot.prepare.is_some(),
            "nothing was posted",
        );
    }

    /// Waits until a normal search waits for the worker.
    pub(crate) fn wait_search(&self) {
        self.wait_for(
            |slot| matches!(slot.request, Some((_, Request::Search { .. }))),
            "no search was posted",
        );
    }

    /// How many times the worker reached `step` so far.
    pub(crate) fn count(&self, step: Step) -> usize {
        self.0.lock().watch.counts[step as usize]
    }

    /// The size of the vector that the last prepare routed, if one ran.
    pub(crate) fn prepared(&self) -> Option<usize> {
        self.0.lock().watch.prepared
    }

    /// Makes the next finish's ranking of the guess's clusters fail, as a failed read would.
    pub(crate) fn fail_next_finish(&self) {
        self.0.lock().watch.fail_finish = true;
    }

    /// Waits until `done` holds for the slot, and fails the test after 30 s with `what`. It looks again at least every
    /// millisecond, since a post wakes only the worker.
    fn wait_for(&self, done: impl Fn(&Slot) -> bool, what: &str) {
        let (start, tick) = (Instant::now(), std::time::Duration::from_millis(1));
        let mut slot = self.0.lock();

        while !done(&slot) {
            assert!(start.elapsed().as_secs() < 30, "{what}");
            slot = self
                .0
                .answers
                .wait_timeout(slot, tick)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/speculate.rs"]
mod tests;

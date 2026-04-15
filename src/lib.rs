//! Generic request batcher.
//!
//! This crate provides two batcher implementations:
//! - [`Batcher`] for async workloads
//! - [`SyncBatcher`] for synchronous workloads with thread-local resources
//!
//! Both transparently merge concurrent `run` calls into batches and forward each
//! batch to a single processing invocation, then fan results back to their
//! respective callers.
//!
//! # Batching strategy
//!
//! After the first request arrives the worker waits up to [`BatcherConfig::max_wait_ms`]
//! for more requests to accumulate, then issues a single process call for all
//! of them. Under high load the batch fills to [`BatcherConfig::max_batch`] before
//! the timer expires, so throughput scales with concurrency. Under low load each
//! request waits at most `max_wait_ms` before being dispatched solo.
//!
//! # Error model
//!
//! Process functions return `Result<Vec<Res>, E>`. A batch-level error (`Err(E)`) is
//! cloned and delivered to every caller in that batch — hence `E: Clone`.
//! Infrastructure failures (worker died, channel closed) also surface as `Err(E)`.
//!
//! # Which batcher to choose?
//!
//! - Use [`Batcher`] when your processing logic is async and doesn't need
//!   thread-local state.
//! - Use [`SyncBatcher`] when your processing logic is synchronous and needs
//!   thread-local resources (like `fastembed`'s `TextEmbedding`).

pub mod batcher;
pub mod common;
pub mod sync_batcher;

pub use batcher::Batcher;
pub use common::BatcherConfig;
pub use sync_batcher::SyncBatcher;

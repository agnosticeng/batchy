//! Common types used by both `Batcher` and `SyncBatcher`.

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Builder, Serialize, Deserialize)]
#[serde(default)]
#[builder(default)]
pub struct BatcherConfig {
    /// Maximum number of `run` calls merged into one `process` invocation.
    /// Default: 32.
    pub max_batch: usize,
    /// Bound of the internal request queue. Callers block when full (backpressure).
    /// Default: 128.
    pub queue_size: usize,
    /// Maximum time in milliseconds to wait for additional requests after the first
    /// one arrives. Higher values increase batching under low load at the cost of
    /// added latency. Default: 50.
    pub max_wait_ms: u64,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        BatcherConfig {
            max_batch: 32,
            queue_size: 128,
            max_wait_ms: 50,
        }
    }
}

// ── Pending request ───────────────────────────────────────────────────────────

pub(crate) struct PendingRequest<Req, Res, E> {
    pub(crate) item: Req,
    pub(crate) reply: oneshot::Sender<Result<Res, E>>,
}

// ── Worker functions ──────────────────────────────────────────────────────────

pub(crate) async fn async_worker<Req, Res, E, F, Fut>(
    mut rx: mpsc::Receiver<PendingRequest<Req, Res, E>>,
    config: BatcherConfig,
    process: F,
) where
    Req: Send + 'static,
    Res: Send + 'static,
    E: Clone + Send + 'static,
    F: Fn(Vec<Req>) -> Fut,
    Fut: Future<Output = Result<Vec<Res>, E>>,
{
    use tokio::time::Instant;

    let max_wait = Duration::from_millis(config.max_wait_ms);

    loop {
        let Some(first) = rx.recv().await else { break };
        let mut batch = vec![first];
        let deadline = Instant::now() + max_wait;

        while batch.len() < config.max_batch {
            if let Ok(req) = rx.try_recv() {
                batch.push(req);
                continue;
            }

            let remaining = deadline.saturating_duration_since(Instant::now());

            if remaining.is_zero() {
                break;
            }

            match tokio::time::timeout(remaining, rx.recv()).await {
                Ok(Some(req)) => batch.push(req),
                _ => break,
            }
        }

        let (items, replies): (Vec<Req>, Vec<_>) =
            batch.into_iter().map(|r| (r.item, r.reply)).unzip();

        match process(items).await {
            Err(e) => {
                for reply in replies {
                    let _ = reply.send(Err(e.clone()));
                }
            }
            Ok(results) => {
                for (reply, res) in replies.into_iter().zip(results) {
                    let _ = reply.send(Ok(res));
                }
            }
        }
    }
}

pub(crate) fn sync_worker<Req, Res, E, G>(
    mut rx: mpsc::Receiver<PendingRequest<Req, Res, E>>,
    config: BatcherConfig,
    process: G,
) where
    Req: Send + 'static,
    Res: Send + 'static,
    E: Clone + Send + 'static,
    G: Fn(Vec<Req>) -> Result<Vec<Res>, E> + Send + 'static,
{
    use std::time::Instant;

    let max_wait = Duration::from_millis(config.max_wait_ms);

    loop {
        let first = match rx.blocking_recv() {
            Some(req) => req,
            None => break,
        };
        let mut batch = vec![first];
        let deadline = Instant::now() + max_wait;

        while batch.len() < config.max_batch {
            if let Ok(req) = rx.try_recv() {
                batch.push(req);
                continue;
            }

            let remaining = deadline.saturating_duration_since(Instant::now());

            if remaining.is_zero() {
                break;
            }

            let poll_interval = Duration::from_millis(5);
            let mut received = None;
            
            while Instant::now() < deadline {
                match rx.try_recv() {
                    Ok(req) => {
                        received = Some(req);
                        break;
                    }
                    Err(_) => {
                        std::thread::sleep(poll_interval);
                    }
                }
            }
            
            if let Some(req) = received {
                batch.push(req);
            } else {
                break;
            }
        }

        let (items, replies): (Vec<Req>, Vec<_>) =
            batch.into_iter().map(|r| (r.item, r.reply)).unzip();

        match process(items) {
            Err(e) => {
                for reply in replies {
                    let _ = reply.send(Err(e.clone()));
                }
            }
            Ok(results) => {
                for (reply, res) in replies.into_iter().zip(results) {
                    let _ = reply.send(Ok(res));
                }
            }
        }
    }
}

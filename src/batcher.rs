//! Async batcher implementation.
//!
//! See [`Batcher`] for details.

use crate::common::{async_worker, BatcherConfig, PendingRequest};
use std::future::Future;
use tokio::sync::mpsc;

/// A cloneable handle to the async batch worker.
///
/// Transparently merges concurrent [`Batcher::run`] calls into batches and
/// forwards each batch to a single `process` invocation, then fans results back
/// to their respective callers.
///
/// Each caller submits one `Req` and receives one `Res` — the batching is
/// invisible at the call site.
///
/// # Batching strategy
///
/// After the first request arrives the worker waits up to [`BatcherConfig::max_wait_ms`]
/// for more requests to accumulate, then issues a single `process` call for all
/// of them. Under high load the batch fills to [`BatcherConfig::max_batch`] before
/// the timer expires, so throughput scales with concurrency. Under low load each
/// request waits at most `max_wait_ms` before being dispatched solo.
///
/// # Error model
///
/// `process` returns `Result<Vec<Res>, E>`. A batch-level error (`Err(E)`) is
/// cloned and delivered to every caller in that batch — hence `E: Clone`.
/// Infrastructure failures (worker died, channel closed) also surface as `Err(E)`
/// via [`Batcher::run`].
///
/// # Blocking work
///
/// `process` is an ordinary async function. If the underlying computation is
/// CPU-heavy or blocking, the caller is responsible for wrapping it in
/// [`tokio::task::spawn_blocking`] inside the closure.
///
/// # Synchronous workloads
///
/// For CPU-heavy synchronous workloads that need thread-local resources,
/// use [`crate::SyncBatcher`] instead.
///
/// # Example
///
/// ```no_run
/// # use batchy::{Batcher, BatcherConfig};
/// # async fn run_example() -> Result<(), String> {
/// let config = BatcherConfig::default();
///
/// let batcher: Batcher<i32, i32, String> = Batcher::new(config, |items: Vec<i32>| async move {
///     Ok(items.into_iter().map(|x| x * 2).collect())
/// });
///
/// let result = batcher.run(5).await?;
/// assert_eq!(result, 10);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Batcher<Req, Res, E> {
    tx: mpsc::Sender<PendingRequest<Req, Res, E>>,
}

impl<Req, Res, E> Batcher<Req, Res, E>
where
    Req: Send + 'static,
    Res: Send + 'static,
    E: Clone + Send + 'static,
{
    /// Spawn the async batch worker.
    ///
    /// `process` receives a `Vec<Req>` (one entry per concurrent [`Batcher::run`] call
    /// merged into this batch) and must return either:
    /// - `Err(E)` — the whole batch failed; every caller receives a clone of `E`
    /// - `Ok(Vec<Res>)` — one result per input item, in the same order
    ///
    /// If the underlying work is CPU-heavy or blocking, `process` should call
    /// [`tokio::task::spawn_blocking`] itself.
    pub fn new<F, Fut>(config: BatcherConfig, process: F) -> Self
    where
        F: Fn(Vec<Req>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Vec<Res>, E>> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel::<PendingRequest<Req, Res, E>>(config.queue_size);
        tokio::spawn(async_worker(rx, config, process));
        Self { tx }
    }

    /// Submit one item for processing.
    ///
    /// Waits until the worker has processed the batch containing this item,
    /// then returns its result.
    ///
    /// Returns `Err(E)` if the worker died before processing the request, or if
    /// `process` returned a batch-level error.
    pub async fn run(&self, item: Req) -> Result<Res, E> {
        use tokio::sync::oneshot;

        let (reply_tx, reply_rx) = oneshot::channel();

        self.tx
            .send(PendingRequest {
                item,
                reply: reply_tx,
            })
            .await
            .map_err(|_| unreachable!("worker exited before all Batcher handles were dropped"))?;

        reply_rx
            .await
            .map_err(|_| unreachable!("worker dropped the reply sender without responding"))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{BatcherConfig, BatcherConfigBuilder};
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_single_request() {
        let config = BatcherConfig {
            max_batch: 32,
            queue_size: 128,
            max_wait_ms: 100,
        };

        let batcher: Batcher<i32, i32, String> = Batcher::new(config, |items: Vec<i32>| async move {
            Ok(items.into_iter().map(|x| x * 2).collect())
        });

        let result = batcher.run(5).await.unwrap();
        assert_eq!(result, 10);
    }

    #[tokio::test]
    async fn test_batching_under_load() {
        let config = BatcherConfig {
            max_batch: 10,
            queue_size: 128,
            max_wait_ms: 500,
        };

        let batch_sizes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let batch_sizes_clone = batch_sizes.clone();
        
        let batcher: Batcher<i32, i32, String> = Batcher::new(config, move |items: Vec<i32>| {
            batch_sizes_clone.lock().unwrap().push(items.len());
            async move {
                sleep(Duration::from_millis(10)).await;
                Ok(items.into_iter().map(|x| x * 2).collect())
            }
        });

        let handles: Vec<_> = (0..25).map(|i| batcher.run(i)).collect();
        let results = futures::future::join_all(handles).await;

        for (i, result) in results.into_iter().enumerate() {
            assert_eq!(result.unwrap(), i as i32 * 2);
        }

        let sizes = batch_sizes.lock().unwrap();
        assert_eq!(sizes.len(), 3);
        assert_eq!(sizes[0], 10);
        assert_eq!(sizes[1], 10);
        assert_eq!(sizes[2], 5);
    }

    #[tokio::test]
    async fn test_timeout_under_low_load() {
        let config = BatcherConfig {
            max_batch: 32,
            queue_size: 128,
            max_wait_ms: 50,
        };

        let call_count = std::sync::Arc::new(std::sync::Mutex::new(0));
        let call_count_clone = call_count.clone();
        
        let batcher: Batcher<i32, i32, String> = Batcher::new(config, move |items: Vec<i32>| {
            *call_count_clone.lock().unwrap() += 1;
            async move {
                Ok(items.into_iter().map(|x| x * 2).collect())
            }
        });

        let start = std::time::Instant::now();
        let result = batcher.run(5).await.unwrap();
        let elapsed = start.elapsed();

        assert_eq!(result, 10);
        assert!(elapsed >= Duration::from_millis(50));
        assert!(elapsed < Duration::from_millis(150));
        assert_eq!(*call_count.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn test_max_batch_limit() {
        let config = BatcherConfig {
            max_batch: 5,
            queue_size: 128,
            max_wait_ms: 500,
        };

        let batch_sizes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let batch_sizes_clone = batch_sizes.clone();
        
        let batcher: Batcher<i32, i32, String> = Batcher::new(config, move |items: Vec<i32>| {
            batch_sizes_clone.lock().unwrap().push(items.len());
            async move {
                sleep(Duration::from_millis(10)).await;
                Ok(items.into_iter().map(|x| x * 2).collect())
            }
        });

        let handles: Vec<_> = (0..12).map(|i| batcher.run(i)).collect();
        let results = futures::future::join_all(handles).await;

        for (i, result) in results.into_iter().enumerate() {
            assert_eq!(result.unwrap(), i as i32 * 2);
        }

        let sizes = batch_sizes.lock().unwrap();
        assert_eq!(sizes.len(), 3);
        assert_eq!(sizes[0], 5);
        assert_eq!(sizes[1], 5);
        assert_eq!(sizes[2], 2);
    }

    #[tokio::test]
    async fn test_error_propagation() {
        let config = BatcherConfig {
            max_batch: 10,
            queue_size: 128,
            max_wait_ms: 100,
        };

        let batcher: Batcher<i32, i32, String> = Batcher::new(config, |items: Vec<i32>| async move {
            if items.iter().any(|&x| x < 0) {
                Err("Negative value".to_string())
            } else {
                Ok(items.into_iter().map(|x| x * 2).collect())
            }
        });

        let result1 = batcher.run(5).await.unwrap();
        assert_eq!(result1, 10);

        let result2 = batcher.run(-1).await;
        assert!(result2.is_err());
        assert_eq!(result2.unwrap_err(), "Negative value");

        let result3 = batcher.run(10).await.unwrap();
        assert_eq!(result3, 20);
    }

    #[tokio::test]
    async fn test_concurrent_submissions() {
        let config = BatcherConfig {
            max_batch: 100,
            queue_size: 256,
            max_wait_ms: 200,
        };

        let batcher: Batcher<String, String, String> = Batcher::new(config, |items: Vec<String>| async move {
            sleep(Duration::from_millis(50)).await;
            Ok(items.into_iter().map(|s| format!("processed: {}", s)).collect())
        });

        let handles: Vec<_> = (0..50)
            .map(|i| {
                let batcher = batcher.clone();
                tokio::spawn(async move {
                    batcher.run(format!("req-{}", i)).await.unwrap()
                })
            })
            .collect();

        let results = futures::future::join_all(handles).await;
        
        for (i, result) in results.into_iter().enumerate() {
            assert_eq!(result.unwrap(), format!("processed: req-{}", i));
        }
    }

    #[tokio::test]
    async fn test_backpressure() {
        let config = BatcherConfig {
            max_batch: 2,
            queue_size: 2,
            max_wait_ms: 100,
        };

        let batcher: Batcher<i32, i32, String> = Batcher::new(config, |items: Vec<i32>| async move {
            sleep(Duration::from_millis(50)).await;
            Ok(items.into_iter().map(|x| x * 2).collect())
        });

        let handle1 = tokio::spawn({
            let batcher = batcher.clone();
            async move { batcher.run(1).await }
        });

        let handle2 = tokio::spawn({
            let batcher = batcher.clone();
            async move { batcher.run(2).await }
        });

        let handle3 = tokio::spawn({
            let batcher = batcher.clone();
            async move { batcher.run(3).await }
        });

        let results = futures::future::join_all(vec![handle1, handle2, handle3])
            .await
            .into_iter()
            .map(|h| h.unwrap().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(results.len(), 3);
        assert!(results.contains(&2));
        assert!(results.contains(&4));
        assert!(results.contains(&6));
    }

    #[tokio::test]
    async fn test_builder_pattern() {
        let config = BatcherConfigBuilder::default()
            .max_batch(64)
            .queue_size(256)
            .max_wait_ms(100)
            .build()
            .unwrap();

        let batcher: Batcher<String, String, String> = Batcher::new(config, |items: Vec<String>| async move {
            Ok(items)
        });

        let result = batcher.run("test".to_string()).await.unwrap();
        assert_eq!(result, "test");
    }

    #[tokio::test]
    async fn test_order_preservation() {
        let config = BatcherConfig {
            max_batch: 32,
            queue_size: 128,
            max_wait_ms: 100,
        };

        let batcher: Batcher<i32, i32, String> = Batcher::new(config, |items: Vec<i32>| async move {
            sleep(Duration::from_millis(10)).await;
            Ok(items.into_iter().map(|x| x * x).collect())
        });

        let inputs: Vec<i32> = (0..20).collect();
        let handles: Vec<_> = inputs
            .iter()
            .copied()
            .map(|i| batcher.run(i))
            .collect();

        let results: Vec<i32> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r: Result<i32, String>| r.unwrap())
            .collect();

        for (input, output) in inputs.iter().zip(results.iter()) {
            assert_eq!(*output, input * input);
        }
    }
}

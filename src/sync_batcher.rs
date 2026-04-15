//! Synchronous batcher implementation.
//!
//! See [`SyncBatcher`] for details.

use crate::common::{sync_worker, BatcherConfig, PendingRequest};
use tokio::sync::mpsc;

/// A batcher for synchronous workloads with thread-local resources.
///
/// Unlike [`crate::Batcher`], which runs an async `process` closure on the tokio runtime,
/// `SyncBatcher` spawns a dedicated OS thread that:
/// 1. Initializes resources once via the `init` closure (e.g., `TextEmbedding`)
/// 2. Processes batches synchronously using those resources
///
/// This avoids `spawn_blocking` overhead for each request and keeps thread-local
/// state alive on the worker thread.
///
/// # Example
///
/// ```no_run
/// # use batchy::{BatcherConfig, SyncBatcher};
/// # struct Embedding(Vec<f32>);
/// # struct TextEmbedding;
/// # impl TextEmbedding {
/// #     fn try_new(_: ()) -> Result<Self, ()> { Ok(TextEmbedding) }
/// #     fn embed(&self, _: Vec<String>) -> Result<Vec<Embedding>, String> { Ok(vec![]) }
/// # }
/// # async fn run_example() -> Result<(), String> {
/// let config = BatcherConfig::default();
///
/// let batcher: SyncBatcher<String, Embedding, String> = SyncBatcher::new(config, || {
///     let embedding = TextEmbedding::try_new(()).unwrap();
///     
///     move |items: Vec<String>| {
///         embedding.embed(items)
///     }
/// });
///
/// let result = batcher.run("hello".to_string()).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct SyncBatcher<Req, Res, E> {
    tx: mpsc::Sender<PendingRequest<Req, Res, E>>,
}

impl<Req, Res, E> SyncBatcher<Req, Res, E>
where
    Req: Send + 'static,
    Res: Send + 'static,
    E: Clone + Send + 'static,
{
    /// Spawn the sync batch worker on a dedicated thread.
    ///
    /// `init` runs once on the worker thread and returns the processing closure.
    /// This is where you initialize thread-local resources (e.g., `TextEmbedding`).
    ///
    /// The processing closure receives a `Vec<Req>` and must return
    /// `Result<Vec<Res>, E>` synchronously.
    pub fn new<F, G>(config: BatcherConfig, init: F) -> Self
    where
        F: FnOnce() -> G + Send + 'static,
        G: Fn(Vec<Req>) -> Result<Vec<Res>, E> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel::<PendingRequest<Req, Res, E>>(config.queue_size);

        std::thread::spawn(move || {
            let process = init();
            sync_worker(rx, config, process);
        });

        Self { tx }
    }

    /// Submit one item for processing.
    ///
    /// Waits until the worker has processed the batch containing this item,
    /// then returns its result.
    ///
    /// Returns `Err(E)` if the worker died before processing the request, or if
    /// the processing closure returned an error.
    pub async fn run(&self, item: Req) -> Result<Res, E> {
        use tokio::sync::oneshot;

        let (reply_tx, reply_rx) = oneshot::channel();

        self.tx
            .send(PendingRequest {
                item,
                reply: reply_tx,
            })
            .await
            .map_err(|_| unreachable!("worker exited before all SyncBatcher handles were dropped"))?;

        reply_rx
            .await
            .map_err(|_| unreachable!("worker dropped the reply sender without responding"))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::BatcherConfig;
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_sync_single_request() {
        let config = BatcherConfig {
            max_batch: 32,
            queue_size: 128,
            max_wait_ms: 100,
        };

        let batcher: SyncBatcher<i32, i32, String> = SyncBatcher::new(config, || {
            move |items: Vec<i32>| Ok(items.into_iter().map(|x| x * 2).collect())
        });

        let result = batcher.run(5).await.unwrap();
        assert_eq!(result, 10);
    }

    #[tokio::test]
    async fn test_sync_batching_under_load() {
        let config = BatcherConfig {
            max_batch: 10,
            queue_size: 128,
            max_wait_ms: 500,
        };

        let batch_sizes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let batch_sizes_clone = batch_sizes.clone();
        
        let batcher: SyncBatcher<i32, i32, String> = SyncBatcher::new(config, move || {
            let batch_sizes = batch_sizes_clone.clone();
            move |items: Vec<i32>| {
                batch_sizes.lock().unwrap().push(items.len());
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
    async fn test_sync_timeout_under_low_load() {
        let config = BatcherConfig {
            max_batch: 32,
            queue_size: 128,
            max_wait_ms: 50,
        };

        let call_count = std::sync::Arc::new(std::sync::Mutex::new(0));
        let call_count_clone = call_count.clone();
        
        let batcher: SyncBatcher<i32, i32, String> = SyncBatcher::new(config, move || {
            let call_count = call_count_clone.clone();
            move |items: Vec<i32>| {
                *call_count.lock().unwrap() += 1;
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
    async fn test_sync_max_batch_limit() {
        let config = BatcherConfig {
            max_batch: 5,
            queue_size: 128,
            max_wait_ms: 500,
        };

        let batch_sizes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let batch_sizes_clone = batch_sizes.clone();
        
        let batcher: SyncBatcher<i32, i32, String> = SyncBatcher::new(config, move || {
            let batch_sizes = batch_sizes_clone.clone();
            move |items: Vec<i32>| {
                batch_sizes.lock().unwrap().push(items.len());
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
    async fn test_sync_error_propagation() {
        let config = BatcherConfig {
            max_batch: 10,
            queue_size: 128,
            max_wait_ms: 100,
        };

        let batcher: SyncBatcher<i32, i32, String> = SyncBatcher::new(config, || {
            move |items: Vec<i32>| {
                if items.iter().any(|&x| x < 0) {
                    Err("Negative value".to_string())
                } else {
                    Ok(items.into_iter().map(|x| x * 2).collect())
                }
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
    async fn test_sync_concurrent_submissions() {
        let config = BatcherConfig {
            max_batch: 100,
            queue_size: 256,
            max_wait_ms: 200,
        };

        let batcher: SyncBatcher<String, String, String> = SyncBatcher::new(config, || {
            move |items: Vec<String>| {
                Ok(items.into_iter().map(|s| format!("processed: {}", s)).collect())
            }
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
    async fn test_sync_order_preservation() {
        let config = BatcherConfig {
            max_batch: 32,
            queue_size: 128,
            max_wait_ms: 100,
        };

        let batcher: SyncBatcher<i32, i32, String> = SyncBatcher::new(config, || {
            move |items: Vec<i32>| {
                Ok(items.into_iter().map(|x| x * x).collect())
            }
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

    #[tokio::test]
    async fn test_sync_thread_local_init() {
        let config = BatcherConfig {
            max_batch: 32,
            queue_size: 128,
            max_wait_ms: 100,
        };

        let batcher: SyncBatcher<i32, i32, String> = SyncBatcher::new(config, || {
            let multiplier = 42;
            
            move |items: Vec<i32>| {
                Ok(items.into_iter().map(|x| x * multiplier).collect())
            }
        });

        let result = batcher.run(5).await.unwrap();
        assert_eq!(result, 210);
    }
}

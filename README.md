# Batchy

Transparently batch concurrent requests into efficient bulk operations.

Batchy merges multiple concurrent requests into larger batches, forwarding them to a single processing call. It's perfect for ML inference, database queries, API calls, or any scenario where batching improves throughput.

## Features

- **Transparent batching**: Callers submit single requests and receive single responses - batching is invisible at the call site
- **Configurable strategy**: Control max batch size, queue size, and wait time
- **Backpressure**: Built-in queue limits prevent memory exhaustion under high load
- **Error handling**: Batch-level errors are delivered to all affected callers
- **Two modes**: Async `Batcher` for async workloads, `SyncBatcher` for sync workloads with thread-local resources

## Quick Start

```toml
[dependencies]
batchy = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## Async Batcher

For async workloads:

```rust
use batchy::{Batcher, BatcherConfig};

#[tokio::main]
async fn main() {
    let config = BatcherConfig {
        max_batch: 16,
        queue_size: 64,
        max_wait_ms: 50,
    };

    let batcher = Batcher::new(config, |prompts: Vec<String>| async move {
        let results: Vec<String> = prompts
            .iter()
            .map(|p| format!("Processed: {}", p))
            .collect();
        Ok(results)
    });

    let handles: Vec<_> = (0..10)
        .map(|i| batcher.run(format!("request-{}", i)))
        .collect();

    let results = futures::future::join_all(handles).await;
    
    for result in results {
        println!("{:?}", result);
    }
}
```

### CPU-Heavy Async Work

For CPU-intensive async operations, wrap in `spawn_blocking`:

```rust
let batcher = Batcher::new(config, move |items: Vec<YourInput>| async move {
    tokio::task::spawn_blocking(move || {
        your_cpu_heavy_function(items)
    })
    .await
    .map_err(|e| format!("Task panicked: {}", e))?
});
```

## Synchronous Batcher

For sync workloads that need thread-local resources (like `fastembed`'s `TextEmbedding`):

```rust
use batchy::{SyncBatcher, BatcherConfig};

#[tokio::main]
async fn main() {
    let config = BatcherConfig {
        max_batch: 16,
        queue_size: 64,
        max_wait_ms: 50,
    };

    // Init runs ONCE on the worker thread
    let batcher: SyncBatcher<String, Vec<f32>, String> = SyncBatcher::new(config, || {
        let model = load_embedding_model(); // Expensive init
        
        move |texts: Vec<String>| {
            // Uses the initialized model
            Ok(texts.into_iter().map(|t| embed_text(&model, &t)).collect())
        }
    });

    let result = batcher.run("hello".to_string()).await?;
    println!("{:?}", result);
}
```

This avoids `spawn_blocking` overhead and keeps thread-local state alive on the worker thread.

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `max_batch` | 32 | Maximum requests merged into one processing call |
| `queue_size` | 128 | Size of the internal request queue (backpressure when full) |
| `max_wait_ms` | 50 | Maximum wait time for batch to fill under low load |

```rust
use batchy::BatcherConfig;
use batchy::BatcherConfigBuilder;

// Using builder pattern
let config = BatcherConfigBuilder::default()
    .max_batch(64)
    .max_wait_ms(100)
    .build()
    .unwrap();
```

## Batching Strategy

1. First request arrives, worker starts timer
2. Worker accumulates requests until:
   - `max_batch` is reached (immediate processing), OR
   - `max_wait_ms` expires (process what we have)
3. Processing call executes with accumulated batch
4. Results are fanned back to respective callers

Under high load: batches fill to `max_batch` for maximum throughput.  
Under low load: each request waits at most `max_wait_ms` for bounded latency.

## Error Handling

The processor returns `Result<Vec<Res>, E>`:
- `Err(e)`: The entire batch failed - every caller in that batch receives a clone of `e`
- `Ok(results)`: One result per input, in the same order

```rust
let batcher = Batcher::new(config, |items: Vec<i32>| async move {
    if items.iter().any(|&x| x < 0) {
        return Err("Negative values not allowed".to_string());
    }
    
    Ok(items.into_iter().map(|x| x * 2).collect())
});
```

## Use Cases

- **ML Inference**: Batch multiple inputs for GPU efficiency (`Batcher` or `SyncBatcher`)
- **Database Queries**: Combine individual queries into bulk operations (`Batcher`)
- **API Calls**: Aggregate requests to respect rate limits (`Batcher`)
- **File I/O**: Batch disk writes for better throughput (`SyncBatcher`)
- **Embedding Models**: Thread-local model instances with sync processing (`SyncBatcher`)

## License

MIT

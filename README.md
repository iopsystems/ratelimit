# ratelimit

A lock-free token bucket ratelimiter for rate limiting and admission control.

## Getting Started

```bash
cargo add ratelimit
```

`std` is enabled by default. For `no_std` targets, disable default features:

```toml
ratelimit = { version = "1", default-features = false }
```

## Usage

`Ratelimiter::new`, `Ratelimiter::builder`, and `StdClock` are available with
the default `std` feature.

```rust
use ratelimit::Ratelimiter;

// 10,000 requests/s
let ratelimiter = Ratelimiter::new(10_000);

loop {
    match ratelimiter.try_wait() {
        Ok(()) => {
            // token acquired -- perform rate-limited action
            break;
        }
        Err(wait) => {
            // rate limited -- sleep and retry
            std::thread::sleep(wait);
        }
    }
}
```

For more control over burst capacity and initial tokens:

```rust
use ratelimit::Ratelimiter;

let ratelimiter = Ratelimiter::builder(10_000)
    .max_tokens(50_000)       // allow up to 5 seconds of burst
    .initial_available(1_000) // start with some tokens available
    .build()
    .unwrap();
```

The rate can be changed dynamically at runtime:

```rust
use ratelimit::Ratelimiter;

let ratelimiter = Ratelimiter::new(1_000);
// later...
ratelimiter.set_rate(5_000);
```

A rate of 0 means unlimited -- `try_wait()` will always succeed.

## no_std

Disable default features and provide your own monotonic clock:

```rust
use core::time::Duration;
use ratelimit::{Clock, Ratelimiter};

struct FixedClock;

impl Clock for FixedClock {
    fn elapsed(&self) -> Duration {
        Duration::from_millis(10)
    }
}

let ratelimiter = Ratelimiter::with_clock(1_000, FixedClock);
assert!(ratelimiter.try_wait().is_ok());
```

## Design

This crate implements a lock-free token bucket algorithm. Tokens accumulate
continuously based on elapsed time using scaled integer arithmetic for
sub-token precision. All synchronization is done with atomic CAS operations,
making it safe to share across threads with no mutex contention.

Key parameters:

- **rate** -- tokens per second. The single knob for controlling throughput.
- **max tokens** -- bucket capacity, bounding burst size. Defaults to `rate`
  (1 second of burst).
- **initial available** -- tokens available at construction. Defaults to 0.

## Migration from `0.10.x`

Before :

```rust
use ratelimit::Ratelimiter;
use std::time::Duration;

// 8 requests/s
let ratelimiter = Ratelimiter::builder(8, Duration::from_secs(1))
    .build()
    .unwrap();
```

After:

```rust
use ratelimit::Ratelimiter;

// 8 requests/s
let ratelimiter = Ratelimiter::builder(8)
    .build()
    .unwrap();
```

## Links

- [API Documentation (docs.rs)](https://docs.rs/ratelimit/)
- [Crate (crates.io)](https://crates.io/crates/ratelimit)
- [Repository (GitHub)](https://github.com/iopsystems/ratelimit)

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.

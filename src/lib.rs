//! A lock-free token bucket ratelimiter that can be shared between threads.
//!
//! The ratelimiter uses scaled tokens internally for sub-token precision,
//! allowing accurate rate limiting at any rate without requiring callers to
//! tune refill intervals.
//!
//! The `std` feature is enabled by default and provides [`StdClock`],
//! [`Ratelimiter::new`], and [`Ratelimiter::builder`]. Disable default
//! features to use the crate in `no_std` environments and supply your own
//! [`Clock`].
//!
//! ```no_run
//! use ratelimit::Ratelimiter;
//!
//! // 1000 requests/s, no initial tokens, burst limited to 1 second
//! let ratelimiter = Ratelimiter::new(1000);
//!
//! // Custom burst capacity and initial tokens
//! let ratelimiter = Ratelimiter::builder(1000)
//!     .max_tokens(5000)
//!     .initial_available(100)
//!     .build()
//!     .unwrap();
//!
//! // Rate of 0 means unlimited — try_wait() always succeeds
//! let ratelimiter = Ratelimiter::new(0);
//! assert!(ratelimiter.try_wait().is_ok());
//!
//! // Sleep-wait loop
//! let ratelimiter = Ratelimiter::new(100);
//! for _ in 0..10 {
//!     while let Err(wait) = ratelimiter.try_wait() {
//!         std::thread::sleep(wait);
//!     }
//!     // do some ratelimited action here
//! }
//! ```
//!
//! ```
//! use core::time::Duration;
//! use ratelimit::{Clock, Ratelimiter};
//!
//! struct FixedClock;
//!
//! impl Clock for FixedClock {
//!     fn elapsed(&self) -> Duration {
//!         Duration::from_millis(10)
//!     }
//! }
//!
//! let ratelimiter = Ratelimiter::with_clock(1000, FixedClock);
//! assert!(ratelimiter.try_wait().is_ok());
//! ```
#![no_std]

#[cfg(any(feature = "std", test))]
extern crate std;

use core::fmt::{self, Debug, Formatter};
use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;
use thiserror::Error;

/// Abstraction over a monotonic clock.
pub trait Clock {
    /// Returns the elapsed time since this clock was created.
    fn elapsed(&self) -> Duration;
}

/// Standard library clock implementation.
///
/// This clock uses [`std::time::Instant`] for high-precision timing.
/// Available only when the `std` feature is enabled, which it is by default.
#[cfg(feature = "std")]
pub struct StdClock(std::time::Instant);

#[cfg(feature = "std")]
impl StdClock {
    /// Create a new clock starting from the current time.
    pub fn new() -> Self {
        Self(std::time::Instant::now())
    }
}

#[cfg(feature = "std")]
impl Clock for StdClock {
    fn elapsed(&self) -> Duration {
        self.0.elapsed()
    }
}

/// Internal scale factor for sub-token precision. Allows smooth token
/// accumulation at any rate without discrete refill intervals.
const TOKEN_SCALE: u64 = 1_000_000;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    #[error("initial available tokens cannot exceed max tokens")]
    AvailableTokensTooHigh,
    #[error("max tokens must be at least 1")]
    MaxTokensTooLow,
}

/// A lock-free token bucket ratelimiter.
///
/// Tokens accumulate continuously based on elapsed time. Each `try_wait()`
/// call consumes one token. A rate of 0 means unlimited (no rate limiting).
#[must_use]
pub struct Ratelimiter<C: Clock> {
    /// Target rate in tokens per second. 0 = unlimited.
    rate: AtomicU64,
    /// Maximum tokens (burst capacity) in real tokens.
    max_tokens: AtomicU64,
    /// Available tokens, scaled by TOKEN_SCALE for sub-token precision.
    tokens: AtomicU64,
    /// Tokens dropped due to bucket overflow, scaled by TOKEN_SCALE.
    dropped: AtomicU64,
    /// Last refill timestamp in nanoseconds since clock creation.
    last_refill_ns: AtomicU64,
    /// Clock for measuring elapsed time.
    clock: C,
}

#[cfg(feature = "std")]
impl Ratelimiter<StdClock> {
    /// Create a new ratelimiter with the given rate in tokens per second.
    ///
    /// A rate of 0 means unlimited — `try_wait()` will always succeed.
    ///
    /// The ratelimiter starts with no tokens available. Burst capacity
    /// defaults to `rate` tokens (1 second worth). Use `builder()` for
    /// more control.
    ///
    /// Available only when the `std` feature is enabled, which it is by
    /// default.
    ///
    /// # Example
    ///
    /// ```
    /// use ratelimit::Ratelimiter;
    ///
    /// // 1000 requests/s, no initial tokens, burst limited to 1 second
    /// let ratelimiter = Ratelimiter::new(1000);
    /// ```
    pub fn new(rate: u64) -> Self {
        Self::with_clock(rate, StdClock::new())
    }

    /// Create a builder for configuring the ratelimiter with StdClock.
    ///
    /// Available only when the `std` feature is enabled, which it is by
    /// default.
    pub fn builder(rate: u64) -> Builder<StdClock> {
        Builder::with_clock(rate, StdClock::new())
    }
}

impl<C> Ratelimiter<C>
where
    C: Clock,
{
    /// Create a new ratelimiter with the given rate and clock.
    ///
    /// This constructor is available for any clock type implementing the
    /// [`Clock`] trait. This is the constructor to use in `no_std`
    /// environments. For the standard library clock, use [`Ratelimiter::new`].
    ///
    /// # Example
    ///
    /// ```
    /// use ratelimit::{Clock, Ratelimiter};
    /// use core::time::Duration;
    ///
    /// struct MyClock;
    /// impl Clock for MyClock {
    ///     fn elapsed(&self) -> Duration {
    ///         Duration::from_nanos(0)
    ///     }
    /// }
    ///
    /// let ratelimiter = Ratelimiter::with_clock(1000, MyClock);
    /// ```
    pub fn with_clock(rate: u64, clock: C) -> Self {
        Self {
            rate: AtomicU64::new(rate),
            max_tokens: AtomicU64::new(if rate == 0 { u64::MAX } else { rate }),
            tokens: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            last_refill_ns: AtomicU64::new(0),
            clock,
        }
    }

    /// Returns the current rate in tokens per second. 0 means unlimited.
    pub fn rate(&self) -> u64 {
        self.rate.load(Ordering::Relaxed)
    }

    /// Set a new rate in tokens per second. Takes effect immediately.
    ///
    /// When setting rate to 0 (unlimited), `max_tokens` is set to `u64::MAX`.
    /// When setting a nonzero rate, if `max_tokens` is currently `u64::MAX`
    /// (from unlimited mode or `new(0)`), it is reset to the new rate (1
    /// second of burst). Otherwise `max_tokens` is left unchanged.
    ///
    /// `max_tokens` is updated before `rate` so that concurrent readers
    /// never observe a nonzero rate with a stale `u64::MAX` max_tokens.
    ///
    /// The token bucket is not reset — it will naturally fill at the new rate.
    pub fn set_rate(&self, rate: u64) {
        if rate == 0 {
            self.max_tokens.store(u64::MAX, Ordering::Release);
        } else if self.max_tokens.load(Ordering::Acquire) == u64::MAX {
            self.max_tokens.store(rate, Ordering::Release);
        }
        self.rate.store(rate, Ordering::Release);
    }

    /// Returns the maximum number of tokens (burst capacity).
    pub fn max_tokens(&self) -> u64 {
        self.max_tokens.load(Ordering::Relaxed)
    }

    /// Set the maximum number of tokens (burst capacity).
    ///
    /// If the current available tokens exceed the new maximum, they are
    /// clamped down.
    ///
    /// Setting this to 0 will prevent any tokens from accumulating,
    /// effectively blocking all calls to `try_wait()` until a nonzero
    /// value is set. (When rate is 0, `try_wait()` always succeeds
    /// regardless of this setting.)
    pub fn set_max_tokens(&self, tokens: u64) {
        self.max_tokens.store(tokens, Ordering::Release);

        // Clamp available tokens down if needed
        let max_scaled = tokens.saturating_mul(TOKEN_SCALE);
        loop {
            let current = self.tokens.load(Ordering::Acquire);
            if current <= max_scaled {
                break;
            }
            if self
                .tokens
                .compare_exchange(current, max_scaled, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
            core::hint::spin_loop();
        }
    }

    /// Returns the approximate number of tokens currently available.
    ///
    /// This value is not updated automatically — tokens only accumulate
    /// when [`try_wait`](Ratelimiter::try_wait) is called. Do not use this
    /// as a pre-check; the value is inherently stale and `try_wait()` may
    /// still return `Err` even when `available()` returns nonzero.
    pub fn available(&self) -> u64 {
        self.tokens.load(Ordering::Relaxed) / TOKEN_SCALE
    }

    /// Returns the approximate number of whole tokens dropped during refill
    /// because the bucket was at capacity. This does not count `try_wait()`
    /// rejections. Sub-token precision is truncated.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed) / TOKEN_SCALE
    }

    /// Refill tokens based on elapsed time.
    fn refill(&self) {
        let rate = self.rate.load(Ordering::Relaxed);
        if rate == 0 {
            return;
        }

        // Wraps after ~584 years of uptime; not a practical concern.
        let now_ns = self.clock.elapsed().as_nanos() as u64;
        let last_ns = self.last_refill_ns.load(Ordering::Relaxed);
        let elapsed_ns = now_ns.saturating_sub(last_ns);

        // Only refill if at least 1μs has passed
        if elapsed_ns < 1_000 {
            return;
        }

        // tokens = rate * (elapsed_ns / 1_000_000_000) * TOKEN_SCALE
        //        = rate * elapsed_ns / 1_000
        let new_tokens = (rate as u128 * elapsed_ns as u128 / 1_000).min(u64::MAX as u128) as u64;

        if new_tokens == 0 {
            return;
        }

        // CAS to claim this refill window — if another thread won, skip
        if self
            .last_refill_ns
            .compare_exchange(last_ns, now_ns, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return;
        }

        // CAS loop to add tokens, capped at max_tokens
        let max_scaled = self
            .max_tokens
            .load(Ordering::Acquire)
            .saturating_mul(TOKEN_SCALE);
        loop {
            let current = self.tokens.load(Ordering::Acquire);
            let new_total = current.saturating_add(new_tokens).min(max_scaled);

            if new_total <= current {
                // Already at capacity — all new tokens are dropped
                self.dropped.fetch_add(new_tokens, Ordering::Relaxed);
                break;
            }

            if self
                .tokens
                .compare_exchange_weak(current, new_total, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                let added = new_total - current;
                if added < new_tokens {
                    self.dropped
                        .fetch_add(new_tokens - added, Ordering::Relaxed);
                }
                break;
            }
            core::hint::spin_loop();
        }
    }

    /// Non-blocking attempt to acquire a single token.
    ///
    /// On success, one token has been consumed. On failure, returns a
    /// `Duration` estimating when the next token will be available.
    /// The returned duration is a lower-bound estimate; the next
    /// `try_wait()` call after sleeping is not guaranteed to succeed
    /// under concurrent load.
    ///
    /// When the rate is 0 (unlimited), always succeeds.
    pub fn try_wait(&self) -> Result<(), Duration> {
        let rate = self.rate.load(Ordering::Relaxed);
        if rate == 0 {
            return Ok(());
        }

        self.refill();

        let cost = TOKEN_SCALE;
        loop {
            let current = self.tokens.load(Ordering::Acquire);
            if current < cost {
                let deficit = cost - current;
                let wait_ns = (deficit as u128 * 1_000 / rate as u128).max(1) as u64;
                return Err(Duration::from_nanos(wait_ns));
            }

            if self
                .tokens
                .compare_exchange_weak(current, current - cost, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(());
            }
            core::hint::spin_loop();
        }
    }
}

const _: () = {
    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}
    fn _check<C: Clock + Send + Sync>() {
        assert_send_sync::<Ratelimiter<C>>();
    }
};

impl<C> Debug for Ratelimiter<C>
where
    C: Clock,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ratelimiter")
            .field("rate", &self.rate.load(Ordering::Relaxed))
            .field("max_tokens", &self.max_tokens.load(Ordering::Relaxed))
            .field("available", &self.available())
            .finish()
    }
}

/// Builder for constructing a `Ratelimiter` with custom settings.
#[derive(Debug, Clone, Copy)]
#[must_use = "call .build() to construct the Ratelimiter"]
pub struct Builder<C> {
    rate: u64,
    max_tokens: Option<u64>,
    initial_available: u64,
    clock: C,
}

impl<C> Builder<C> {
    #[cfg_attr(not(any(feature = "std", test)), allow(dead_code))]
    pub(crate) fn with_clock(rate: u64, clock: C) -> Self {
        Self {
            rate,
            max_tokens: None,
            initial_available: 0,
            clock,
        }
    }

    /// Set the maximum number of tokens (burst capacity).
    ///
    /// Defaults to `rate` (1 second of burst), or `u64::MAX` when rate is 0
    /// (unlimited). Set higher for larger bursts or lower to restrict
    /// burstiness.
    pub fn max_tokens(mut self, tokens: u64) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set the number of tokens initially available.
    ///
    /// Defaults to 0. For admission control scenarios, you may want to start
    /// with some tokens available. For outbound request limiting, starting at
    /// 0 prevents bursts on application restart.
    pub fn initial_available(mut self, tokens: u64) -> Self {
        self.initial_available = tokens;
        self
    }

    /// Consume this builder and construct a `Ratelimiter`.
    pub fn build(self) -> Result<Ratelimiter<C>, Error>
    where
        C: Clock,
    {
        let max_tokens =
            self.max_tokens
                .unwrap_or(if self.rate == 0 { u64::MAX } else { self.rate });

        if max_tokens == 0 && self.rate != 0 {
            return Err(Error::MaxTokensTooLow);
        }

        if self.initial_available > max_tokens {
            return Err(Error::AvailableTokensTooHigh);
        }

        Ok(Ratelimiter {
            rate: AtomicU64::new(self.rate),
            max_tokens: AtomicU64::new(max_tokens),
            tokens: AtomicU64::new(self.initial_available.saturating_mul(TOKEN_SCALE)),
            dropped: AtomicU64::new(0),
            last_refill_ns: AtomicU64::new(0),
            clock: self.clock,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::AtomicU64;
    use core::time::Duration;
    use std::sync::Arc;

    #[derive(Clone, Debug)]
    struct TestClock {
        elapsed_ns: Arc<AtomicU64>,
    }

    impl TestClock {
        fn new() -> Self {
            Self {
                elapsed_ns: Arc::new(AtomicU64::new(0)),
            }
        }

        fn advance(&self, duration: Duration) {
            let elapsed_ns = duration.as_nanos().min(u64::MAX as u128) as u64;
            self.elapsed_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        }
    }

    impl Clock for TestClock {
        fn elapsed(&self) -> Duration {
            Duration::from_nanos(self.elapsed_ns.load(Ordering::Relaxed))
        }
    }

    #[test]
    fn unlimited() {
        let rl = Ratelimiter::with_clock(0, TestClock::new());
        for _ in 0..1000 {
            assert!(rl.try_wait().is_ok());
        }
    }

    #[test]
    fn basic_rate() {
        let clock = TestClock::new();
        let rl = Builder::with_clock(1000, clock)
            .initial_available(10)
            .build()
            .unwrap();

        // Should be able to consume the initial 10 tokens
        for _ in 0..10 {
            assert!(rl.try_wait().is_ok());
        }
        // Next should fail (not enough time for more tokens)
        assert!(rl.try_wait().is_err());
    }

    #[test]
    fn refill_over_time() {
        let clock = TestClock::new();
        let rl = Ratelimiter::with_clock(1000, clock.clone());

        // Advance 100ms — should accumulate ~100 tokens
        clock.advance(Duration::from_millis(100));

        let mut count = 0;
        while rl.try_wait().is_ok() {
            count += 1;
        }

        // Allow some tolerance for timing
        assert!(count >= 50, "expected >= 50, got {count}");
        assert!(count <= 200, "expected <= 200, got {count}");
    }

    #[test]
    fn burst_capacity() {
        let clock = TestClock::new();
        let rl = Builder::with_clock(100, clock)
            .max_tokens(10)
            .initial_available(10)
            .build()
            .unwrap();

        // Can consume burst
        for _ in 0..10 {
            assert!(rl.try_wait().is_ok());
        }
        assert!(rl.try_wait().is_err());
    }

    #[test]
    fn idle_does_not_exceed_capacity() {
        let clock = TestClock::new();
        let rl = Builder::with_clock(1000, clock.clone())
            .max_tokens(10)
            .build()
            .unwrap();

        // Advance long enough to accumulate way more than max_tokens
        clock.advance(Duration::from_millis(100));

        let mut count = 0;
        while rl.try_wait().is_ok() {
            count += 1;
        }

        assert!(count <= 10, "expected <= 10, got {count}");
    }

    #[test]
    fn set_rate() {
        let clock = TestClock::new();
        let rl = Ratelimiter::with_clock(100, clock.clone());

        // Accumulate some tokens
        clock.advance(Duration::from_millis(50));

        // Increase rate 10x
        rl.set_rate(1000);

        // Advance again — should accumulate faster
        clock.advance(Duration::from_millis(50));

        let mut count = 0;
        while rl.try_wait().is_ok() {
            count += 1;
        }

        // Should have tokens from both periods
        assert!(count >= 30, "expected >= 30, got {count}");
    }

    #[test]
    fn set_max_tokens_clamps_down() {
        let clock = TestClock::new();
        let rl = Builder::with_clock(1000, clock)
            .max_tokens(100)
            .initial_available(100)
            .build()
            .unwrap();

        assert_eq!(rl.available(), 100);

        rl.set_max_tokens(10);
        assert!(rl.available() <= 10);
    }

    #[test]
    fn try_wait_returns_duration_hint() {
        let rl = Ratelimiter::with_clock(1000, TestClock::new());
        // No tokens available yet and not enough time passed
        let err = rl.try_wait().unwrap_err();
        // Should hint at ~1ms (1_000_000ns for 1000/s)
        assert_eq!(err, Duration::from_micros(1000));
    }

    #[test]
    fn builder_error_available_too_high() {
        let clock = TestClock::new();
        let result = Builder::with_clock(100, clock)
            .max_tokens(10)
            .initial_available(20)
            .build();
        assert!(matches!(result, Err(Error::AvailableTokensTooHigh)));
    }

    #[test]
    fn dropped_tokens() {
        let clock = TestClock::new();
        let rl = Builder::with_clock(1000, clock.clone())
            .max_tokens(10)
            .build()
            .unwrap();

        // Advance long enough for many tokens to try to accumulate
        clock.advance(Duration::from_millis(100));

        // Trigger a refill
        let _ = rl.try_wait();

        // Should have dropped excess tokens
        assert!(rl.dropped() > 0, "expected dropped > 0");
    }

    #[test]
    fn wait_loop() {
        let clock = TestClock::new();
        let rl = Ratelimiter::with_clock(10_000, clock.clone());
        let mut count = 0;

        while clock.elapsed() < Duration::from_millis(100) {
            match rl.try_wait() {
                Ok(()) => count += 1,
                Err(wait) => clock.advance(wait),
            }
        }

        // 10k/s for 100ms ≈ 1000
        assert!(count >= 500, "expected >= 500, got {count}");
        assert!(count <= 2000, "expected <= 2000, got {count}");
    }

    #[test]
    fn high_rate() {
        // Verify no overflow/truncation at very high rates
        let clock = TestClock::new();
        let rl = Ratelimiter::with_clock(1_000_000_000_000, clock.clone()); // 1 trillion/s
        clock.advance(Duration::from_millis(10));
        assert!(rl.try_wait().is_ok());
    }

    #[test]
    fn try_wait_hint_at_high_rate() {
        // Verify the wait hint is at least 1ns even at very high rates
        let rl = Ratelimiter::with_clock(10_000_000_000, TestClock::new()); // 10B/s
        let err = rl.try_wait().unwrap_err();
        assert!(err >= Duration::from_nanos(1));
    }

    #[test]
    fn unlimited_then_set_rate() {
        let clock = TestClock::new();
        let rl = Ratelimiter::with_clock(0, clock.clone());
        assert!(rl.try_wait().is_ok()); // unlimited

        rl.set_rate(1000);
        clock.advance(Duration::from_millis(50));
        assert!(rl.try_wait().is_ok()); // set_rate alone resets max_tokens
    }

    #[test]
    fn set_rate_to_zero_and_back() {
        let clock = TestClock::new();
        let rl = Ratelimiter::with_clock(1000, clock.clone());

        // Switch to unlimited
        rl.set_rate(0);
        assert_eq!(rl.max_tokens(), u64::MAX);
        for _ in 0..100 {
            assert!(rl.try_wait().is_ok());
        }

        // Switch back to rate-limited
        rl.set_rate(500);
        assert_eq!(rl.max_tokens(), 500);

        // Should work after some time
        clock.advance(Duration::from_millis(50));
        assert!(rl.try_wait().is_ok());
    }

    #[test]
    fn builder_error_max_tokens_zero() {
        let clock = TestClock::new();
        let result = Builder::with_clock(100, clock).max_tokens(0).build();
        assert!(matches!(result, Err(Error::MaxTokensTooLow)));
    }

    #[test]
    fn max_tokens_zero() {
        let clock = TestClock::new();
        let rl = Ratelimiter::with_clock(1000, clock.clone());
        rl.set_max_tokens(0);
        clock.advance(Duration::from_millis(10));
        // With max_tokens=0, no tokens can accumulate
        assert!(rl.try_wait().is_err());
        // Restore capacity
        rl.set_max_tokens(1000);
        clock.advance(Duration::from_millis(10));
        assert!(rl.try_wait().is_ok());
    }

    // Test std convenience APIs when std feature is enabled
    #[cfg(feature = "std")]
    #[test]
    fn std_convenience_apis() {
        // Test Ratelimiter::new()
        let rl = Ratelimiter::new(1000);
        assert_eq!(rl.rate(), 1000);

        // Test Ratelimiter::builder()
        let rl = Ratelimiter::builder(1000)
            .max_tokens(100)
            .initial_available(50)
            .build()
            .unwrap();
        assert_eq!(rl.max_tokens(), 100);
        assert_eq!(rl.available(), 50);

        // Test StdClock directly
        let clock = StdClock::new();
        let rl = Ratelimiter::with_clock(1000, clock);
        assert_eq!(rl.rate(), 1000);
    }

    #[cfg(feature = "std")]
    #[test]
    fn multithread() {
        use std::sync::Arc;
        use std::vec::Vec;

        let rl = Arc::new(
            Ratelimiter::builder(10_000)
                .max_tokens(10_000)
                .build()
                .unwrap(),
        );
        let duration = Duration::from_millis(200);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let rl = rl.clone();
                std::thread::spawn(move || {
                    let start = std::time::Instant::now();
                    let mut count = 0u64;
                    while start.elapsed() < duration {
                        if rl.try_wait().is_ok() {
                            count += 1;
                        }
                    }
                    count
                })
            })
            .collect();

        let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();

        assert!(total >= 1000, "expected >= 1000, got {total}");
        assert!(total <= 4000, "expected <= 4000, got {total}");
    }
}

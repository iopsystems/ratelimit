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
//! // Sub-Hz rates: 1 token per minute
//! let ratelimiter = Ratelimiter::builder(1)
//!     .period(std::time::Duration::from_secs(60))
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
impl Default for StdClock {
    fn default() -> Self {
        Self::new()
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
#[non_exhaustive]
pub enum Error {
    #[error("initial available tokens cannot exceed max tokens")]
    AvailableTokensTooHigh,
    #[error("max tokens must be at least 1")]
    MaxTokensTooLow,
    #[error("period must be greater than zero")]
    PeriodTooShort,
}

/// Failure modes for [`Ratelimiter::try_wait`] and [`Ratelimiter::try_wait_n`].
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TryWaitError {
    /// Not enough tokens are available right now. Retry after the returned
    /// duration — same semantics as the `Err` from [`Ratelimiter::try_wait`].
    #[error("insufficient tokens; retry after {0:?}")]
    Insufficient(Duration),
    /// `n` exceeds the bucket's current [`max_tokens`](Ratelimiter::max_tokens).
    /// Waiting will never satisfy the request; the caller must reduce `n` or
    /// increase `max_tokens`.
    #[error("requested tokens exceed bucket capacity")]
    ExceedsCapacity,
}

/// A lock-free token bucket ratelimiter.
///
/// Tokens accumulate continuously based on elapsed time at a rate of
/// `rate` tokens per `period` — where `period` defaults to 1 second, so
/// `rate` is interpreted as tokens per second unless otherwise configured.
/// Set a longer `period` (via [`Builder::period`]) to express sub-Hz
/// rates such as "1 token per minute". A `rate` of 0 means unlimited
/// (no rate limiting).
///
/// The `C` type parameter defaults to [`StdClock`] when the `std` feature
/// is enabled, so `Ratelimiter` without generics resolves to the standard
/// configuration. In `no_std` builds the clock must be specified.
#[must_use]
#[cfg(feature = "std")]
pub struct Ratelimiter<C: Clock = StdClock> {
    /// Tokens added per `period_ns`. 0 = unlimited.
    rate: AtomicU64,
    /// Length of the rate period in nanoseconds.
    period_ns: AtomicU64,
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

#[must_use]
#[cfg(not(feature = "std"))]
pub struct Ratelimiter<C: Clock> {
    /// Tokens added per `period_ns`. 0 = unlimited.
    rate: AtomicU64,
    /// Length of the rate period in nanoseconds.
    period_ns: AtomicU64,
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

/// Default rate period: one second.
const DEFAULT_PERIOD_NS: u64 = 1_000_000_000;

/// Estimate the wait time for `deficit` scaled tokens at `rate` tokens per
/// `period_ns`. Result is clamped to at least 1ns and at most `u64::MAX` ns.
#[inline]
fn wait_ns_for_deficit(deficit: u64, rate: u64, period_ns: u64) -> u64 {
    // wait_ns = deficit * period_ns / (rate * TOKEN_SCALE)
    let denom = (rate as u128).saturating_mul(TOKEN_SCALE as u128).max(1);
    ((deficit as u128).saturating_mul(period_ns as u128) / denom)
        .max(1)
        .min(u64::MAX as u128) as u64
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
            period_ns: AtomicU64::new(DEFAULT_PERIOD_NS),
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

    /// Set a new rate (tokens per [`period`](Ratelimiter::period)). Takes
    /// effect immediately.
    ///
    /// When setting rate to 0 (unlimited), `max_tokens` is set to `u64::MAX`.
    /// When setting a nonzero rate, if `max_tokens` is currently `u64::MAX`
    /// (from unlimited mode or `new(0)`), it is reset to the new rate (one
    /// period of burst). Otherwise `max_tokens` is left unchanged.
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

    /// Returns the current rate period.
    ///
    /// The ratelimiter accumulates [`rate`](Ratelimiter::rate) tokens per
    /// period. Defaults to one second.
    pub fn period(&self) -> Duration {
        Duration::from_nanos(self.period_ns.load(Ordering::Relaxed))
    }

    /// Set the rate period. Takes effect immediately.
    ///
    /// A longer period produces a slower effective rate. For example,
    /// `rate = 1` with `period = Duration::from_secs(60)` is one token per
    /// minute.
    ///
    /// A zero-length period is silently clamped to 1 nanosecond to avoid
    /// division by zero — use [`Builder::period`] with a non-zero value at
    /// construction to get a clear error instead.
    pub fn set_period(&self, period: Duration) {
        let ns = period.as_nanos().min(u64::MAX as u128) as u64;
        self.period_ns.store(ns.max(1), Ordering::Release);
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

        // scaled_tokens = rate * TOKEN_SCALE * elapsed_ns / period_ns
        let period_ns = self.period_ns.load(Ordering::Relaxed).max(1);
        let new_tokens = ((rate as u128)
            .saturating_mul(elapsed_ns as u128)
            .saturating_mul(TOKEN_SCALE as u128)
            / period_ns as u128)
            .min(u64::MAX as u128) as u64;

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

        let period_ns = self.period_ns.load(Ordering::Relaxed).max(1);
        let cost = TOKEN_SCALE;
        loop {
            let current = self.tokens.load(Ordering::Acquire);
            if current < cost {
                let deficit = cost - current;
                let wait_ns = wait_ns_for_deficit(deficit, rate, period_ns);
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

    /// Non-blocking attempt to atomically acquire `n` tokens.
    ///
    /// Like [`try_wait`](Ratelimiter::try_wait), but consumes `n` tokens in
    /// a single atomic operation — partial consumption never occurs.
    ///
    /// `try_wait_n(0)` is a no-op and always returns `Ok(())`. When rate is
    /// 0 (unlimited), always succeeds.
    ///
    /// # Errors
    ///
    /// - [`TryWaitError::Insufficient`] — not enough tokens right now;
    ///   retry after the returned duration.
    /// - [`TryWaitError::ExceedsCapacity`] — `n` is greater than the
    ///   current [`max_tokens`](Ratelimiter::max_tokens). The bucket can
    ///   never hold enough, so waiting will not help; reduce `n` or
    ///   increase `max_tokens`.
    pub fn try_wait_n(&self, n: u64) -> Result<(), TryWaitError> {
        let rate = self.rate.load(Ordering::Relaxed);
        if rate == 0 {
            return Ok(());
        }
        if n == 0 {
            return Ok(());
        }

        if n > self.max_tokens.load(Ordering::Relaxed) {
            return Err(TryWaitError::ExceedsCapacity);
        }

        self.refill();

        let period_ns = self.period_ns.load(Ordering::Relaxed).max(1);
        let cost = n.saturating_mul(TOKEN_SCALE);
        loop {
            let current = self.tokens.load(Ordering::Acquire);
            if current < cost {
                let deficit = cost - current;
                let wait_ns = wait_ns_for_deficit(deficit, rate, period_ns);
                return Err(TryWaitError::Insufficient(Duration::from_nanos(wait_ns)));
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
            .field("period", &self.period())
            .field("max_tokens", &self.max_tokens.load(Ordering::Relaxed))
            .field("available", &self.available())
            .finish()
    }
}

/// Builder for constructing a `Ratelimiter` with custom settings.
///
/// The `C` type parameter defaults to [`StdClock`] when the `std` feature
/// is enabled, matching [`Ratelimiter`]'s default.
#[derive(Debug, Clone, Copy)]
#[must_use = "call .build() to construct the Ratelimiter"]
#[cfg(feature = "std")]
pub struct Builder<C = StdClock> {
    rate: u64,
    period: Duration,
    max_tokens: Option<u64>,
    initial_available: u64,
    clock: C,
}

#[derive(Debug, Clone, Copy)]
#[must_use = "call .build() to construct the Ratelimiter"]
#[cfg(not(feature = "std"))]
pub struct Builder<C> {
    rate: u64,
    period: Duration,
    max_tokens: Option<u64>,
    initial_available: u64,
    clock: C,
}

impl<C> Builder<C> {
    /// Create a builder configured with the given rate and clock.
    ///
    /// This is the builder constructor for `no_std` environments or for any
    /// caller that wants to supply a custom [`Clock`]. For the standard
    /// library clock, use [`Ratelimiter::builder`].
    pub fn with_clock(rate: u64, clock: C) -> Self {
        Self {
            rate,
            period: Duration::from_nanos(DEFAULT_PERIOD_NS),
            max_tokens: None,
            initial_available: 0,
            clock,
        }
    }

    /// Set the period over which `rate` tokens accumulate.
    ///
    /// Defaults to one second, so `rate` is "tokens per second" by default.
    /// Set a longer period to express sub-Hz rates — for example, `rate = 1`
    /// with `period = Duration::from_secs(60)` is one token per minute.
    ///
    /// A zero-length period is rejected at [`build`](Builder::build) with
    /// [`Error::PeriodTooShort`].
    pub fn period(mut self, period: Duration) -> Self {
        self.period = period;
        self
    }

    /// Set the maximum number of tokens (burst capacity).
    ///
    /// Defaults to `rate` (one period of burst), or `u64::MAX` when rate is 0
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
        let period_ns = self.period.as_nanos();
        if period_ns == 0 {
            return Err(Error::PeriodTooShort);
        }
        let period_ns = period_ns.min(u64::MAX as u128) as u64;

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
            period_ns: AtomicU64::new(period_ns),
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

    // Proves the `C: Clock = StdClock` default works in type position —
    // callers can name `Ratelimiter` / `Builder` without generics.
    #[cfg(feature = "std")]
    #[test]
    fn type_default_clock() {
        let rl: Ratelimiter = Ratelimiter::new(1000);
        assert_eq!(rl.rate(), 1000);

        let b: Builder = Ratelimiter::builder(1000);
        let rl = b.max_tokens(10).build().unwrap();
        assert_eq!(rl.max_tokens(), 10);
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

    #[test]
    fn try_wait_n_basic() {
        let clock = TestClock::new();
        let rl = Builder::with_clock(1000, clock)
            .initial_available(10)
            .build()
            .unwrap();
        assert!(rl.try_wait_n(5).is_ok());
        assert!(rl.try_wait_n(5).is_ok());
        assert!(matches!(
            rl.try_wait_n(1),
            Err(TryWaitError::Insufficient(_))
        ));
    }

    #[test]
    fn try_wait_n_zero_is_noop() {
        let rl = Ratelimiter::with_clock(1000, TestClock::new());
        // No tokens yet, but n=0 still succeeds and consumes nothing.
        assert!(rl.try_wait_n(0).is_ok());
        assert_eq!(rl.available(), 0);
    }

    #[test]
    fn try_wait_n_unlimited() {
        let rl = Ratelimiter::with_clock(0, TestClock::new());
        assert!(rl.try_wait_n(1_000_000).is_ok());
    }

    #[test]
    fn try_wait_n_does_not_partially_consume() {
        let clock = TestClock::new();
        let rl = Builder::with_clock(1000, clock)
            .initial_available(5)
            .build()
            .unwrap();
        // Asking for more than available (but <= max_tokens) must fail without consuming.
        assert!(matches!(
            rl.try_wait_n(10),
            Err(TryWaitError::Insufficient(_))
        ));
        for _ in 0..5 {
            assert!(rl.try_wait().is_ok());
        }
    }

    #[test]
    fn try_wait_n_exceeds_capacity() {
        let clock = TestClock::new();
        let rl = Builder::with_clock(1000, clock)
            .max_tokens(10)
            .build()
            .unwrap();
        assert_eq!(rl.try_wait_n(100), Err(TryWaitError::ExceedsCapacity));
    }

    // "1 token per minute" — sub-Hz rate expressed via a 60s period.
    #[test]
    fn sub_hz_refill() {
        let clock = TestClock::new();
        let rl = Builder::with_clock(1, clock.clone())
            .period(Duration::from_secs(60))
            .build()
            .unwrap();

        // Not enough time — no token yet.
        clock.advance(Duration::from_secs(30));
        assert!(rl.try_wait().is_err());

        // A full period should produce one token.
        clock.advance(Duration::from_secs(30));
        assert!(rl.try_wait().is_ok());
        // And no second token without another full period.
        assert!(rl.try_wait().is_err());
    }

    #[test]
    fn sub_hz_wait_hint() {
        let rl = Builder::with_clock(1, TestClock::new())
            .period(Duration::from_secs(60))
            .build()
            .unwrap();
        // Empty bucket at 1/minute: next token ~60s away.
        let wait = rl.try_wait().unwrap_err();
        assert_eq!(wait, Duration::from_secs(60));
    }

    #[test]
    fn set_period_changes_rate() {
        let clock = TestClock::new();
        let rl = Ratelimiter::with_clock(1, clock.clone());
        assert_eq!(rl.period(), Duration::from_secs(1));

        // Stretch the period to 10s — now 1 token every 10s.
        rl.set_period(Duration::from_secs(10));
        assert_eq!(rl.period(), Duration::from_secs(10));

        clock.advance(Duration::from_secs(5));
        assert!(rl.try_wait().is_err());
        clock.advance(Duration::from_secs(5));
        assert!(rl.try_wait().is_ok());
    }

    #[test]
    fn builder_error_period_zero() {
        let result = Builder::with_clock(1, TestClock::new())
            .period(Duration::ZERO)
            .build();
        assert!(matches!(result, Err(Error::PeriodTooShort)));
    }
}

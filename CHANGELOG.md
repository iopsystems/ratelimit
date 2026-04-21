# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-04-21

### Added

- `no_std` support via a new `Clock` trait (`StdClock` is the default
  under the `std` feature).
- `Ratelimiter::try_wait_n(n)` for atomic multi-token acquisition.
- `TryWaitError` distinguishing `Insufficient(Duration)` from
  `ExceedsCapacity`.
- Configurable rate period: `Builder::period`, `Ratelimiter::period` /
  `set_period` / `set_rate_per`. Enables sub-Hz rates.
- `impl Default for StdClock`.

### Changed

- `Ratelimiter` and `Builder` are generic over `Clock`; `C` defaults to
  `StdClock` under `std`.
- `try_wait` now returns `Result<(), TryWaitError>`.
- `Error` and `TryWaitError` are `#[non_exhaustive]`.
- New `Error::PeriodTooShort` variant.

### Fixed

- Sub-second and sub-Hz rate limits (regression from 1.0.0).
- `try_wait` no longer spins when `max_tokens == 0`; returns
  `ExceedsCapacity`.

## [1.0.0] - 2026-03-20

First release with changelog.

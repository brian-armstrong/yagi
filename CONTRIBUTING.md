# Contributing to Yagi

Thank you for your interest in contributing to Yagi! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository and clone your fork
2. Ensure you have Rust installed (stable toolchain)
3. Run `cargo build` to verify your setup
4. Run `cargo test` to ensure tests pass

## How to Contribute

### Reporting Bugs

- Check existing issues to avoid duplicates
- Include a minimal reproducible example when possible
- Describe expected vs actual behavior
- Include Rust version and platform information

### Suggesting Features

- Open an issue describing the use case
- For DSP features, reference relevant literature or existing implementations if applicable

### Submitting Pull Requests

1. Create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Add tests for new functionality
4. Ensure `cargo test` passes
5. Ensure `cargo fmt` passes
6. Open a PR with a clear description of the changes

### Expectations

Yagi is maintained on a best-effort basis. Please keep in mind:

- Response times may vary — this is a side project, not a full-time job
- Feature requests are welcome but may not be implemented
- PRs are appreciated, but acceptance is not guaranteed
- The maintainer has final discretion on what gets merged

This isn't meant to discourage contributions — quite the opposite. Clear expectations help everyone have a better experience. If you're unsure whether a contribution would be welcome, open an issue to discuss before investing significant time.

## Code Standards

- Follow standard Rust conventions and idioms
- Use `rustfmt` for formatting
- Add documentation comments for public APIs
- Include tests for new functionality

## Testing

Yagi inherits a comprehensive test suite from liquid-dsp. When porting or adding functionality:

- Port corresponding tests from liquid-dsp when applicable (see [LIQUID_COMPAT.md](LIQUID_COMPAT.md))
- Use the `#[autotest_annotate]` macro to link tests to their liquid-dsp counterparts
- Aim for test parity with the original C implementation

## Development Process

This project was initially developed with assistance from large language models. Contributions from all sources — human or AI-assisted — are welcome. What matters is code quality, correctness, and clear communication.

If you use AI tools in your contributions, no special disclosure is required, but please ensure you understand and can explain any code you submit.

## Attribution

Yagi is a Rust port of [liquid-dsp](https://liquidsdr.org/) by Joseph Gaeddert. When contributing code derived from liquid-dsp, the existing dual copyright in the LICENSE file covers this attribution. For code from other sources, please ensure proper attribution and license compatibility (MIT).

## Questions?

Open an issue for questions about contributing. We're happy to help newcomers get started.

//! Fixed-length sample histories, scalar delays, and FIFO ring buffers.
//!
//! [`Window`] exposes a fixed-size history in oldest-to-newest order.
//! [`WindowedDelay`]
//! returns one delayed sample at a time.
//! [`CircularBuffer`] holds a variable-length
//! FIFO queue and supports zero-copy reads and writes.

mod circular_buffer;
mod window;
mod windowed_delay;

pub use circular_buffer::CircularBuffer;
pub use window::Window;
pub use windowed_delay::WindowedDelay;

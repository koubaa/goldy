//! Interactive demos for documentation
//!
//! All demos require Slang shaders compiled via slang-wasm in JavaScript.
//! The compiled shader source is passed to each create_*_demo() function.

pub mod plasma;
pub mod triangle;
pub mod mandelbrot;
pub mod starfield;
pub mod digital_clock;
pub mod gradient;
pub mod particles;
pub mod tunnel;
pub mod spinning_cube;

pub use plasma::*;
pub use triangle::*;
pub use mandelbrot::*;
pub use starfield::*;
pub use digital_clock::*;
pub use gradient::*;
pub use particles::*;
pub use tunnel::*;
pub use spinning_cube::*;

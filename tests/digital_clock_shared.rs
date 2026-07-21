#[path = "../examples/digital_clock_shared.rs"]
mod digital_clock_shared;

use digital_clock_shared::{generate_clock_vertices, TimeData};
use goldy::Color;

#[test]
fn digital_clock_shared_time_data_from_elapsed() {
    let time = TimeData::from_elapsed_secs(3661);
    assert_eq!(time.hours, 1);
    assert_eq!(time.minutes, 1);
    assert_eq!(time.seconds, 1);
}

#[test]
fn digital_clock_shared_time_data_to_digits() {
    let time = TimeData {
        hours: 12,
        minutes: 34,
        seconds: 56,
    };
    assert_eq!(time.to_digits(), [1, 2, 10, 3, 4, 10, 5, 6]);
}

#[test]
fn digital_clock_shared_vertex_generation() {
    let time = TimeData::from_elapsed_secs(0);
    let vertices = generate_clock_vertices(time, Color::GREEN, 800, 600);
    assert!(!vertices.is_empty());
}

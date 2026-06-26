pub struct Time {
    elapsed_time: f32,
    delta_time: f32,
}

impl Time {
    pub fn new() -> Self {
        Time {
            elapsed_time: 0.0,
            delta_time: 0.0,
        }
    }
}
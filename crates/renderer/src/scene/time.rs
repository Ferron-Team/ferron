pub struct Time {
    elapsed_time: f32,
    delta_time: f32,
}

impl Default for Time {
    fn default() -> Self {
        Self::new()
    }
}

impl Time {
    pub fn new() -> Self {
        Time {
            elapsed_time: 0.0,
            delta_time: 0.0,
        }
    }

    pub fn update(&mut self, delta_time: f32) {
        self.elapsed_time += delta_time;
        self.delta_time = delta_time;
    }

    pub fn delta_time(&self) -> f32 {
        self.delta_time
    }
}
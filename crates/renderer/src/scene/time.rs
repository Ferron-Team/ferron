   pub struct Time {
    elapsed_time: f32,
    delta_time: f32,
    frame_count: u64,
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
            frame_count: 0,
        }
    }

    pub fn update(&mut self, delta_time: f32) {
        self.elapsed_time += delta_time;
        self.delta_time = delta_time;
        self.frame_count += 1;
    }

    pub fn delta_time(&self) -> f32 {
        self.delta_time
    }

    pub fn elapsed_time(&self) -> f32 {
        self.elapsed_time
    }

    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

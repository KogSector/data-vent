use std::time::Instant;

#[allow(dead_code)]
pub struct PerformanceProfiler {
    measurements: Vec<(String, f64)>,
}

#[allow(dead_code)]
impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }

    pub fn measure<F, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed().as_secs_f64() * 1000.0;
        self.measurements.push((name.to_string(), duration));
        tracing::info!("PERF: {} took {:.2}ms", name, duration);
        result
    }

    pub async fn measure_async<F, Fut, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        let start = Instant::now();
        let result = f().await;
        let duration = start.elapsed().as_secs_f64() * 1000.0;
        self.measurements.push((name.to_string(), duration));
        tracing::info!("PERF: {} took {:.2}ms", name, duration);
        result
    }

    pub fn get_measurements(&self) -> &[(String, f64)] {
        &self.measurements
    }

    pub fn total_duration_ms(&self) -> f64 {
        self.measurements.iter().map(|(_, d)| *d).sum()
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_measure() {
        let mut profiler = PerformanceProfiler::new();
        let val = profiler.measure("test_op", || 10 + 20);
        assert_eq!(val, 30);
        assert_eq!(profiler.get_measurements().len(), 1);
        assert_eq!(profiler.get_measurements()[0].0, "test_op");
        assert!(profiler.total_duration_ms() >= 0.0);
    }

    #[tokio::test]
    async fn test_profiler_measure_async() {
        let mut profiler = PerformanceProfiler::new();
        let val = profiler
            .measure_async("test_async_op", || async { 42 })
            .await;
        assert_eq!(val, 42);
        assert_eq!(profiler.get_measurements().len(), 1);
    }
}

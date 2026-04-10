/*!
# cuda-perception

Raw sensor data → structured understanding.

Perception is the agent's window to the world. Raw sensor readings are
noise without processing. This pipeline transforms noisy signals into
structured percepts the agent can reason about.

- Signal filtering (moving average, low-pass)
- Feature extraction from raw data
- Object detection and tracking
- Scene composition
- Confidence-aware processing (uncertain signals don't crash the pipeline)
*/

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// A raw sensor reading
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawReading {
    pub sensor_id: String,
    pub data_type: DataType,
    pub values: Vec<f64>,
    pub timestamp: u64,
    pub noise_level: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    Scalar,     // single value (temperature, pressure)
    Vector,     // direction (accelerometer, velocity)
    Image,      // pixel grid (represented as flat vec)
    Audio,      // amplitude samples
    Binary,     // on/off signal
}

/// A processed percept — what the agent actually "sees"
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Percept {
    pub id: u64,
    pub kind: PerceptKind,
    pub properties: HashMap<String, f64>,
    pub labels: Vec<String>,
    pub confidence: f64,
    pub source_sensors: Vec<String>,
    pub timestamp: u64,
    pub ttl_ms: u64,        // how long this percept is valid
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptKind {
    Object,      // detected thing with location
    Event,       // detected change
    Property,    // measured quality
    Spatial,     // spatial relationship
    Temporal,    // timing relationship
    Absence,     // something expected but not detected
}

impl Percept {
    pub fn new(kind: PerceptKind) -> Self {
        Percept { id: 0, kind, properties: HashMap::new(), labels: vec![], confidence: 0.5, source_sensors: vec![], timestamp: now(), ttl_ms: 5000 }
    }
}

/// Signal filter — smooths noisy data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignalFilter {
    pub window: VecDeque<f64>,
    pub window_size: usize,
    pub method: FilterMethod,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum FilterMethod {
    MovingAverage,
    Exponential(EMAAlpha),  // 0.0 = smooth, 1.0 = raw
    Median,
    LowPass(f64),           // cutoff frequency as fraction
}

#[derive(Clone, Copy, Debug)]
pub struct EMAAlpha(pub f64);

impl SignalFilter {
    pub fn new(method: FilterMethod, window_size: usize) -> Self {
        SignalFilter { window: VecDeque::with_capacity(window_size), window_size, method }
    }

    /// Push value, get filtered output
    pub fn process(&mut self, value: f64) -> f64 {
        if self.window.len() >= self.window_size { self.window.pop_front(); }
        self.window.push_back(value);

        match self.method {
            FilterMethod::MovingAverage => {
                if self.window.is_empty() { return 0.0; }
                self.window.iter().sum::<f64>() / self.window.len() as f64
            }
            FilterMethod::Exponential(EMAAlpha(alpha)) => {
                let prev = self.window.iter().rev().nth(1).copied().unwrap_or(value);
                alpha * value + (1.0 - alpha) * prev
            }
            FilterMethod::Median => {
                let mut sorted: Vec<f64> = self.window.iter().copied().collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = sorted.len() / 2;
                if sorted.len() % 2 == 0 { (sorted[mid - 1] + sorted[mid]) / 2.0 } else { sorted[mid] }
            }
            FilterMethod::LowPass(cutoff) => {
                let alpha = cutoff.min(0.99);
                let prev = self.window.iter().rev().nth(1).copied().unwrap_or(value);
                alpha * prev + (1.0 - alpha) * value
            }
        }
    }

    /// Reset filter state
    pub fn reset(&mut self) { self.window.clear(); }
}

/// Object tracker — tracks objects across percept frames
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrackedObject {
    pub id: String,
    pub kind: String,
    pub position: (f64, f64),
    pub velocity: (f64, f64),
    pub confidence: f64,
    pub last_seen: u64,
    pub track_length: u32,
    pub properties: HashMap<String, f64>,
}

impl TrackedObject {
    /// Update with new observation
    pub fn update(&mut self, pos: (f64, f64), timestamp: u64) {
        let dt = (timestamp.saturating_sub(self.last_seen) as f64 / 1000.0).max(0.001);
        self.velocity = ((pos.0 - self.position.0) / dt, (pos.1 - self.position.1) / dt);
        self.position = pos;
        self.last_seen = timestamp;
        self.track_length += 1;
        self.confidence = (self.confidence + 0.1).min(1.0);
    }

    /// Decay if not seen recently
    pub fn decay(&mut self, now: u64, max_age_ms: u64) {
        let age = now.saturating_sub(self.last_seen);
        if age > max_age_ms { self.confidence *= 0.5; }
    }
}

/// Scene — composed view of the world at a moment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scene {
    pub percepts: Vec<Percept>,
    pub objects: Vec<TrackedObject>,
    pub timestamp: u64,
    pub confidence: f64,
}

impl Scene {
    pub fn new() -> Self {
        Scene { percepts: vec![], objects: vec![], timestamp: now(), confidence: 0.5 }
    }

    /// Expire old percepts
    pub fn expire(&mut self, now: u64) {
        self.percepts.retain(|p| now - p.timestamp < p.ttl_ms);
    }

    /// Find objects by label
    pub fn find_objects(&self, label: &str) -> Vec<&TrackedObject> {
        self.objects.iter().filter(|o| o.kind.contains(label)).collect()
    }

    /// Scene summary
    pub fn summary(&self) -> String {
        format!("Scene: {} percepts, {} objects, confidence={:.2}", self.percepts.len(), self.objects.len(), self.confidence)
    }
}

/// The perception pipeline
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerceptionPipeline {
    pub filters: HashMap<String, SignalFilter>,
    pub objects: HashMap<String, TrackedObject>,
    pub current_scene: Scene,
    pub max_objects: usize,
    pub object_ttl_ms: u64,
    pub next_percept_id: u64,
}

impl PerceptionPipeline {
    pub fn new() -> Self {
        PerceptionPipeline { filters: HashMap::new(), objects: HashMap::new(), current_scene: Scene::new(), max_objects: 50, object_ttl_ms: 30_000, next_percept_id: 1 }
    }

    /// Register a filter for a sensor
    pub fn add_filter(&mut self, sensor_id: &str, method: FilterMethod, window_size: usize) {
        self.filters.insert(sensor_id.to_string(), SignalFilter::new(method, window_size));
    }

    /// Process a raw reading into percepts
    pub fn process(&mut self, reading: RawReading) -> Vec<Percept> {
        let mut percepts = vec![];

        // Apply filter
        let filtered_value = if let Some(filter) = self.filters.get_mut(&reading.sensor_id) {
            reading.values.iter().map(|v| filter.process(*v)).collect()
        } else {
            reading.values.clone()
        };

        // Create basic percepts based on data type
        match reading.data_type {
            DataType::Scalar => {
                if let Some(&val) = filtered_value.first() {
                    let mut p = Percept::new(PerceptKind::Property);
                    p.labels.push(reading.sensor_id.clone());
                    p.properties.insert("value".into(), val);
                    p.confidence = (1.0 - reading.noise_level).max(0.1);
                    p.source_sensors.push(reading.sensor_id.clone());
                    p.id = self.next_percept_id;
                    self.next_percept_id += 1;
                    percepts.push(p);
                }
            }
            DataType::Vector => {
                if filtered_value.len() >= 2 {
                    let mut p = Percept::new(PerceptKind::Spatial);
                    p.labels.push("vector".into());
                    p.properties.insert("x".into(), filtered_value[0]);
                    p.properties.insert("y".into(), filtered_value[1]);
                    p.properties.insert("magnitude".into(), (filtered_value[0].powi(2) + filtered_value[1].powi(2)).sqrt());
                    p.confidence = (1.0 - reading.noise_level).max(0.1);
                    p.source_sensors.push(reading.sensor_id.clone());
                    p.id = self.next_percept_id;
                    self.next_percept_id += 1;
                    percepts.push(p);
                }
            }
            DataType::Binary => {
                let val = filtered_value.first().copied().unwrap_or(0.0);
                let mut p = Percept::new(if val > 0.5 { PerceptKind::Event } else { PerceptKind::Absence });
                p.labels.push(reading.sensor_id.clone());
                p.confidence = (1.0 - reading.noise_level).max(0.1);
                p.source_sensors.push(reading.sensor_id.clone());
                p.id = self.next_percept_id;
                self.next_percept_id += 1;
                percepts.push(p);
            }
            DataType::Image | DataType::Audio => {
                // Simplified: extract summary stats
                let mut p = Percept::new(PerceptKind::Property);
                p.labels.push(reading.sensor_id.clone());
                let mean = filtered_value.iter().sum::<f64>() / filtered_value.len().max(1) as f64;
                let variance = filtered_value.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / filtered_value.len().max(1) as f64;
                p.properties.insert("mean".into(), mean);
                p.properties.insert("std".into(), variance.sqrt());
                p.properties.insert("samples".into(), filtered_value.len() as f64);
                p.confidence = (1.0 - reading.noise_level).max(0.1);
                p.source_sensors.push(reading.sensor_id.clone());
                p.id = self.next_percept_id;
                self.next_percept_id += 1;
                percepts.push(p);
            }
        }

        // Update scene
        self.current_scene.percepts.extend(percepts.iter().cloned());
        self.current_scene.expire(reading.timestamp);
        self.current_scene.confidence = if !percepts.is_empty() {
            percepts.iter().map(|p| p.confidence).sum::<f64>() / percepts.len() as f64
        } else { self.current_scene.confidence };

        percepts
    }

    /// Track an object
    pub fn track(&mut self, obj: TrackedObject) {
        if self.objects.len() >= self.max_objects {
            // Remove oldest
            if let Some(oldest) = self.objects.iter().min_by_key(|(_, o)| o.last_seen).map(|(k, _)| k.clone()) {
                self.objects.remove(&oldest);
            }
        }
        self.objects.insert(obj.id.clone(), obj);
    }

    /// Update object position
    pub fn update_object(&mut self, id: &str, pos: (f64, f64), timestamp: u64) -> bool {
        if let Some(obj) = self.objects.get_mut(id) { obj.update(pos, timestamp); true } else { false }
    }

    /// Expire old objects
    pub fn expire_objects(&mut self, now: u64) {
        self.objects.retain(|_, obj| obj.confidence > 0.05);
        for obj in self.objects.values_mut() { obj.decay(now, self.object_ttl_ms); }
    }

    /// Get current scene snapshot
    pub fn scene(&self) -> &Scene { &self.current_scene }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moving_average_filter() {
        let mut f = SignalFilter::new(FilterMethod::MovingAverage, 3);
        let v = f.process(10.0); assert_eq!(v, 10.0);
        let v = f.process(20.0); assert!((v - 15.0).abs() < 0.01);
        let v = f.process(30.0); assert!((v - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_median_filter() {
        let mut f = SignalFilter::new(FilterMethod::Median, 3);
        f.process(100.0); f.process(1.0); // outlier
        let v = f.process(2.0);
        assert_eq!(v, 2.0); // median of [100, 1, 2] = 2
    }

    #[test]
    fn test_ema_filter() {
        let mut f = SignalFilter::new(FilterMethod::Exponential(EMAAlpha(0.3)), 5);
        let v = f.process(10.0); assert_eq!(v, 10.0);
        let v = f.process(20.0); assert!((v - 13.0).abs() < 0.01); // 0.3*20 + 0.7*10
    }

    #[test]
    fn test_process_scalar() {
        let mut pipe = PerceptionPipeline::new();
        pipe.add_filter("temp", FilterMethod::MovingAverage, 3);
        let reading = RawReading { sensor_id: "temp".into(), data_type: DataType::Scalar, values: vec![22.5], timestamp: 0, noise_level: 0.1 };
        let percepts = pipe.process(reading);
        assert_eq!(percepts.len(), 1);
        assert_eq!(percepts[0].kind, PerceptKind::Property);
    }

    #[test]
    fn test_process_vector() {
        let mut pipe = PerceptionPipeline::new();
        let reading = RawReading { sensor_id: "accel".into(), data_type: DataType::Vector, values: vec![1.0, 2.0], timestamp: 0, noise_level: 0.0 };
        let percepts = pipe.process(reading);
        assert_eq!(percepts.len(), 1);
        assert!((percepts[0].properties["magnitude"] - 2.236).abs() < 0.01);
    }

    #[test]
    fn test_process_binary() {
        let mut pipe = PerceptionPipeline::new();
        let reading = RawReading { sensor_id: "button".into(), data_type: DataType::Binary, values: vec![1.0], timestamp: 0, noise_level: 0.0 };
        let percepts = pipe.process(reading);
        assert_eq!(percepts[0].kind, PerceptKind::Event);
    }

    #[test]
    fn test_track_object() {
        let mut pipe = PerceptionPipeline::new();
        let obj = TrackedObject { id: "obj1".into(), kind: "wall".into(), position: (1.0, 2.0), velocity: (0.0, 0.0), confidence: 0.8, last_seen: 0, track_length: 0, properties: HashMap::new() };
        pipe.track(obj);
        assert!(pipe.update_object("obj1", (2.0, 3.0), 1000));
    }

    #[test]
    fn test_scene_expire() {
        let mut scene = Scene::new();
        let mut p = Percept::new(PerceptKind::Object);
        p.ttl_ms = 100; p.timestamp = 0;
        scene.percepts.push(p);
        scene.expire(200);
        assert!(scene.percepts.is_empty());
    }

    #[test]
    fn test_find_objects() {
        let mut scene = Scene::new();
        let obj = TrackedObject { id: "1".into(), kind: "wall_north".into(), position: (0.0, 0.0), velocity: (0.0, 0.0), confidence: 0.8, last_seen: 0, track_length: 0, properties: HashMap::new() };
        scene.objects.push(obj);
        let found = scene.find_objects("wall");
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_object_velocity() {
        let mut obj = TrackedObject { id: "1".into(), kind: "x".into(), position: (0.0, 0.0), velocity: (0.0, 0.0), confidence: 0.8, last_seen: 1000, track_length: 0, properties: HashMap::new() };
        obj.update((10.0, 0.0), 2000); // 1 second later, 10 units
        assert!((obj.velocity.0 - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_low_pass_filter() {
        let mut f = SignalFilter::new(FilterMethod::LowPass(0.1), 5);
        f.process(5.0);
        let v = f.process(100.0); // sudden spike
        assert!(v < 50.0); // low pass smooths it
    }

    #[test]
    fn test_filter_reset() {
        let mut f = SignalFilter::new(FilterMethod::MovingAverage, 5);
        f.process(10.0); f.process(20.0);
        f.reset();
        assert!(f.window.is_empty());
    }
}

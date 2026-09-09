use lru::LruCache;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

use crate::infra::Config;
use crate::services::intelligent_retriever::SearchResult;

#[allow(dead_code)]
#[derive(Default)]
pub struct CacheMetrics {
    pub embedding_hits: AtomicU64,
    pub embedding_misses: AtomicU64,
    pub vector_hits: AtomicU64,
    pub vector_misses: AtomicU64,
    pub algorithm_hits: AtomicU64,
    pub algorithm_misses: AtomicU64,
    pub result_hits: AtomicU64,
    pub result_misses: AtomicU64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VectorCacheKey {
    pub graph_name: String,
    pub vector_hash: u64,
    pub limit: usize,
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AlgorithmCacheKey {
    pub graph_name: String,
    pub algo_name: String,
    pub params_hash: u64,
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResultCacheKey {
    pub graph_name: String,
    pub query_hash: u64,
    pub limit: usize,
    pub options_hash: u64,
}

#[derive(Clone)]
pub struct MultiLevelCache {
    embedding_cache: Arc<Mutex<LruCache<String, (Vec<f64>, Instant)>>>,
    embedding_ttl: Duration,

    vector_cache: Arc<Mutex<LruCache<VectorCacheKey, (Vec<SearchResult>, Instant)>>>,
    vector_ttl: Duration,

    #[allow(dead_code)]
    algorithm_cache: Arc<Mutex<LruCache<AlgorithmCacheKey, (serde_json::Value, Instant)>>>,
    #[allow(dead_code)]
    algorithm_ttl: Duration,

    #[allow(dead_code)]
    result_cache: Arc<Mutex<LruCache<ResultCacheKey, (serde_json::Value, Instant)>>>,
    #[allow(dead_code)]
    result_ttl: Duration,

    pub metrics: Arc<CacheMetrics>,
}

impl MultiLevelCache {
    pub fn new(config: &Config) -> Self {
        let emb_cap = NonZeroUsize::new(config.cache_embedding_capacity).unwrap_or(NonZeroUsize::new(1000).unwrap());
        let vec_cap = NonZeroUsize::new(config.cache_vector_capacity).unwrap_or(NonZeroUsize::new(1000).unwrap());
        let algo_cap = NonZeroUsize::new(config.cache_algorithm_capacity).unwrap_or(NonZeroUsize::new(500).unwrap());
        let res_cap = NonZeroUsize::new(config.cache_result_capacity).unwrap_or(NonZeroUsize::new(500).unwrap());

        Self {
            embedding_cache: Arc::new(Mutex::new(LruCache::new(emb_cap))),
            embedding_ttl: Duration::from_secs(config.cache_embedding_ttl_secs),

            vector_cache: Arc::new(Mutex::new(LruCache::new(vec_cap))),
            vector_ttl: Duration::from_secs(config.cache_vector_ttl_secs),

            algorithm_cache: Arc::new(Mutex::new(LruCache::new(algo_cap))),
            algorithm_ttl: Duration::from_secs(config.cache_algorithm_ttl_secs),

            result_cache: Arc::new(Mutex::new(LruCache::new(res_cap))),
            result_ttl: Duration::from_secs(config.cache_result_ttl_secs),

            metrics: Arc::new(CacheMetrics::default()),
        }
    }

    pub fn hash_embedding(vec: &[f64]) -> u64 {
        let mut hasher = DefaultHasher::new();
        for &val in vec {
            hasher.write_u64(val.to_bits());
        }
        hasher.finish()
    }

    #[allow(dead_code)]
    pub fn hash_string(s: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    // --- Tier 1: Query Embeddings ---

    pub async fn get_embedding(&self, text: &str) -> Option<Vec<f64>> {
        let mut cache = self.embedding_cache.lock().await;
        if let Some((vec, timestamp)) = cache.get(text) {
            if timestamp.elapsed() < self.embedding_ttl {
                self.metrics.embedding_hits.fetch_add(1, Ordering::Relaxed);
                return Some(vec.clone());
            }
        }
        self.metrics.embedding_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    pub async fn put_embedding(&self, text: String, embedding: Vec<f64>) {
        if !embedding.is_empty() {
            let mut cache = self.embedding_cache.lock().await;
            cache.put(text, (embedding, Instant::now()));
        }
    }

    // --- Tier 2: Vector Search Results ---

    pub async fn get_vector_results(&self, key: &VectorCacheKey) -> Option<Vec<SearchResult>> {
        let mut cache = self.vector_cache.lock().await;
        if let Some((results, timestamp)) = cache.get(key) {
            if timestamp.elapsed() < self.vector_ttl {
                self.metrics.vector_hits.fetch_add(1, Ordering::Relaxed);
                return Some(results.clone());
            }
        }
        self.metrics.vector_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    pub async fn put_vector_results(&self, key: VectorCacheKey, results: Vec<SearchResult>) {
        let mut cache = self.vector_cache.lock().await;
        cache.put(key, (results, Instant::now()));
    }

    // --- Tier 3: Graph Algorithm Computations ---

    #[allow(dead_code)]
    pub async fn get_algorithm_result(&self, key: &AlgorithmCacheKey) -> Option<serde_json::Value> {
        let mut cache = self.algorithm_cache.lock().await;
        if let Some((val, timestamp)) = cache.get(key) {
            if timestamp.elapsed() < self.algorithm_ttl {
                self.metrics.algorithm_hits.fetch_add(1, Ordering::Relaxed);
                return Some(val.clone());
            }
        }
        self.metrics.algorithm_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    #[allow(dead_code)]
    pub async fn put_algorithm_result(&self, key: AlgorithmCacheKey, val: serde_json::Value) {
        let mut cache = self.algorithm_cache.lock().await;
        cache.put(key, (val, Instant::now()));
    }

    // --- Tier 4: Aggregated Responses ---

    #[allow(dead_code)]
    pub async fn get_result(&self, key: &ResultCacheKey) -> Option<serde_json::Value> {
        let mut cache = self.result_cache.lock().await;
        if let Some((val, timestamp)) = cache.get(key) {
            if timestamp.elapsed() < self.result_ttl {
                self.metrics.result_hits.fetch_add(1, Ordering::Relaxed);
                return Some(val.clone());
            }
        }
        self.metrics.result_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    #[allow(dead_code)]
    pub async fn put_result(&self, key: ResultCacheKey, val: serde_json::Value) {
        let mut cache = self.result_cache.lock().await;
        cache.put(key, (val, Instant::now()));
    }

    // --- Cache Invalidation ---

    pub async fn invalidate_graph(&self, graph_name: &str) {
        {
            let mut v_cache = self.vector_cache.lock().await;
            v_cache.clear();
        }
        {
            let mut a_cache = self.algorithm_cache.lock().await;
            a_cache.clear();
        }
        {
            let mut r_cache = self.result_cache.lock().await;
            r_cache.clear();
        }
        tracing::info!("Invalidated caches for graph: {}", graph_name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_config() -> Config {
        Config {
            app_port: 3002,
            host: "0.0.0.0".to_string(),
            falkordb_host: "localhost".to_string(),
            falkordb_port: 6379,
            falkordb_username: "default".to_string(),
            falkordb_password: None,
            falkordb_database: 0,
            falkordb_graph_name: "test_graph".to_string(),
            falkordb_vector_dimension: 768,
            falkordb_similarity_threshold: 0.7,
            falkordb_max_results: 10,
            falkordb_use_tls: false,
            nvidia_nim_api_key: None,
            nvidia_nim_base_url: "".to_string(),
            default_embedding_model: "test".to_string(),
            pipeline_max_query_chunks: 5,
            pipeline_per_chunk_timeout: 5.0,
            pipeline_vector_top_k: 10,
            pipeline_dfs_depth: 2,
            pipeline_dfs_min_relevance: 0.5,
            pipeline_dfs_max_results: 20,
            pipeline_max_total_results: 50,
            pipeline_vector_weight: 0.7,
            pipeline_graph_weight: 0.3,
            pipeline_cross_chunk_weight: 0.1,
            enable_bfs: true,
            enable_pagerank: true,
            enable_betweenness: false,
            enable_wcc: false,
            enable_sppaths: false,
            pagerank_boost_weight: 0.15,
            hnsw_m: 24,
            hnsw_ef_construction: 250,
            hnsw_ef_runtime: 25,
            hnsw_similarity_function: "COSINE".to_string(),
            hnsw_mode: "balanced".to_string(),
            cache_embedding_capacity: 100,
            cache_embedding_ttl_secs: 3600,
            cache_vector_capacity: 100,
            cache_vector_ttl_secs: 300,
            cache_algorithm_capacity: 50,
            cache_algorithm_ttl_secs: 1800,
            cache_result_capacity: 50,
            cache_result_ttl_secs: 120,
            fusion_algorithm: "wrrf".to_string(),
            wrrf_k: 60.0,
            log_level: "INFO".to_string(),
        }
    }

    #[tokio::test]
    async fn test_embedding_cache_hit_and_miss() {
        let config = dummy_config();
        let cache = MultiLevelCache::new(&config);

        assert!(cache.get_embedding("hello").await.is_none());
        assert_eq!(cache.metrics.embedding_misses.load(Ordering::Relaxed), 1);

        cache.put_embedding("hello".to_string(), vec![0.1, 0.2, 0.3]).await;
        let hit = cache.get_embedding("hello").await;
        assert!(hit.is_some());
        assert_eq!(hit.unwrap(), vec![0.1, 0.2, 0.3]);
        assert_eq!(cache.metrics.embedding_hits.load(Ordering::Relaxed), 1);
    }
}

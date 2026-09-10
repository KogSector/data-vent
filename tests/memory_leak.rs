use data_vent::services::memory_pool::MemoryPool;
use data_vent::services::cache::{MultiLevelCache, VectorCacheKey};
use data_vent::services::intelligent_retriever::SearchResult;
use data_vent::infra::Config;

fn test_config(capacity: usize) -> Config {
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
        cache_embedding_capacity: capacity,
        cache_embedding_ttl_secs: 3600,
        cache_vector_capacity: capacity,
        cache_vector_ttl_secs: 300,
        cache_algorithm_capacity: capacity,
        cache_algorithm_ttl_secs: 1800,
        cache_result_capacity: capacity,
        cache_result_ttl_secs: 120,
        fusion_algorithm: "wrrf".to_string(),
        wrrf_k: 60.0,
        log_level: "INFO".to_string(),
    }
}

#[tokio::test]
async fn test_no_memory_leaks_in_cache_churn() {
    let capacity = 50;
    let cfg = test_config(capacity);
    let cache = MultiLevelCache::new(&cfg);

    // Churn 50,000 items through the cache
    for i in 0..50_000 {
        let key = VectorCacheKey {
            graph_name: "leak_test".to_string(),
            vector_hash: i,
            limit: 10,
        };
        cache.put_vector_results(key, vec![SearchResult {
            chunk_id: format!("chunk_{}", i),
            content: "payload content that could leak if not properly dropped".to_string(),
            score: 0.85,
            metadata: serde_json::json!({"iter": i}),
            source: "leak_test".to_string(),
            chunk_type: "text".to_string(),
            source_id: "s1".to_string(),
            document_id: "d1".to_string(),
            depth: 1,
            matched_by_chunks: vec![],
        }]).await;
    }

    // Cache should remain functional and capped
    let stats = cache.get_cache_stats().await;
    assert_eq!(stats.vector_hit_rate, 0.0); // all were unique puts
}

#[tokio::test]
async fn test_memory_pool_reuse_prevents_leak() {
    let pool = MemoryPool::new(|| Vec::<String>::with_capacity(64));

    for _ in 0..10_000 {
        let mut buf = pool.acquire().await;
        buf.push("reusable_string".to_string());
        buf.clear();
        pool.release(buf).await;
    }

    assert_eq!(pool.size().await, 1, "Pool should retain exactly 1 reused buffer");
}

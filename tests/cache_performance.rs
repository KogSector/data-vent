use std::sync::Arc;
use tokio::task::JoinSet;

use data_vent::infra::Config;
use data_vent::services::cache::{MultiLevelCache, VectorCacheKey};
use data_vent::services::intelligent_retriever::SearchResult;

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
async fn test_cache_hit_rate() {
    let cfg = test_config(100);
    let cache = MultiLevelCache::new(&cfg);

    // Warm 10 distinct items
    for i in 0..10 {
        cache.put_embedding(format!("query_{}", i), vec![i as f64]).await;
    }

    // Access 100 times: 80 from warmed set, 20 new (misses)
    for i in 0..100 {
        let key = if i < 80 {
            format!("query_{}", i % 10)
        } else {
            format!("unknown_query_{}", i)
        };
        let _ = cache.get_embedding(&key).await;
    }

    let stats = cache.get_cache_stats().await;
    assert!(
        stats.embedding_hit_rate >= 0.60,
        "Expected >= 60% hit rate, got {:.2}%",
        stats.embedding_hit_rate * 100.0
    );
}

#[tokio::test]
async fn test_cache_memory_usage_and_capacity() {
    let capacity = 20;
    let cfg = test_config(capacity);
    let cache = MultiLevelCache::new(&cfg);

    // Insert more than capacity
    for i in 0..50 {
        let key = VectorCacheKey {
            graph_name: "test_graph".to_string(),
            vector_hash: i,
            limit: 10,
        };
        cache.put_vector_results(key, vec![SearchResult {
            chunk_id: format!("chunk_{}", i),
            content: "payload".to_string(),
            score: 0.9,
            metadata: serde_json::json!({}),
            source: "test".to_string(),
            chunk_type: "code".to_string(),
            source_id: "src".to_string(),
            document_id: "doc".to_string(),
            depth: 1,
            matched_by_chunks: vec![],
        }]).await;
    }

    // Earliest items should have been evicted by LRU
    let old_key = VectorCacheKey {
        graph_name: "test_graph".to_string(),
        vector_hash: 0,
        limit: 10,
    };
    assert!(cache.get_vector_results(&old_key).await.is_none());

    // Latest item must exist
    let recent_key = VectorCacheKey {
        graph_name: "test_graph".to_string(),
        vector_hash: 49,
        limit: 10,
    };
    assert!(cache.get_vector_results(&recent_key).await.is_some());
}

#[tokio::test]
async fn test_cache_concurrent_access() {
    let cfg = test_config(500);
    let cache = Arc::new(MultiLevelCache::new(&cfg));
    let mut set = JoinSet::new();

    for task_id in 0..10 {
        let cache_clone = cache.clone();
        set.spawn(async move {
            for i in 0..50 {
                let key = format!("task_{}_item_{}", task_id, i);
                cache_clone.put_embedding(key.clone(), vec![i as f64]).await;
                let hit = cache_clone.get_embedding(&key).await;
                assert!(hit.is_some());
            }
        });
    }

    while let Some(res) = set.join_next().await {
        assert!(res.is_ok());
    }

    let stats = cache.get_cache_stats().await;
    assert_eq!(stats.embedding_hit_rate, 1.0);
}

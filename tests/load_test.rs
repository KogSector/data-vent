use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use data_vent::infra::Config;
use data_vent::services::cache::MultiLevelCache;
use data_vent::services::fusion::FusionEngine;
use data_vent::services::parallel_search::{ChunkSearchResult, ParallelSearchResult};
use data_vent::services::query_decomposer::QueryDecomposer;
use data_vent::services::result_aggregator::ResultAggregator;

fn create_test_config() -> Config {
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
        cache_embedding_capacity: 1000,
        cache_embedding_ttl_secs: 3600,
        cache_vector_capacity: 1000,
        cache_vector_ttl_secs: 300,
        cache_algorithm_capacity: 500,
        cache_algorithm_ttl_secs: 1800,
        cache_result_capacity: 500,
        cache_result_ttl_secs: 120,
        fusion_algorithm: "wrrf".to_string(),
        wrrf_k: 60.0,
        log_level: "INFO".to_string(),
    }
}

#[tokio::test]
async fn test_single_call_load() {
    let decomposer = Arc::new(QueryDecomposer::new(5));
    let aggregator = Arc::new(ResultAggregator::new(
        20, 0.5, 3, 0.7, 0.3, 0.1, "wrrf".to_string(), 60.0, 0.15,
    ));

    let start = Instant::now();
    let total_requests = 1000;
    let mut successful = 0;

    for i in 0..total_requests {
        let query = format!("query parameter performance test iteration {}", i % 50);
        let decomp = decomposer.decompose_fast(&query).await;
        if !decomp.chunks.is_empty() {
            let mock_response = ParallelSearchResult {
                chunk_results: vec![ChunkSearchResult {
                    query_chunk: decomp.chunks[0].clone(),
                    vector_results: vec![],
                    graph_results: vec![],
                    search_time_ms: 0.1,
                    error: None,
                }],
                total_vector_hits: 0,
                total_graph_hits: 0,
                _total_time_ms: 0.1,
                chunks_searched: 1,
                _chunks_failed: 0,
            };

            let res = aggregator.aggregate_with_centrality(
                mock_response,
                &query,
                10,
                &HashMap::new(),
                &HashMap::new(),
            );
            if res.aggregation_time_ms >= 0.0 {
                successful += 1;
            }
        }
    }

    let elapsed = start.elapsed();
    let avg_rps = (total_requests as f64) / elapsed.as_secs_f64();
    println!("Single call load: {} in {:?} ({:.2} RPS)", successful, elapsed, avg_rps);

    assert_eq!(successful, total_requests, "All 1000 requests must succeed");
    assert!(avg_rps > 1000.0, "Expected >1000 RPS for single call pipeline");
}

#[tokio::test]
async fn test_multi_call_load() {
    let cfg = create_test_config();
    let cache = Arc::new(MultiLevelCache::new(&cfg));
    let fusion = Arc::new(FusionEngine::new(60.0, 0.15));

    let start = Instant::now();
    let sessions = 50;
    let calls_per_session = 10;
    let mut completed_sessions = 0;

    for s in 0..sessions {
        let mut session_success = true;
        for c in 0..calls_per_session {
            let key = format!("session_{}_call_{}", s, c);
            cache.put_embedding(key.clone(), vec![0.1 * (c as f64), 0.2, 0.3]).await;
            if cache.get_embedding(&key).await.is_none() {
                session_success = false;
            }
            let _ = fusion.wrrf_fuse(&[], &HashMap::new(), &HashMap::new(), 1, 10);
        }
        if session_success {
            completed_sessions += 1;
        }
    }

    let elapsed = start.elapsed();
    assert_eq!(completed_sessions, sessions);
    assert!(elapsed < Duration::from_secs(5), "Multi-call load exceeded time limit: {:?}", elapsed);
}

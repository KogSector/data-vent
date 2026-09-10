use std::collections::HashMap;
use std::time::Instant;

use data_vent::services::fusion::{FusionEngine, RankedList};
use data_vent::services::intelligent_retriever::SearchResult;
use data_vent::services::query_decomposer::QueryDecomposer;

fn mock_search_result(id: &str, score: f64) -> SearchResult {
    SearchResult {
        chunk_id: id.to_string(),
        content: format!("Content for {}", id),
        score,
        metadata: serde_json::json!({}),
        source: "regression_test".to_string(),
        chunk_type: "code".to_string(),
        source_id: "src_1".to_string(),
        document_id: "doc_1".to_string(),
        depth: 1,
        matched_by_chunks: vec!["test".to_string()],
    }
}

#[tokio::test]
async fn test_no_performance_regression_in_decomposition() {
    let decomposer = QueryDecomposer::new(5);
    let fast_query = "redis cluster rust service";

    let mut timings = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let res = decomposer.decompose_fast(fast_query).await;
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
        assert!(!res.chunks.is_empty());
    }

    let avg_ms: f64 = timings.iter().sum::<f64>() / (timings.len() as f64);
    println!("Average fast decomposition latency: {:.3}ms", avg_ms);
    assert!(avg_ms < 1.0, "Fast decomposition regressed: {:.2}ms > 1ms", avg_ms);

    let complex_query = "find database connection configuration for redis cluster in rust service";
    let start_complex = Instant::now();
    let complex_res = decomposer.decompose(complex_query).await;
    let complex_elapsed = start_complex.elapsed();
    assert!(!complex_res.chunks.is_empty());
    assert!(complex_elapsed.as_millis() < 60, "Complex decomposition too slow: {:?}", complex_elapsed);
}

#[test]
fn test_no_performance_regression_in_fusion() {
    let fusion = FusionEngine::new(60.0, 0.15);

    let list1 = RankedList {
        name: "vector".to_string(),
        weight: 1.0,
        items: (0..50).map(|i| mock_search_result(&format!("c_{}", i), 0.95 - (i as f64 * 0.01))).collect(),
    };

    let list2 = RankedList {
        name: "bfs_graph".to_string(),
        weight: 0.8,
        items: (25..75).map(|i| mock_search_result(&format!("c_{}", i), 0.85 - ((i - 25) as f64 * 0.01))).collect(),
    };

    let pr = HashMap::new();
    let bc = HashMap::new();

    let start = Instant::now();
    for _ in 0..500 {
        let res = fusion.wrrf_fuse(&[list1.clone(), list2.clone()], &pr, &bc, 2, 20);
        assert_eq!(res.len(), 20);
    }
    let elapsed = start.elapsed();
    let avg_us = (elapsed.as_micros() as f64) / 500.0;
    println!("Average fusion latency: {:.2}µs", avg_us);
    assert!(avg_us < 1000.0, "Fusion regressed: {:.2}µs > 1000µs (1ms)", avg_us);
}

use std::collections::HashMap;

use data_vent::services::fusion::{FusionEngine, RankedList};
use data_vent::services::intelligent_retriever::SearchResult;
use data_vent::services::query_decomposer::QueryDecomposer;
use data_vent::services::vector_search::HNSWConfig;

fn dummy_item(id: &str, score: f64) -> SearchResult {
    SearchResult {
        chunk_id: id.to_string(),
        content: format!("Content for {}", id),
        score,
        metadata: serde_json::json!({}),
        source: "test".to_string(),
        chunk_type: "code".to_string(),
        source_id: "src1".to_string(),
        document_id: "doc1".to_string(),
        depth: 1,
        matched_by_chunks: vec!["chunk1".to_string()],
    }
}

#[test]
fn test_hnsw_config_accuracy() {
    let balanced = HNSWConfig::from_mode("balanced");
    assert_eq!(balanced.m, 24);
    assert_eq!(balanced.ef_construction, 250);
    assert_eq!(balanced.ef_runtime, 25);

    let low_lat = HNSWConfig::from_mode("low_latency");
    assert_eq!(low_lat.m, 16);
    assert_eq!(low_lat.ef_construction, 200);
    assert_eq!(low_lat.ef_runtime, 10);

    let high_rec = HNSWConfig::from_mode("high_recall");
    assert_eq!(high_rec.m, 32);
    assert_eq!(high_rec.ef_construction, 300);
    assert_eq!(high_rec.ef_runtime, 50);
}

#[test]
fn test_wrrf_score_monotonicity() {
    let engine = FusionEngine::new(60.0, 0.15);

    let list = RankedList {
        name: "vector".to_string(),
        weight: 1.0,
        items: vec![
            dummy_item("item_1", 0.99),
            dummy_item("item_2", 0.85),
            dummy_item("item_3", 0.70),
        ],
    };

    let fused = engine.wrrf_fuse(&[list], &HashMap::new(), &HashMap::new(), 1, 10);
    assert_eq!(fused.len(), 3);
    assert!(fused[0].final_score >= fused[1].final_score);
    assert!(fused[1].final_score >= fused[2].final_score);
}

#[test]
fn test_pagerank_centrality_boosting_accuracy() {
    let engine = FusionEngine::new(60.0, 0.20);

    let list = RankedList {
        name: "vector".to_string(),
        weight: 1.0,
        items: vec![
            dummy_item("low_node", 0.80),
            dummy_item("high_node", 0.80),
        ],
    };

    let mut pr = HashMap::new();
    pr.insert("high_node".to_string(), 10.0);
    pr.insert("low_node".to_string(), 0.0);

    let fused = engine.wrrf_fuse(&[list], &pr, &HashMap::new(), 1, 10);
    assert_eq!(fused[0].chunk_id, "high_node");
    assert!(fused[0].final_score > fused[1].final_score);
}

#[tokio::test]
async fn test_query_decomposer_accuracy() {
    let decomposer = QueryDecomposer::new(5);
    let short_q = "authentication token";
    let fast_res = decomposer.decompose_fast(short_q).await;
    assert_eq!(fast_res.chunks.len(), 1);
    assert_eq!(fast_res.chunks[0].text, short_q);
}

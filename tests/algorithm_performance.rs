use std::time::Instant;

use data_vent::services::graph_algorithms::GraphAlgorithms;
use data_vent::services::vector_search::FalkorDBClient;

#[tokio::test]
async fn test_bfs_performance_dummy_mode() {
    let client = FalkorDBClient::new_dummy();
    let algo = GraphAlgorithms::new(client);

    let start = Instant::now();
    let candidates = vec!["chunk_1".to_string(), "chunk_2".to_string()];
    let res = algo.bfs_traversal("test_graph", &candidates, 3, 20).await;
    let elapsed = start.elapsed();

    assert!(res.is_empty());
    assert!(elapsed.as_millis() < 50, "BFS traversal took too long: {:?}", elapsed);
}

#[tokio::test]
async fn test_pagerank_performance_dummy_mode() {
    let client = FalkorDBClient::new_dummy();
    let algo = GraphAlgorithms::new(client);

    let start = Instant::now();
    let candidates = vec!["chunk_1".to_string(), "chunk_2".to_string()];
    let res = algo.get_pagerank_scores("test_graph", &candidates).await;
    let elapsed = start.elapsed();

    assert!(res.is_empty());
    assert!(elapsed.as_millis() < 50, "PageRank took too long: {:?}", elapsed);
}

#[tokio::test]
async fn test_sppaths_performance_dummy_mode() {
    let client = FalkorDBClient::new_dummy();
    let algo = GraphAlgorithms::new(client);

    let start = Instant::now();
    let res = algo.shortest_paths("test_graph", "chunk_1", "chunk_2", 3, 3, &[]).await;
    let elapsed = start.elapsed();

    assert!(res.is_empty());
    assert!(elapsed.as_millis() < 50, "SPpaths took too long: {:?}", elapsed);
}

#[tokio::test]
async fn test_parallel_algorithms_performance() {
    let client = FalkorDBClient::new_dummy();
    let algo = GraphAlgorithms::new(client);

    let start = Instant::now();
    let candidates = vec!["chunk_1".to_string(), "chunk_2".to_string()];
    let res = algo.run_parallel_algorithms("test_graph", &candidates).await;
    let elapsed = start.elapsed();

    assert!(res.total_duration_ms >= 0.0);
    assert!(elapsed.as_millis() < 100, "Parallel algorithms took too long: {:?}", elapsed);
}

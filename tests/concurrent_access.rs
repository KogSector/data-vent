use std::collections::HashMap;
use std::sync::Arc;
use tokio::task::JoinSet;

use data_vent::services::fusion::{FusionEngine, RankedList};
use data_vent::services::intelligent_retriever::SearchResult;
use data_vent::services::memory_pool::MemoryPool;

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
        matched_by_chunks: vec!["query1".to_string()],
    }
}

#[tokio::test]
async fn test_concurrent_retrieval_fusion() {
    let engine = Arc::new(FusionEngine::new(60.0, 0.15));
    let mut tasks = JoinSet::new();

    for task_id in 0..100 {
        let engine_clone = engine.clone();
        tasks.spawn(async move {
            let list1 = RankedList {
                name: "vector".to_string(),
                weight: 1.0,
                items: vec![
                    dummy_item(&format!("t{}_a", task_id), 0.9),
                    dummy_item(&format!("t{}_b", task_id), 0.8),
                ],
            };
            let list2 = RankedList {
                name: "graph".to_string(),
                weight: 0.5,
                items: vec![
                    dummy_item(&format!("t{}_b", task_id), 0.85),
                    dummy_item(&format!("t{}_c", task_id), 0.7),
                ],
            };

            let mut pr = HashMap::new();
            pr.insert(format!("t{}_b", task_id), 5.0);

            let res = engine_clone.wrrf_fuse(&[list1, list2], &pr, &HashMap::new(), 2, 5);
            assert!(!res.is_empty());
            assert_eq!(res[0].chunk_id, format!("t{}_b", task_id));
        });
    }

    let mut completed = 0;
    while let Some(res) = tasks.join_next().await {
        assert!(res.is_ok());
        completed += 1;
    }
    assert_eq!(completed, 100);
}

#[tokio::test]
async fn test_concurrent_memory_pool_access() {
    let pool = Arc::new(MemoryPool::new(|| Vec::<i32>::with_capacity(32)));
    let mut tasks = JoinSet::new();

    for _ in 0..100 {
        let p = pool.clone();
        tasks.spawn(async move {
            let mut v = p.acquire().await;
            v.push(1);
            v.push(2);
            v.clear();
            p.release(v).await;
        });
    }

    let mut completed = 0;
    while let Some(res) = tasks.join_next().await {
        assert!(res.is_ok());
        completed += 1;
    }
    assert_eq!(completed, 100);
}

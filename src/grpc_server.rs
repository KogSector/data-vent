use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status};
use tracing::info;

use crate::services::intelligent_retriever::IntelligentRetriever;
use crate::services::parallel_search::ParallelSearchDispatcher;
use crate::services::query_decomposer::QueryDecomposer;
use crate::services::result_aggregator::ResultAggregator;

pub mod pb {
    tonic::include_proto!("confuse.retrieval.v1");
}

use pb::retrieval_service_server::{RetrievalService, RetrievalServiceServer};
use pb::{
    HybridSearchRequest, HybridSearchResponse, QueryChunkInfo, RetrievalDfsRequest,
    RetrievalDfsResponse, RetrievalHealthRequest, RetrievalHealthResponse,
    RetrievalSearchRequest, RetrievalSearchResponse, RetrieveRequest, RetrieveResponse,
    RetrievedChunk, ScoredResult,
};

pub struct MyRetrievalService {
    retriever: Arc<IntelligentRetriever>,
    decomposer: Arc<QueryDecomposer>,
    dispatcher: Arc<ParallelSearchDispatcher>,
    aggregator: Arc<ResultAggregator>,
}

impl MyRetrievalService {
    pub fn new(
        retriever: Arc<IntelligentRetriever>,
        decomposer: Arc<QueryDecomposer>,
        dispatcher: Arc<ParallelSearchDispatcher>,
        aggregator: Arc<ResultAggregator>,
    ) -> Self {
        Self {
            retriever,
            decomposer,
            dispatcher,
            aggregator,
        }
    }
}

#[tonic::async_trait]
impl RetrievalService for MyRetrievalService {
    async fn retrieve(
        &self,
        request: Request<RetrieveRequest>,
    ) -> Result<Response<RetrieveResponse>, Status> {
        let req = request.into_inner();
        let start = std::time::Instant::now();

        let decomp_res = self.decomposer.decompose(&req.query).await;
        
        let mut all_chunks = decomp_res.chunks;
        
        let search_res = self.dispatcher.dispatch(all_chunks.clone(), &self.retriever).await;
        let agg_res = self.aggregator.aggregate(search_res, &req.query, req.limit as usize);

        let mut results = vec![];
        for c in agg_res.chunks {
            let mut metadata = std::collections::HashMap::new();
            if let Some(obj) = c.metadata.as_object() {
                for (k, v) in obj {
                    metadata.insert(k.clone(), v.to_string());
                }
            }
            results.push(ScoredResult {
                chunk_id: c.chunk_id,
                content: c.content,
                final_score: c.final_score as f32,
                vector_score: c.vector_score as f32,
                graph_score: c.graph_score as f32,
                cross_chunk_boost: c.cross_chunk_boost as f32,
                chunk_type: c.chunk_type,
                source_id: c.source_id,
                document_id: c.document_id,
                metadata,
                matched_by_chunks: c.matched_by_chunks,
            });
        }

        let mut query_chunks = vec![];
        for c in all_chunks {
            query_chunks.push(QueryChunkInfo {
                text: c.text,
                intent: c.intent,
                weight: c.weight as f32,
            });
        }

        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(Response::new(RetrieveResponse {
            results,
            total_results: agg_res.total_results as i32,
            unique_sources: agg_res.unique_sources as i32,
            vector_matches: agg_res.vector_matches as i32,
            graph_matches: agg_res.graph_matches as i32,
            completion_reached: agg_res.completion_reached,
            query_chunks,
            decomposition_time_ms: decomp_res.decomposition_time_ms as f32,
            search_time_ms: 0.0,
            aggregation_time_ms: agg_res.aggregation_time_ms as f32,
            total_time_ms: total_time_ms as f32,
        }))
    }

    async fn search(
        &self,
        request: Request<RetrievalSearchRequest>,
    ) -> Result<Response<RetrievalSearchResponse>, Status> {
        let req = request.into_inner();
        let start = std::time::Instant::now();
        let vectors: Vec<f64> = req.query_vectors.iter().map(|f| *f as f64).collect();
        let results = self.retriever.vector_search(&vectors, req.limit as usize).await;
        
        let mut chunks = vec![];
        for c in results {
            let mut metadata = std::collections::HashMap::new();
            if let Some(obj) = c.metadata.as_object() {
                for (k, v) in obj {
                    metadata.insert(k.clone(), v.to_string());
                }
            }
            chunks.push(RetrievedChunk {
                chunk_id: c.chunk_id,
                content: c.content,
                score: c.score as f32,
                chunk_type: c.chunk_type,
                source_id: c.source_id,
                document_id: c.document_id,
                metadata,
            });
        }
        
        Ok(Response::new(RetrievalSearchResponse {
            total: chunks.len() as i32,
            chunks,
            search_time_ms: start.elapsed().as_secs_f32() * 1000.0,
        }))
    }

    async fn dfs_traverse(
        &self,
        request: Request<RetrievalDfsRequest>,
    ) -> Result<Response<RetrievalDfsResponse>, Status> {
        let req = request.into_inner();
        let start = std::time::Instant::now();
        
        let results = self.retriever.dfs_traversal(
            &req.start_chunk_ids,
            req.max_depth as usize,
            req.min_relevance as f64,
            req.max_results as usize
        ).await;

        let mut chunks = vec![];
        for c in results {
            let mut metadata = std::collections::HashMap::new();
            if let Some(obj) = c.metadata.as_object() {
                for (k, v) in obj {
                    metadata.insert(k.clone(), v.to_string());
                }
            }
            chunks.push(RetrievedChunk {
                chunk_id: c.chunk_id,
                content: c.content,
                score: c.score as f32,
                chunk_type: c.chunk_type,
                source_id: c.source_id,
                document_id: c.document_id,
                metadata,
            });
        }

        Ok(Response::new(RetrievalDfsResponse {
            nodes_visited: chunks.len() as i32,
            chunks,
            completion_reached: false,
            traversal_time_ms: start.elapsed().as_secs_f32() * 1000.0,
        }))
    }

    async fn hybrid_search(
        &self,
        request: Request<HybridSearchRequest>,
    ) -> Result<Response<HybridSearchResponse>, Status> {
        let req = request.into_inner();
        let start = std::time::Instant::now();

        let results = self.retriever.retrieve(
            &req.query_text,
            Some(req.source_ids),
            req.limit as usize,
            None
        ).await;

        let mut chunks = vec![];
        for c in results {
            let mut metadata = std::collections::HashMap::new();
            if let Some(obj) = c.metadata.as_object() {
                for (k, v) in obj {
                    metadata.insert(k.clone(), v.to_string());
                }
            }
            chunks.push(RetrievedChunk {
                chunk_id: c.chunk_id,
                content: c.content,
                score: c.score as f32,
                chunk_type: c.chunk_type,
                source_id: c.source_id,
                document_id: c.document_id,
                metadata,
            });
        }

        Ok(Response::new(HybridSearchResponse {
            vector_matches: chunks.len() as i32,
            graph_matches: 0,
            completion_reached: false,
            total_time_ms: start.elapsed().as_secs_f32() * 1000.0,
            chunks,
        }))
    }

    async fn health_check(
        &self,
        _request: Request<RetrievalHealthRequest>,
    ) -> Result<Response<RetrievalHealthResponse>, Status> {
        Ok(Response::new(RetrievalHealthResponse {
            status: "ok".to_string(),
            version: "0.2.0".to_string(),
            falkordb_connected: true,
            embeddings_service_connected: true,
        }))
    }
}

pub async fn start_grpc_server(
    addr: std::net::SocketAddr,
    retriever: Arc<IntelligentRetriever>,
    decomposer: Arc<QueryDecomposer>,
    dispatcher: Arc<ParallelSearchDispatcher>,
    aggregator: Arc<ResultAggregator>,
) -> anyhow::Result<()> {
    info!("Starting gRPC server on {}", addr);
    
    let service = MyRetrievalService::new(retriever, decomposer, dispatcher, aggregator);

    Server::builder()
        .add_service(RetrievalServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}

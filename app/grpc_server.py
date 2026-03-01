"""
Data Vent - gRPC Server
Serves retrieval RPCs for the distributed pipeline.
Includes the full Retrieve RPC for the decompose → search → aggregate pipeline.
"""
import asyncio
import time
import grpc
from concurrent import futures
import structlog

from app.config import settings
from app.services.intelligent_retriever import IntelligentRetriever, ChunkNode
from app.services.query_decomposer import QueryDecomposer
from app.services.parallel_search import ParallelSearchDispatcher
from app.services.result_aggregator import ResultAggregator

logger = structlog.get_logger()


# Proto stubs will be generated from proto/retrieval.proto
# Run: python -m grpc_tools.protoc -I proto --python_out=app/proto --grpc_python_out=app/proto proto/retrieval.proto

try:
    from app.proto import retrieval_pb2
    from app.proto import retrieval_pb2_grpc
    GRPC_STUBS_AVAILABLE = True
except ImportError:
    GRPC_STUBS_AVAILABLE = False
    logger.warning("gRPC stubs not generated yet. Run proto generation first.")


class RetrievalServicer:
    """gRPC servicer for data-vent retrieval operations."""
    
    def __init__(
        self,
        retriever: IntelligentRetriever,
        decomposer: QueryDecomposer = None,
        dispatcher: ParallelSearchDispatcher = None,
        aggregator: ResultAggregator = None,
    ):
        self.retriever = retriever
        self.decomposer = decomposer
        self.dispatcher = dispatcher
        self.aggregator = aggregator
    
    async def Retrieve(self, request, context):
        """Handle full retrieval pipeline: decompose → parallel search → aggregate."""
        pipeline_start = time.perf_counter()
        
        if not self.decomposer or not self.dispatcher or not self.aggregator:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Pipeline not initialized")
            return retrieval_pb2.RetrieveResponse()
        
        # Step 1: Decompose
        decomposition = await self.decomposer.decompose(request.query)
        
        # Step 2: Parallel search
        parallel_result = await self.dispatcher.dispatch(
            chunks=decomposition.chunks,
            retriever=self.retriever,
        )
        
        # Step 3: Aggregate
        aggregated = await self.aggregator.aggregate(
            parallel_result=parallel_result,
            original_query=request.query,
            limit=request.limit or 20,
        )
        
        total_time = (time.perf_counter() - pipeline_start) * 1000
        
        # Build response
        results = []
        for c in aggregated.chunks:
            results.append(retrieval_pb2.ScoredResult(
                chunk_id=c.chunk_id,
                content=c.content,
                final_score=c.final_score,
                vector_score=c.vector_score,
                graph_score=c.graph_score,
                cross_chunk_boost=c.cross_chunk_boost,
                chunk_type=c.chunk_type,
                source_id=c.source_id,
                document_id=c.document_id,
                metadata=c.metadata,
                matched_by_chunks=c.matched_by_chunks,
            ))
        
        query_chunks = [
            retrieval_pb2.QueryChunkInfo(
                text=qc.text,
                intent=qc.intent,
                weight=qc.weight,
            )
            for qc in decomposition.chunks
        ]
        
        return retrieval_pb2.RetrieveResponse(
            results=results,
            total_results=aggregated.total_results,
            unique_sources=aggregated.unique_sources,
            vector_matches=aggregated.vector_matches,
            graph_matches=aggregated.graph_matches,
            completion_reached=aggregated.completion_reached,
            query_chunks=query_chunks,
            decomposition_time_ms=decomposition.decomposition_time_ms,
            search_time_ms=parallel_result.total_time_ms,
            aggregation_time_ms=aggregated.aggregation_time_ms,
            total_time_ms=total_time,
        )
    
    async def Search(self, request, context):
        """Handle vector similarity search."""
        results = await self.retriever.vector_search(
            query_vectors=list(request.query_vectors),
            limit=request.limit or 10,
            similarity_threshold=request.similarity_threshold or settings.FALCORDB_SIMILARITY_THRESHOLD,
            source_ids=list(request.source_ids) if request.source_ids else None,
        )
        
        chunks = []
        for r in results:
            chunk = retrieval_pb2.RetrievedChunk(
                chunk_id=r.chunk_id,
                content=r.content,
                score=r.score,
                chunk_type=r.chunk_type,
                source_id=r.source_id,
                document_id=r.document_id,
                metadata=r.metadata,
            )
            chunks.append(chunk)
        
        return retrieval_pb2.RetrievalSearchResponse(
            chunks=chunks,
            total=len(chunks),
            search_time_ms=0.0,
        )
    
    async def DFSTraverse(self, request, context):
        """Handle DFS traversal."""
        result = await self.retriever.dfs_traversal(
            start_chunk_ids=list(request.start_chunk_ids),
            max_depth=request.max_depth or 3,
            min_relevance=request.min_relevance or 0.3,
            max_results=request.max_results or 50,
        )
        
        chunks = []
        for r in result.chunks:
            chunk = retrieval_pb2.RetrievedChunk(
                chunk_id=r.chunk_id,
                content=r.content,
                score=r.score,
                chunk_type=r.chunk_type,
                source_id=r.source_id,
                document_id=r.document_id,
                metadata=r.metadata,
            )
            chunks.append(chunk)
        
        return retrieval_pb2.RetrievalDFSResponse(
            chunks=chunks,
            nodes_visited=result.nodes_visited,
            completion_reached=result.completion_reached,
            traversal_time_ms=0.0,
        )
    
    async def HybridSearch(self, request, context):
        """Handle hybrid search (vector + graph)."""
        result = await self.retriever.hybrid_search(
            query_text=request.query_text,
            query_vectors=list(request.query_vectors),
            limit=request.limit or 20,
            similarity_threshold=request.similarity_threshold or settings.FALCORDB_SIMILARITY_THRESHOLD,
            dfs_depth=request.dfs_depth or 2,
            dfs_min_relevance=request.dfs_min_relevance or 0.3,
            source_ids=list(request.source_ids) if request.source_ids else None,
        )
        
        chunks = []
        for r in result["chunks"]:
            chunk = retrieval_pb2.RetrievedChunk(
                chunk_id=r.chunk_id,
                content=r.content,
                score=r.score,
                chunk_type=r.chunk_type,
                source_id=r.source_id,
                document_id=r.document_id,
                metadata=r.metadata,
            )
            chunks.append(chunk)
        
        return retrieval_pb2.HybridSearchResponse(
            chunks=chunks,
            vector_matches=result["vector_matches"],
            graph_matches=result["graph_matches"],
            completion_reached=result["completion_reached"],
            total_time_ms=result["total_time_ms"],
        )
    
    async def HealthCheck(self, request, context):
        """Handle health check."""
        return retrieval_pb2.RetrievalHealthResponse(
            status="healthy",
            version="0.2.0",
            falkordb_connected=self.retriever._driver is not None,
            embeddings_service_connected=self.retriever._http_client is not None,
        )


async def start_grpc_server(
    retriever: IntelligentRetriever,
    decomposer: QueryDecomposer = None,
    dispatcher: ParallelSearchDispatcher = None,
    aggregator: ResultAggregator = None,
):
    """Start the gRPC server for retrieval operations."""
    if not GRPC_STUBS_AVAILABLE:
        logger.warning("gRPC stubs not available, skipping gRPC server")
        return
    
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    
    servicer = RetrievalServicer(retriever, decomposer, dispatcher, aggregator)
    retrieval_pb2_grpc.add_RetrievalServiceServicer_to_server(servicer, server)
    
    listen_addr = f"{settings.GRPC_HOST}:{settings.GRPC_PORT}"
    server.add_insecure_port(listen_addr)
    
    logger.info("data_vent_grpc_starting", address=listen_addr)
    await server.start()
    logger.info("data_vent_grpc_started", address=listen_addr)
    
    await server.wait_for_termination()

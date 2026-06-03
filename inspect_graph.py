import asyncio
import sys
from app.config import settings
from app.services.vector_search import build_client_from_settings

async def main():
    client = build_client_from_settings(settings)
    await client.connect()
    
    # 1. Total nodes
    res = await client.query("MATCH (n) RETURN count(n) AS c")
    print(f"Total nodes: {res}")
    
    # 2. Sample Vector_Chunk
    res = await client.query("MATCH (n:Vector_Chunk) RETURN n.id, n.content, n.chunk_type LIMIT 1")
    data = res[1]
    if data and data[0]:
        print(f"Sample node: id={data[0][0]}, type={data[0][2]}, content={str(data[0][1])[:100]}")
        
        # Check embedding length
        res_emb = await client.query("MATCH (n:Vector_Chunk) WHERE n.id = '" + str(data[0][0]) + "' RETURN size(n.embedding)")
        emb_data = res_emb[1]
        if emb_data and emb_data[0]:
            print(f"Embedding length: {emb_data[0][0]}")
        else:
            print("Embedding length query failed.")
    else:
        print("No Vector_Chunk nodes found.")
        
    await client.close()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

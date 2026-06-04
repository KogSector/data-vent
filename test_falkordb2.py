import asyncio
from app.services.vector_search import FalkorDBClient
async def main():
    class Settings:
        FALKORDB_HOST = "r-6jissuruar.instance-tju0dagr0.hc-7up0crkyn.ap-south-1.aws.f2e0a955bb84.cloud"
        FALKORDB_PORT = 64172
        FALKORDB_USERNAME = "default"
        FALKORDB_PASSWORD = "password"
        FALKORDB_GRAPH_NAME = "knowledge-layer"
    client = FalkorDBClient(Settings())
    await client.connect()
    try:
        res = await client.query("MATCH (n:Vector_Chunk) WHERE toLower(n.content) CONTAINS 'toefl' RETURN count(n)")
        print(res)
        res2 = await client.query("MATCH (n:Vector_Chunk) WHERE toLower(n.content) CONTAINS 'practice' RETURN n.id, n.content LIMIT 1")
        print(res2)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()
asyncio.run(main())

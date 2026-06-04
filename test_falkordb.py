import asyncio
from app.config import settings
from app.services.vector_search import build_client_from_settings

async def main():
    client = build_client_from_settings(settings)
    await client.connect()
    
    # Query for distinct source_ids
    res = await client.query("MATCH (n:Vector_Chunk) WHERE NOT n.source_id STARTS WITH 'web:' RETURN substring(n.content, 0, 200) LIMIT 5")
    print("Non-web chunk contents:")
    for row in res[1]:
        print(row)
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())

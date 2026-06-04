import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        # Test the retrieve endpoint
        try:
            res = await client.post("http://localhost:3005/api/v1/retrieve", json={"query": "TOEFL Practice test"})
            print(f"Status Code: {res.status_code}")
            print(res.json())
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

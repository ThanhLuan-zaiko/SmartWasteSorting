from __future__ import annotations

from typing import Any


class LocalAgentApiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def health(self) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0) as client:
            response = await client.get("/health")
            response.raise_for_status()
            return response.json()

    async def classify(self, sample_ids: list[str]) -> list[dict[str, Any]]:
        import httpx

        async with httpx.AsyncClient(base_url=self.base_url, timeout=30.0) as client:
            response = await client.post("/classify", json={"sample_ids": sample_ids})
            response.raise_for_status()
            return response.json()

import inspect
import json
from datetime import datetime
from typing import Any, Awaitable, Dict, List, Optional, Protocol

from aibrix.batch.job_entity import BatchJob, BatchJobSpec
from aibrix.batch.state import JobEntityManager
from aibrix.storage.byteredis import AsyncRedis, _AsyncBytedRedisClient


class RedisJobCacheClient(Protocol):
    def get(self, key: str) -> bytes | str | None | Awaitable[bytes | str | None]: ...

    def zrevrange(
        self, key: str, start: int, end: int
    ) -> list[bytes | str] | Awaitable[list[bytes | str]]: ...

    def delete(self, key: str) -> Any | Awaitable[Any]: ...

    def zrem(self, key: str, value: str) -> Any | Awaitable[Any]: ...

    def set(self, key: str, value: str) -> Any | Awaitable[Any]: ...

    def zadd(self, key: str, mapping: dict[str, float]) -> Any | Awaitable[Any]: ...


class RedisJobCache(JobEntityManager):
    def __init__(
        self,
        redis_psm: Optional[str] = None,
        db: int = 0,
        key_prefix: str = "batch_jobs",
        redis_client: Optional[RedisJobCacheClient] = None,
    ) -> None:
        super().__init__()
        self.active_jobs: Dict[str, BatchJob] = {}
        if redis_client is not None:
            self._client: RedisJobCacheClient = redis_client
        else:
            if redis_psm is None:
                raise ValueError(
                    "redis_psm is required when redis_client is not provided"
                )
            self._client = self._build_client(
                redis_psm=redis_psm,
                db=db,
            )
        self._key_prefix = key_prefix
        self._index_key = f"{key_prefix}:index"

    async def get_job(self, job_id: str) -> Optional[BatchJob]:
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        payload = await self._maybe_await(self._client.get(self._job_key(job_id)))
        if payload is None:
            return None
        return self._deserialize_job(payload)

    async def list_jobs(self) -> List[BatchJob]:
        job_ids = await self._maybe_await(
            self._client.zrevrange(self._index_key, 0, -1)
        )
        jobs: List[BatchJob] = []
        for raw_job_id in job_ids:
            job_id = (
                raw_job_id.decode("utf-8")
                if isinstance(raw_job_id, bytes)
                else raw_job_id
            )
            payload = await self._maybe_await(self._client.get(self._job_key(job_id)))
            if payload is None:
                continue
            jobs.append(self._deserialize_job(payload))
        self.active_jobs = {job.job_id: job for job in jobs if job.job_id is not None}
        return jobs

    async def submit_job(
        self, session_id: str, job_spec: BatchJobSpec, request_count: int = 0
    ):
        job = BatchJob.new_local(spec=job_spec, request_count=request_count)
        job.session_id = session_id
        stored_job = await self._upsert_job(job, None)
        await self.job_committed(stored_job)

    async def update_job_ready(self, job: BatchJob):
        await self._update_existing_job(job)

    async def update_job_status(self, job: BatchJob):
        await self._update_existing_job(job)

    async def cancel_job(self, job: BatchJob):
        await self._update_existing_job(job)

    async def delete_job(self, job: BatchJob):
        if job.job_id is None:
            raise ValueError("job_id is required")
        existing_job = await self.get_job(job.job_id) or job
        await self._maybe_await(self._client.delete(self._job_key(job.job_id)))
        await self._maybe_await(self._client.zrem(self._index_key, job.job_id))
        self.active_jobs.pop(job.job_id, None)
        await self.job_deleted(existing_job)

    def _build_client(
        self,
        redis_psm: str,
        db: int,
    ) -> AsyncRedis:
        return _AsyncBytedRedisClient(redis_psm, db)

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _update_existing_job(self, job: BatchJob) -> None:
        if job.job_id is None:
            raise ValueError("job_id is required")
        old_job = await self.get_job(job.job_id)
        stored_job = await self._upsert_job(job, old_job)
        if old_job is not None:
            await self.job_updated(old_job, stored_job)

    async def _upsert_job(self, job: BatchJob, old_job: Optional[BatchJob]) -> BatchJob:
        if job.job_id is None:
            raise ValueError("job_id is required")
        stored_job = job.model_copy(deep=True)
        stored_job_id = stored_job.job_id
        if stored_job_id is None:
            raise ValueError("job_id is required")
        stored_job.metadata.resource_version = self._next_resource_version(old_job)
        payload = stored_job.model_dump_json(by_alias=True)
        await self._maybe_await(self._client.set(self._job_key(stored_job_id), payload))
        await self._maybe_await(
            self._client.zadd(
                self._index_key,
                {stored_job_id: self._created_at_score(stored_job)},
            )
        )
        self.active_jobs[stored_job_id] = stored_job
        return stored_job

    def _job_key(self, job_id: str) -> str:
        return f"{self._key_prefix}:{job_id}"

    def _deserialize_job(self, payload: Any) -> BatchJob:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        job = BatchJob.model_validate(json.loads(payload))
        if job.job_id is not None:
            self.active_jobs[job.job_id] = job
        return job

    def _next_resource_version(self, old_job: Optional[BatchJob]) -> str:
        if old_job is None or old_job.metadata.resource_version is None:
            return "1"
        try:
            return str(int(old_job.metadata.resource_version) + 1)
        except ValueError:
            return "1"

    def _created_at_score(self, job: BatchJob) -> float:
        created_at = job.status.created_at
        if created_at is not None:
            return created_at.timestamp()
        payload = job.model_dump(mode="json", by_alias=True)
        return datetime.fromisoformat(payload["status"]["createdAt"]).timestamp()

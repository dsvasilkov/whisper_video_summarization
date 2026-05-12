import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import aioboto3
from botocore.exceptions import ClientError


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str


def s3_bucket() -> str:
    return os.getenv("S3_BUCKET", "whisper-audio").strip() or "whisper-audio"


def parse_s3_uri(uri: str) -> S3Location:
    raw = (uri or "").strip()
    if not raw.startswith("s3://"):
        raise ValueError(f"Not an s3 uri: {uri!r}")
    rest = raw[len("s3://") :]
    if "/" not in rest:
        raise ValueError(f"Invalid s3 uri: {uri!r}")
    bucket, key = rest.split("/", 1)
    bucket = bucket.strip()
    key = key.strip()
    if not bucket or not key:
        raise ValueError(f"Invalid s3 uri: {uri!r}")
    return S3Location(bucket=bucket, key=key)


def build_s3_uri(bucket: str, key: str) -> str:
    bucket = (bucket or "").strip()
    key = (key or "").lstrip("/")
    return f"s3://{bucket}/{key}"


def _endpoint_url() -> str | None:
    raw = os.getenv("S3_ENDPOINT_URL", "").strip()
    return raw or None


def _presign_endpoint_url() -> str | None:
    raw = os.getenv("S3_PRESIGN_ENDPOINT_URL", "").strip()
    return raw or None


def _session() -> aioboto3.Session:
    # Works both for AWS S3 and MinIO (S3-compatible).
    return aioboto3.Session(
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID", "").strip() or None,
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY", "").strip() or None,
        region_name=os.getenv("S3_REGION", "").strip() or None,
    )


def _client_kwargs(*, endpoint_url: str | None) -> dict[str, object]:
    out: dict[str, object] = {}
    if endpoint_url:
        out["endpoint_url"] = endpoint_url
    return out


async def ensure_bucket_exists(bucket: str) -> None:
    async with _session().client("s3", **_client_kwargs(endpoint_url=_endpoint_url())) as client:
        try:
            await client.head_bucket(Bucket=bucket)
            return
        except ClientError:
            pass
        await client.create_bucket(Bucket=bucket)


def _minio_webhook_queue_arn() -> str | None:
    # MinIO SQS target ARN when configured via MINIO_NOTIFY_WEBHOOK_* env.
    raw = os.getenv("MINIO_WEBHOOK_QUEUE_ARN", "").strip()
    return raw or "arn:minio:sqs::api:webhook"


async def ensure_bucket_event_notifications(bucket: str) -> None:
    """Ensure bucket notifies MinIO webhook on object create.

    Requires the MinIO server to have a webhook target configured, e.g. via:
    MINIO_NOTIFY_WEBHOOK_ENABLE_api=on and MINIO_NOTIFY_WEBHOOK_ENDPOINT_api=...
    """
    queue_arn = _minio_webhook_queue_arn()
    if not queue_arn:
        return
    async with _session().client("s3", **_client_kwargs(endpoint_url=_endpoint_url())) as client:
        try:
            cfg = await client.get_bucket_notification_configuration(Bucket=bucket)
        except Exception:
            cfg = {}

        # Normalize existing queue configs (if any).
        existing = cfg.get("QueueConfigurations") if isinstance(cfg, dict) else None
        queue_configs = list(existing) if isinstance(existing, list) else []
        for qc in queue_configs:
            if isinstance(qc, dict) and qc.get("QueueArn") == queue_arn:
                return

        queue_configs.append(
            {
                "Id": "webhook-object-created",
                "QueueArn": queue_arn,
                "Events": ["s3:ObjectCreated:*"],
            }
        )
        await client.put_bucket_notification_configuration(
            Bucket=bucket,
            NotificationConfiguration={"QueueConfigurations": queue_configs},
        )


async def presign_put_object_url(
    *,
    bucket: str,
    key: str,
    content_type: str,
    expires_seconds: int = 900,
) -> str:
    # aiobotocore: generate_presigned_url is async — must await (else client gets a coroutine object).
    async with _session().client(
        "s3",
        **_client_kwargs(endpoint_url=_presign_endpoint_url() or _endpoint_url()),
    ) as client:
        url = await client.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": bucket,
                "Key": key,
                "ContentType": content_type,
            },
            ExpiresIn=max(1, int(expires_seconds)),
        )
        return str(url)


async def download_to_temp_file(*, bucket: str, key: str, suffix: str = "") -> Path:
    fd, name = tempfile.mkstemp(prefix="whisper_audio_", suffix=suffix)
    os.close(fd)
    path = Path(name)

    try:
        async with _session().client("s3", **_client_kwargs(endpoint_url=_endpoint_url())) as client:
            await client.download_file(Bucket=bucket, Key=key, Filename=str(path))
        return path
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


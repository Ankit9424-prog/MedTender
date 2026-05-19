import atexit
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import boto3
from pydantic_settings import BaseSettings

from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    MAX_PDF_SIZE_MB: int = 50
    MAX_PAGES: int = 200
    LLM_BATCH_SIZE: int = 15
    LLM_MAX_RETRIES: int = 3

    @property
    def max_pdf_bytes(self) -> int:
        return self.MAX_PDF_SIZE_MB * 1024 * 1024

    def validate_aws_credentials(self) -> bool:
        return bool(self.AWS_ACCESS_KEY_ID and self.AWS_SECRET_ACCESS_KEY)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


_temp_dir: Path | None = None


def get_temp_dir() -> Path:
    global _temp_dir
    if _temp_dir is None or not _temp_dir.exists():
        _temp_dir = Path(tempfile.mkdtemp(prefix="medtender_"))
        atexit.register(_cleanup_temp)
    return _temp_dir


def _cleanup_temp():
    global _temp_dir
    if _temp_dir and _temp_dir.exists():
        shutil.rmtree(_temp_dir, ignore_errors=True)
        _temp_dir = None


@lru_cache(maxsize=1)
def get_bedrock_client():
    settings = get_settings()
    if not settings.validate_aws_credentials():
        raise ValueError(
            "AWS credentials not configured. "
            "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file."
        )
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )

import json
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    runpod_api_key: str
    runpod_endpoint_id: str
    api_keys: str = "{}"  # JSON string: {"app_name": "key"}

    def get_api_keys(self) -> dict[str, str]:
        return json.loads(self.api_keys)

    class Config:
        env_file = ".env"


settings = Settings()

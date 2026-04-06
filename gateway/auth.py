from gateway.config import settings


def validate_api_key(api_key: str) -> str | None:
    """Return app name if key is valid, None otherwise."""
    api_keys = settings.get_api_keys()
    for app_name, key in api_keys.items():
        if key == api_key:
            return app_name
    return None

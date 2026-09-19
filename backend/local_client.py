"""Run existing API routes in-process, without a listening backend server."""
import asyncio
from pathlib import Path
from threading import RLock
from urllib.parse import urlsplit

import httpx
import requests

from .database import SESSION_DB, init_db

# Serialize model/Grad-CAM work across sessions to bound peak memory usage.
_LOCK = RLock()


class LocalClient:
    RequestException = requests.RequestException
    JSONDecodeError = requests.JSONDecodeError

    def __init__(self, directory):
        self.database = Path(directory) / "session.db"

    def request(self, method, url, **kwargs):
        from .main import app

        async def dispatch():
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://local") as client:
                return await client.request(method, urlsplit(url).path, **kwargs)

        with _LOCK:
            token = SESSION_DB.set(self.database)
            try:
                init_db()
                result = asyncio.run(dispatch())
            except Exception as exc:
                raise requests.RequestException("Local screening service failed.") from exc
            finally:
                SESSION_DB.reset(token)
        response = requests.Response()
        response.status_code = result.status_code
        response._content = result.content
        response.headers.update(result.headers)
        response.encoding = "utf-8"
        response.url = url
        return response

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def delete(self, url, **kwargs):
        return self.request("DELETE", url, **kwargs)

#!/usr/bin/env python3
"""Offline tests for vision endpoint readiness/error classification."""

from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from vision_solver import OllamaVisionClient


class _FakeResponse:
    def __init__(self, status: int, payload=None, json_error: Exception | None = None):
        self.status = status
        self._payload = payload
        self._json_error = json_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, *args, **kwargs):
        return self._response


class TestVisionEndpointCheck(unittest.IsolatedAsyncioTestCase):
    async def _check(self, response: _FakeResponse, *, api_key: str = ""):
        logs = []
        client = OllamaVisionClient(
            base="https://vision.invalid",
            model="test-model:latest",
            log=lambda message, level="info": logs.append((level, message)),
        )
        client._api_key = api_key
        with mock.patch(
            "vision_solver.aiohttp.ClientSession",
            return_value=_FakeSession(response),
        ):
            result = await client.check()
        return client, result, logs

    async def test_success_returns_models_and_clears_failure(self):
        client, result, _ = await self._check(
            _FakeResponse(200, {"models": [{"name": "test-model:latest"}]})
        )
        self.assertEqual(result, (True, ["test-model:latest"]))
        self.assertEqual(client.last_check_error, "")
        self.assertIsNone(client.last_check_http_status)

    async def test_401_is_authentication_not_unreachable(self):
        client, result, logs = await self._check(
            _FakeResponse(401, {"error": "Invalid or missing API key."}),
            api_key="configured-client-key",
        )
        self.assertEqual(result, (False, []))
        self.assertEqual(client.last_check_error, "authentication")
        self.assertEqual(client.last_check_http_status, 401)
        self.assertTrue(any("configured VISION_API_KEY was rejected" in line
                            for _, line in logs))

    async def test_404_is_terminal_protocol_mismatch(self):
        client, result, _ = await self._check(_FakeResponse(404, {}))
        self.assertEqual(result, (False, []))
        self.assertEqual(client.last_check_error, "protocol")
        self.assertEqual(client.last_check_http_status, 404)

    async def test_timeout_is_transient(self):
        client = OllamaVisionClient(base="https://vision.invalid")

        class _TimeoutSession(_FakeSession):
            def get(self, *args, **kwargs):
                raise asyncio.TimeoutError

        with mock.patch(
            "vision_solver.aiohttp.ClientSession",
            return_value=_TimeoutSession(_FakeResponse(200, {})),
        ):
            result = await client.check()
        self.assertEqual(result, (False, []))
        self.assertEqual(client.last_check_error, "timeout")
        self.assertIsNone(client.last_check_http_status)

    async def test_invalid_json_is_protocol_error(self):
        client, result, _ = await self._check(
            _FakeResponse(200, json_error=ValueError("not json"))
        )
        self.assertEqual(result, (False, []))
        self.assertEqual(client.last_check_error, "protocol")


if __name__ == "__main__":
    unittest.main()

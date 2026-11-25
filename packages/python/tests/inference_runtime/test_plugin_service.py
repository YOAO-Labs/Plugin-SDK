from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from yoao_protocol_v2.routing import ROUTE_STATUS_PATH

from yoao_plugin_sdk.inference.runtime.plugin_service import (
    install_plugin_service,
    plugin_lifecycle_options,
)


def test_plugin_auth_rejects_missing_and_incorrect_tokens(tmp_path, monkeypatch):
    token = "a" * 64
    token_file = tmp_path / "token"
    token_file.write_text(token)
    control_token = "b" * 64
    control_file = tmp_path / "route-token"
    control_file.write_text(control_token)
    registry = "12345678-1234-4234-8234-123456789abc"
    monkeypatch.setenv("YOAO_PLUGIN_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("YOAO_PLUGIN_DEVICE", "cpu")
    monkeypatch.setenv("YOAO_PLUGIN_REGISTRY_ID", registry)
    monkeypatch.setenv("ROUTE_REGISTRY_ID", registry)
    monkeypatch.setenv("ROUTE_CONTROL_TOKEN_FILE", str(control_file))
    for retired in (
        "AAP_PLUGIN_TOKEN_FILE",
        "AAP_PLUGIN_DEVICE",
        "AAP_PLUGIN_REGISTRY_ID",
        "AAP_ROUTE_REGISTRY_ID",
        "AAP_ROUTE_CONTROL_TOKEN_FILE",
        "AAP_ROUTE_CONTROL_TOKEN",
        "AAP_ROUTE_ALLOW_LEGACY_EXECUTION",
        "YOAO_PLUGIN_GPU_RESOURCE_ID",
        "YOAO_PLUGIN_GPU_UUID",
        "AAP_PLUGIN_GPU_RESOURCE_ID",
        "AAP_PLUGIN_GPU_UUID",
    ):
        monkeypatch.delenv(retired, raising=False)
    app = FastAPI()
    install_plugin_service(app, lambda: None)

    @app.get("/setup")
    def setup():
        return {"service": "plugin"}

    with TestClient(app) as client:
        assert client.get("/setup").status_code == 403
        assert (
            client.get("/setup", headers={"Authorization": "Bearer wrong"}).status_code
            == 403
        )
        response = client.get("/setup", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json() == {"service": "plugin"}
        status = ROUTE_STATUS_PATH
        assert (
            client.get(status, headers={"Authorization": f"Bearer {token}"}).status_code
            == 401
        )
        assert (
            client.get(
                status, headers={"Authorization": f"Bearer {control_token}"}
            ).status_code
            == 503
        )


def test_plugin_lifecycle_rejects_retired_aap_env_vars(monkeypatch):
    monkeypatch.delenv("YOAO_PLUGIN_TOKEN_FILE", raising=False)
    monkeypatch.setenv("AAP_PLUGIN_TOKEN_FILE", "/tmp/legacy-token")
    with pytest.raises(RuntimeError, match="AAP_PLUGIN_TOKEN_FILE"):
        plugin_lifecycle_options()

"""The qualification client binds and uses actual route admission over loopback."""
import httpx
import pytest
from fastapi import FastAPI
from aap_backend_runtime.plugin_service import install_plugin_service
from aap_backend_runtime.qualification_http import bind_operator_routes, http_client


@pytest.mark.asyncio
async def test_operator_binds_real_route_admission(tmp_path, monkeypatch):
    registry = "12345678-1234-4234-8234-123456789abc"
    plugin_token = "plugin-credential-" + "x" * 48
    route_token = "route-credential-" + "y" * 48
    plugin_file = tmp_path / "plugin-token"
    route_file = tmp_path / "route-token"
    plugin_file.write_text(plugin_token)
    route_file.write_text(route_token)
    monkeypatch.setenv("AAP_PLUGIN_TOKEN_FILE", str(plugin_file))
    monkeypatch.setenv("AAP_PLUGIN_DEVICE", "cpu")
    monkeypatch.setenv("AAP_PLUGIN_REGISTRY_ID", registry)
    monkeypatch.setenv("AAP_ROUTE_REGISTRY_ID", registry)
    monkeypatch.setenv("AAP_ROUTE_CONTROL_TOKEN_FILE", str(route_file))
    monkeypatch.delenv("AAP_ROUTE_CONTROL_TOKEN", raising=False)
    monkeypatch.delenv("AAP_PLUGIN_GPU_RESOURCE_ID", raising=False)
    monkeypatch.delenv("AAP_PLUGIN_GPU_UUID", raising=False)
    app = FastAPI()
    install_plugin_service(app, lambda: None)
    calls = []

    @app.post("/work")
    async def work():
        calls.append(1)
        return {"calls": len(calls)}

    async with http_client(app) as client:
        denied = await client.post("/work")
        assert denied.status_code == 503
        await bind_operator_routes(client, registry_id=registry, route_secret=route_token)
        allowed = await client.post("/work")
        assert allowed.status_code == 200
        assert allowed.json() == {"calls": 1}
        async with httpx.AsyncClient(base_url=str(client.base_url), headers={"Authorization": f"Bearer {plugin_token}"}) as unbound:
            rejected = await unbound.post("/work")
            assert rejected.status_code == 401
        assert calls == [1]

"""Ephemeral loopback route authority for the real qualification process only."""
import asyncio
import base64
from contextlib import asynccontextmanager
import os
from pathlib import Path
import secrets
import socket
import time
import uuid

import httpx
import jwt
import uvicorn
from aap_protocol_v2.routing import (
    ROUTE_ADMISSION_AUDIENCE, ROUTE_ADMISSION_HEADER, ROUTE_ADMISSION_PROTOCOL,
    ROUTE_CONTROL_PATH, ROUTE_CONTROL_PROTOCOL,
)


@asynccontextmanager
async def http_client(app):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    sock.setblocking(False)
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="warning", access_log=False))
    server_task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        while not server.started:
            if server_task.done():
                await server_task
                raise RuntimeError("qualification HTTP server exited before startup")
            await asyncio.sleep(0.01)
        token = Path(os.environ["AAP_PLUGIN_TOKEN_FILE"]).read_text().strip()
        async with httpx.AsyncClient(headers={"Authorization": "Bearer " + token}, base_url=f"http://127.0.0.1:{sock.getsockname()[1]}", timeout=600) as client:
            yield client
    finally:
        server.should_exit = True
        try:
            await server_task
        finally:
            sock.close()


async def bind_operator_routes(client, *, registry_id, route_secret):
    key = secrets.token_bytes(32)
    command = {"schema": ROUTE_CONTROL_PROTOCOL, "command_id": str(uuid.uuid4()), "registry_id": registry_id, "authority_epoch": str(uuid.uuid4()), "retired_epoch": None, "challenge": secrets.token_hex(32), "action": "drain", "signing_key": None}
    headers = {"Authorization": f"Bearer {route_secret}"}
    response = await client.post(ROUTE_CONTROL_PATH, json=command, headers=headers)
    response.raise_for_status()
    observed = response.json()
    if (observed.get("registry_id") != registry_id or observed.get("command_id") != command["command_id"] or observed.get("challenge") != command["challenge"] or observed.get("authority_epoch") != command["authority_epoch"] or observed.get("phase") != "drained" or observed.get("active_requests") != 0):
        raise RuntimeError("operator route drain did not bind the reserved registry")
    command["action"] = "bind"
    command["signing_key"] = base64.urlsafe_b64encode(key).decode().rstrip("=")
    response = await client.post(ROUTE_CONTROL_PATH, json=command, headers=headers)
    response.raise_for_status()
    bound = response.json()
    if (bound.get("registry_id") != registry_id or bound.get("command_id") != command["command_id"] or bound.get("challenge") != command["challenge"] or bound.get("authority_epoch") != command["authority_epoch"] or bound.get("backend_incarnation") != observed.get("backend_incarnation") or bound.get("phase") != "bound"):
        raise RuntimeError("operator route bind did not match the drained backend")
    incarnation = bound["backend_incarnation"]

    async def authorize(request):
        if request.method == "GET":
            return
        now = int(time.time())
        claims = {"schema": ROUTE_ADMISSION_PROTOCOL, "aud": ROUTE_ADMISSION_AUDIENCE, "registry_id": registry_id, "backend_incarnation": incarnation, "authority_epoch": command["authority_epoch"], "scope_id": str(uuid.uuid4()), "attempt_id": str(uuid.uuid4()), "operation": "gpu:unload" if request.url.path == "/unload" else "validation", "method": request.method, "path": request.url.path, "iat": now, "exp": now + 60}
        request.headers[ROUTE_ADMISSION_HEADER] = jwt.encode(claims, key, algorithm="HS256", headers={"typ": "JWT"})

    client.event_hooks["request"].append(authorize)

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
import uvicorn

from tool_runtime import LocalMcpServer


def create_app(*, runs_dir: Optional[Path] = None) -> Starlette:
    """Create an ASGI Starlette app serving real MCP JSON-RPC 2.0 endpoints over HTTP."""
    perturbed_server = LocalMcpServer(include_indirect_perturbations=True, runs_dir=runs_dir)
    clean_server = LocalMcpServer(include_indirect_perturbations=False, runs_dir=runs_dir)

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "server": "work121-local-mcp",
                "endpoints": {"perturbed": "/mcp", "clean_baseline": "/mcp/clean"},
                "tools": [t.name for t in perturbed_server.list_tools()],
            }
        )

    async def list_tools_get(request: Request) -> JSONResponse:
        mode = request.query_params.get("mode", "perturbed")
        srv = clean_server if mode == "clean" else perturbed_server
        return JSONResponse(
            srv.handle_jsonrpc({"jsonrpc": "2.0", "id": "http-get-list", "method": "tools/list"})
        )

    async def _handle_rpc(request: Request, srv: LocalMcpServer) -> JSONResponse:
        try:
            body = await request.json()
        except Exception as exc:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                },
                status_code=400,
            )
        if not isinstance(body, dict):
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Request body must be a JSON object."},
                },
                status_code=400,
            )
        result = srv.handle_jsonrpc(body)
        return JSONResponse(result)

    async def mcp_perturbed(request: Request) -> JSONResponse:
        return await _handle_rpc(request, perturbed_server)

    async def mcp_clean(request: Request) -> JSONResponse:
        return await _handle_rpc(request, clean_server)

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/tools", list_tools_get, methods=["GET"]),
            Route("/mcp", mcp_perturbed, methods=["POST"]),
            Route("/mcp/clean", mcp_clean, methods=["POST"]),
        ]
    )


app = create_app()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the work121 live HTTP JSON-RPC 2.0 MCP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8775)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

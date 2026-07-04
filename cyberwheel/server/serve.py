from __future__ import annotations


def run_frontend_server(port) -> None:
    """Serve the experimentation API + web UI on ``0.0.0.0:<port>``.

    Binds all interfaces so the sandbox/container port-publishing path works;
    the UI is for a trusted operator on a trusted network.
    """
    import uvicorn

    from cyberwheel.server.app import create_app

    uvicorn.run(create_app(), host="0.0.0.0", port=int(port), log_level="info")

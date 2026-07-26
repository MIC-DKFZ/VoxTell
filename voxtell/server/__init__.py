"""HTTP inference server for VoxTell (optional; install with ``voxtell[server]``)."""

from voxtell.server.app import make_app

__all__ = ["make_app"]

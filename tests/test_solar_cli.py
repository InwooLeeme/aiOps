"""CLI가 서버 하나의 장애에도 나머지 비교를 수행하는지 검증."""

import contextlib
import importlib.util
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "simulate_cli",
    Path(__file__).resolve().parents[1] / "project/scripts/simulate_drift.py",
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.received.append((self.path, payload))
        result = {
            "dataset_sha256": self.server.dataset_hash,
            "drift_check": {"status": "ok"},
            "retraining": None,
            "served_model_version": "1",
        }
        raw = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(raw)


class CLITests(unittest.TestCase):
    def server(self, digest="same"):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.received = []
        server.dataset_hash = digest
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server, f"http://127.0.0.1:{server.server_port}"

    def test_both_scenarios_on_both_servers_get_identical_requests(self):
        first, a = self.server()
        second, b = self.server()
        output = io.StringIO()
        with (
            patch.object(cli, "TARGETS", {"local": a, "container": b}),
            contextlib.redirect_stdout(output),
        ):
            code = cli.main(["--target", "both", "--scenario", "both"])
        self.assertEqual(code, 0)
        self.assertEqual(first.received, second.received)
        self.assertEqual(
            [payload["scenario"] for _, payload in first.received], ["normal", "drift"]
        )
        report = json.loads(output.getvalue())
        self.assertTrue(report["comparable"])
        self.assertEqual(len(report["results"]), 4)

    def test_failed_first_server_does_not_skip_second(self):
        server, url = self.server()
        output = io.StringIO()
        with (
            patch.object(
                cli, "TARGETS", {"local": "http://127.0.0.1:1", "container": url}
            ),
            contextlib.redirect_stdout(output),
        ):
            code = cli.main(
                ["--target", "both", "--scenario", "normal", "--timeout", "1"]
            )
        self.assertEqual(code, 1)
        self.assertEqual(len(server.received), 1)
        report = json.loads(output.getvalue())
        self.assertFalse(report["results"][0]["ok"])
        self.assertTrue(report["results"][1]["ok"])

    def test_different_data_warns_instead_of_claiming_fair_comparison(self):
        _, a = self.server("A")
        _, b = self.server("B")
        output = io.StringIO()
        with (
            patch.object(cli, "TARGETS", {"local": a, "container": b}),
            contextlib.redirect_stdout(output),
        ):
            code = cli.main(["--target", "both", "--scenario", "normal"])
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(output.getvalue())["comparable"])

    def test_replay_option_keeps_start_and_limit(self):
        server, url = self.server()
        output = io.StringIO()
        with (
            patch.object(cli, "TARGETS", {"local": url}),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(
                cli.main(
                    [
                        "--scenario",
                        "replay",
                        "--start",
                        "2024-06-01T00:00:00",
                        "--limit",
                        "84",
                    ]
                ),
                0,
            )
        self.assertEqual(
            server.received,
            [
                (
                    "/predict/batch-test",
                    {"start_timestamp": "2024-06-01T00:00:00", "limit": 84},
                )
            ],
        )

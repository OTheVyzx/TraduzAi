from __future__ import annotations

import base64
import importlib.util
import io
import json
import unittest
from pathlib import Path

WORKER_PATH = Path(__file__).with_name("worker.py")
SPEC = importlib.util.spec_from_file_location("traduzai_studio_flux_worker", WORKER_PATH)
assert SPEC and SPEC.loader
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)
data_url_to_bytes = WORKER.data_url_to_bytes
validate_request = WORKER.validate_request
variant_seeds = WORKER.variant_seeds


def request_payload(variant_count: int = 2) -> dict:
    return {
        "contract_version": "1.0",
        "job_id": "flux-job",
        "prompt": "reconstruir textura",
        "negative_prompt": "texto",
        "model": "D:/models/flux-fill-local",
        "source_png_data": "data:image/png;base64," + base64.b64encode(b"source").decode("ascii"),
        "mask_png_data": "data:image/png;base64," + base64.b64encode(b"mask").decode("ascii"),
        "width": 512,
        "height": 512,
        "variant_count": variant_count,
        "seed": 42,
        "steps": 20,
        "guidance_scale": 18,
    }


class FluxAdapterContractTests(unittest.TestCase):
    def test_validates_local_contract_and_variant_bounds(self) -> None:
        validated = validate_request(request_payload())
        self.assertEqual(validated["job_id"], "flux-job")
        with self.assertRaisesRegex(ValueError, "2 e 4"):
            validate_request(request_payload(1))
        with self.assertRaisesRegex(ValueError, "2 e 4"):
            validate_request(request_payload(5))

    def test_decodes_only_local_png_data_urls(self) -> None:
        encoded = "data:image/png;base64," + base64.b64encode(b"png-bytes").decode("ascii")
        self.assertEqual(data_url_to_bytes(encoded), b"png-bytes")
        with self.assertRaisesRegex(ValueError, "PNG local"):
            data_url_to_bytes("https://example.com/image.png")

    def test_builds_deterministic_variant_seeds(self) -> None:
        self.assertEqual(variant_seeds(42, 4), [42, 43, 44, 45])

    def test_serves_multiple_jobs_without_restarting_the_worker(self) -> None:
        first = request_payload()
        second = {**request_payload(), "job_id": "flux-job-2"}
        source = io.StringIO(json.dumps(first) + "\n" + json.dumps(second) + "\n")
        output = io.StringIO()
        calls: list[str] = []

        def fake_generation(payload: dict) -> dict:
            calls.append(payload["job_id"])
            return {
                "contract_version": "1.0",
                "job_id": payload["job_id"],
                "provider": "fake-local",
                "model": payload["model"],
                "variants": [],
            }

        self.assertEqual(WORKER.serve(source, output, fake_generation), 0)
        self.assertEqual(calls, ["flux-job", "flux-job-2"])
        self.assertEqual(
            [json.loads(line)["job_id"] for line in output.getvalue().splitlines()],
            ["flux-job", "flux-job-2"],
        )


if __name__ == "__main__":
    unittest.main()

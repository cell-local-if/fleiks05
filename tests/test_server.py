import unittest

from semantic_state_engine.server import health_payload


class HealthPayloadTests(unittest.TestCase):
    def test_health_payload_is_stable(self) -> None:
        self.assertEqual(
            health_payload(),
            {"service": "semantic-state-engine", "status": "ok"},
        )


if __name__ == "__main__":
    unittest.main()

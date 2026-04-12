import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402


class MainEmitTests(unittest.TestCase):
    def setUp(self) -> None:
        main._EMIT_STDOUT_FAILED = False

    def test_emit_swallow_oserror_from_stdout_once(self) -> None:
        stderr = io.StringIO()

        with patch("builtins.print", side_effect=OSError(22, "Invalid argument")):
            with patch.object(main.sys, "stderr", stderr):
                main.emit("progress", message="primeira")
                main.emit("progress", message="segunda")

        log_output = stderr.getvalue()
        self.assertIn("Falha ao emitir evento JSON no stdout", log_output)
        self.assertEqual(log_output.count("Falha ao emitir evento JSON no stdout"), 1)


if __name__ == "__main__":
    unittest.main()

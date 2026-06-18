"""Runner — invokes the project's venv Python."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

if VENV_PYTHON.is_file():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    args = ["-m", "self_healing_rag.main"] + sys.argv[1:]
    sys.exit(subprocess.call([str(VENV_PYTHON)] + args, cwd=str(ROOT), env=env))
else:
    sys.path.insert(0, str(ROOT / "src"))
    from self_healing_rag.main import main
    main()

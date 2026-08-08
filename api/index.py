import sys
from pathlib import Path

# Add the src directory to Python path for Vercel Serverless
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from phishvpn.api.app import app  # noqa: E402

# Vercel Serverless handler
handler = app

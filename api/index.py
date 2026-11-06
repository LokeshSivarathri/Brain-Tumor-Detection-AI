import os
import sys

# Resolve root directory dynamically so backend and dependencies resolve properly
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

for path in [ROOT_DIR, BACKEND_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.app import app

# Vercel looks for the WSGI application callable named 'app'
app = app

if __name__ == "__main__":
    app.run(debug=True, port=8080)


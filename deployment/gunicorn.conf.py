# deployment/gunicorn.conf.py
# Gunicorn configuration for production deployment.
import multiprocessing
import os

# ---- Binding ---------------------------------------------------------------
host = os.getenv("FLASK_HOST", "0.0.0.0")
port = int(os.getenv("FLASK_PORT", 5000))
bind = f"{host}:{port}"

# ---- Workers ---------------------------------------------------------------
# 2 × CPU cores + 1 is the recommended starting point.
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "sync"            # Sync workers are simpler and appropriate here
threads = 1

# ---- Timeouts --------------------------------------------------------------
timeout = int(os.getenv("GUNICORN_TIMEOUT", 120))
graceful_timeout = 30
keepalive = 5

# ---- Logging ---------------------------------------------------------------
accesslog = "logs/api.log"
errorlog = "logs/api.log"
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)sµs'

# ---- Process name ----------------------------------------------------------
proc_name = "product-recommender"

# ---- Preload ---------------------------------------------------------------
# Load the app before forking workers to save memory (copy-on-write).
preload_app = True

# ---- Security --------------------------------------------------------------
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

from pathlib import Path

import fsspec


def is_gcs_path(path: str) -> bool:
    """True for gs:// URIs. `Path(...)` collapses their double slash, so
    data/output paths are kept as plain strings and only routed through
    `Path` for local filesystem operations (e.g. mkdir)."""
    return str(path).startswith("gs://")


def ensure_parent_dir(path: str):
    """mkdir -p the parent directory, skipped for gs:// paths (buckets need no
    directory creation, and Path() would mangle the gs:// URI anyway)."""
    if not is_gcs_path(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def path_exists(path: str) -> bool:
    """True if `path` exists, whether local or a gs:// URI."""
    fs, p = fsspec.core.url_to_fs(str(path))
    return fs.exists(p)

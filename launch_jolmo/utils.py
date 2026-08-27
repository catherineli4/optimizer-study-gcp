import os
import subprocess
import random
import string
from typing import Set, Dict
import sys
import time
from experiments import Project

_exists_cache: Set[str] = set()
_listed_prefixes: Dict[str, int] = {}
_storage_client = None


def _get_storage_client():
    """Lazily build and cache a google-cloud-storage client (reuses one
    authenticated session for all checks)."""
    global _storage_client
    if _storage_client is None:
        from google.cloud import storage
        _storage_client = storage.Client()
    return _storage_client


def blob_exists(path: str) -> bool:
    """Fast existence check for a SINGLE object via the GCS JSON API.

    One metadata request (~60 ms steady state), no per-call gsutil subprocess and
    no directory listing — so the cost scales with the number of objects checked,
    not with how many files live in the directory. Use this instead of
    ``check_exists_remote`` when the parent dir holds a huge number of files (e.g.
    ModelEvaluation). Falls back to a direct gsutil check on any client error.
    """
    path = path.rstrip('/')
    if path in _exists_cache:
        return True
    if not path.startswith("gs://"):
        return check_exists_remote(path, cache_depth=0)
    try:
        bucket_name, _, blob_name = path[len("gs://"):].partition("/")
        # timeout so one wedged connection can't stall a whole sweep's checks;
        # on timeout the except below falls back to gsutil.
        exists = _get_storage_client().bucket(bucket_name).blob(blob_name).exists(timeout=30)
        if exists:
            _exists_cache.add(path)
        return exists
    except Exception:
        # Client/credential problem — fall back to a direct (single-file) gsutil ls.
        return check_exists_remote(path, cache_depth=0)


def check_exists_remote(path: str, cache_depth: int = 3) -> bool:
    path = path.rstrip('/')
    
    if path in _exists_cache:
        return True
    
    if _is_covered_by_listing(path):
        return False
    
    if cache_depth == 0:
        exists = _check_direct(path)
        if exists:
            _exists_cache.add(path)
        return exists
    
    list_prefix = _get_list_prefix(path, cache_depth)
    _list_and_cache(list_prefix, cache_depth)
    return path in _exists_cache


def _check_direct(path: str) -> bool:
    try:
        result = subprocess.run(['gsutil', 'ls', path], capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except Exception:
        return False


def _get_list_prefix(path: str, cache_depth: int) -> str:
    prefix = os.path.dirname(path)
    for _ in range(cache_depth - 1):
        prefix = os.path.dirname(prefix)
    return prefix


def _get_depth_from_prefix(path: str, prefix: str) -> int:
    if not path.startswith(prefix + '/'):
        return -1
    suffix = path[len(prefix) + 1:]
    return suffix.count('/') + 1


def _is_covered_by_listing(path: str) -> bool:
    for prefix, depth in _listed_prefixes.items():
        if path.startswith(prefix + '/'):
            if _get_depth_from_prefix(path, prefix) == depth:
                return True
    return False


def _list_and_cache(prefix: str, depth: int) -> None:
    global _exists_cache, _listed_prefixes
    
    pattern = prefix + '/*' * depth
    print(f'Listing {pattern}...', flush=True, file=sys.stderr, end='')
    start_time = time.time()
    
    try:
        result = subprocess.run(['gsutil', 'ls', pattern], capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line and not line.endswith(':'):
                    _exists_cache.add(line.rstrip('/'))
            _listed_prefixes[prefix] = depth
            print(f'done ({len(_exists_cache)} cached, {elapsed:.2f}s).', flush=True, file=sys.stderr)
        else:
            print(f'no matches ({elapsed:.2f}s).', flush=True, file=sys.stderr)
            _listed_prefixes[prefix] = depth
    except Exception as e:
        print(f'failed with {e}.', flush=True, file=sys.stderr)


def _random_suffix():
    global _random_suffix_cache
    if _random_suffix_cache is None:
        _random_suffix_cache = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    return _random_suffix_cache

def local_path(*relpath):
    if Project.config.cluster == 'orchard':
        base_path = os.path.join(Project.config.local_data_path, _random_suffix())
        if (len(relpath) > 0):
            return os.path.join(base_path, *relpath)
        return base_path
    elif Project.config.cluster == 'babel':
        if (len(relpath) > 0):
            return os.path.join(Project.config.local_data_path, *relpath)
        return Project.config.local_data_path
    else:
        raise ValueError(f'Unknown cluster: {Project.config.cluster}')

def local_cache_path(*relpath):
    if Project.config.cluster == 'orchard':
        base_path = os.path.join(Project.config.local_cache_path, _random_suffix())
        if (len(relpath) > 0):
            return os.path.join(base_path, *relpath)
        return base_path
    elif Project.config.cluster == 'babel':
        if (len(relpath) > 0):
            return os.path.join(Project.config.local_cache_path, *relpath)
        return Project.config.local_cache_path
    else:
        raise ValueError(f'Unknown cluster: {Project.config.cluster}')

def remote_path(*relpath) -> str:
    components = [str(part).strip('/') for part in relpath if part is not None and str(part).strip() != '']
    joined = '/'.join(components)
    return f"{Project.config.remote_path.rstrip('/')}/{joined}" if joined else Project.config.remote_path.rstrip('/')
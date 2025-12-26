#!/usr/bin/env python3
"""
Cache Module for RAG System

Supports:
- Redis (production)
- In-memory LRU cache
- Disk-based cache
"""

import json
import time
import hashlib
from typing import Any, Optional, Dict
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
import pickle


class BaseCache(ABC):
    """Abstract base class for cache."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = None):
        pass

    @abstractmethod
    def delete(self, key: str):
        pass

    @abstractmethod
    def clear(self):
        pass

    def get_stats(self) -> Dict:
        return {}


class LRUCache(BaseCache):
    """In-memory LRU cache."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.expiry = {}
        self.hits = 0
        self.misses = 0

    def _is_expired(self, key: str) -> bool:
        if key not in self.expiry:
            return False
        return time.time() > self.expiry[key]

    def _cleanup_expired(self):
        expired = [k for k in self.cache if self._is_expired(k)]
        for k in expired:
            self.delete(k)

    def get(self, key: str) -> Optional[Any]:
        self._cleanup_expired()

        if key in self.cache and not self._is_expired(key):
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]

        self.misses += 1
        return None

    def set(self, key: str, value: Any, ttl: int = None):
        ttl = ttl or self.default_ttl

        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                oldest = next(iter(self.cache))
                self.delete(oldest)

        self.cache[key] = value
        self.expiry[key] = time.time() + ttl

    def delete(self, key: str):
        self.cache.pop(key, None)
        self.expiry.pop(key, None)

    def clear(self):
        self.cache.clear()
        self.expiry.clear()

    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0
        }


class RedisCache(BaseCache):
    """Redis-based cache for production."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, prefix: str = "rag:"):
        self.prefix = prefix
        self.default_ttl = 3600

        try:
            import redis
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
            self.client.ping()
            self.redis = redis
        except Exception as e:
            print(f"Redis connection failed: {e}. Using fallback.")
            self.client = None
            self._fallback = LRUCache()

    def _make_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        if self.client is not None:
            try:
                value = self.client.get(self._make_key(key))
                if value:
                    return pickle.loads(value)
            except Exception:
                pass
            return None
        else:
            return self._fallback.get(key)

    def set(self, key: str, value: Any, ttl: int = None):
        ttl = ttl or self.default_ttl
        if self.client is not None:
            try:
                self.client.setex(
                    self._make_key(key),
                    ttl,
                    pickle.dumps(value)
                )
            except Exception:
                pass
        else:
            self._fallback.set(key, value, ttl)

    def delete(self, key: str):
        if self.client is not None:
            try:
                self.client.delete(self._make_key(key))
            except Exception:
                pass
        else:
            self._fallback.delete(key)

    def clear(self):
        if self.client is not None:
            try:
                keys = self.client.keys(f"{self.prefix}*")
                if keys:
                    self.client.delete(*keys)
            except Exception:
                pass
        else:
            self._fallback.clear()

    def get_stats(self) -> Dict:
        if self.client is not None:
            try:
                info = self.client.info()
                return {
                    "connected": True,
                    "keys": self.client.dbsize(),
                    "memory_used": info.get("used_memory_human", "N/A"),
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0)
                }
            except Exception:
                return {"connected": False}
        else:
            return self._fallback.get_stats()


class DiskCache(BaseCache):
    """Disk-based cache for persistence."""

    def __init__(self, cache_dir: str = ".cache/rag", max_size_mb: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict:
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"entries": {}, "size": 0}

    def _save_index(self):
        with open(self.index_file, "w") as f:
            json.dump(self.index, f)

    def _get_file_path(self, key: str) -> Path:
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def get(self, key: str) -> Optional[Any]:
        entry = self.index["entries"].get(key)
        if not entry:
            return None

        if time.time() > entry["expiry"]:
            self.delete(key)
            return None

        file_path = self._get_file_path(key)
        if file_path.exists():
            try:
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        file_path = self._get_file_path(key)

        # Serialize
        data = pickle.dumps(value)
        data_size = len(data)

        # Check size limit
        while self.index["size"] + data_size > self.max_size_bytes and self.index["entries"]:
            # Remove oldest entry
            oldest_key = min(self.index["entries"].keys(),
                            key=lambda k: self.index["entries"][k]["created"])
            self.delete(oldest_key)

        # Write file
        with open(file_path, "wb") as f:
            f.write(data)

        # Update index
        self.index["entries"][key] = {
            "file": str(file_path),
            "size": data_size,
            "created": time.time(),
            "expiry": time.time() + ttl
        }
        self.index["size"] += data_size
        self._save_index()

    def delete(self, key: str):
        entry = self.index["entries"].pop(key, None)
        if entry:
            self.index["size"] -= entry["size"]
            file_path = Path(entry["file"])
            if file_path.exists():
                file_path.unlink()
            self._save_index()

    def clear(self):
        for key in list(self.index["entries"].keys()):
            self.delete(key)

    def get_stats(self) -> Dict:
        return {
            "entries": len(self.index["entries"]),
            "size_mb": self.index["size"] / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024)
        }


class CacheManager:
    """Unified cache manager with multiple layers."""

    def __init__(self, l1_cache: BaseCache = None, l2_cache: BaseCache = None):
        self.l1 = l1_cache or LRUCache(max_size=100)
        self.l2 = l2_cache

    def get(self, key: str) -> Optional[Any]:
        # Try L1
        value = self.l1.get(key)
        if value is not None:
            return value

        # Try L2
        if self.l2:
            value = self.l2.get(key)
            if value is not None:
                self.l1.set(key, value)  # Promote to L1
                return value

        return None

    def set(self, key: str, value: Any, ttl: int = None):
        self.l1.set(key, value, ttl)
        if self.l2:
            self.l2.set(key, value, ttl)

    def delete(self, key: str):
        self.l1.delete(key)
        if self.l2:
            self.l2.delete(key)

    def clear(self):
        self.l1.clear()
        if self.l2:
            self.l2.clear()

    def get_stats(self) -> Dict:
        stats = {"l1": self.l1.get_stats()}
        if self.l2:
            stats["l2"] = self.l2.get_stats()
        return stats


if __name__ == "__main__":
    # Test cache
    cache = CacheManager(
        l1_cache=LRUCache(max_size=10),
        l2_cache=DiskCache(cache_dir="/tmp/rag_cache")
    )

    # Set values
    cache.set("key1", {"data": "test1"})
    cache.set("key2", {"data": "test2"})

    # Get values
    print("key1:", cache.get("key1"))
    print("key2:", cache.get("key2"))
    print("key3:", cache.get("key3"))

    # Stats
    print("Stats:", cache.get_stats())

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Optional


# ---------------------------------------------------------
# Cache Utilities
# ---------------------------------------------------------

def build_cache_key(
    *components,
) -> str:
    """
    Build deterministic cache key.
    """

    combined = "||".join(
        str(component)
        for component in components
    )

    return hashlib.sha256(
        combined.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------
# File Cache
# ---------------------------------------------------------

class FileCache:
    """
    Lightweight JSON file cache.

    Useful for:
    -----------------------------------------
    - embedding cache
    - retrieval cache
    - QA response cache
    - expensive pipeline caching
    -----------------------------------------
    """

    def __init__(
        self,
        cache_dir: str = "cache",
        ttl_seconds: Optional[int] = None,
    ):
        self.cache_dir = cache_dir

        self.ttl_seconds = ttl_seconds

        os.makedirs(
            self.cache_dir,
            exist_ok=True,
        )

    # ---------------------------------------------------------
    # Cache Path
    # ---------------------------------------------------------

    def get_cache_path(
        self,
        cache_key: str,
    ) -> str:
        """
        Resolve cache file path.
        """

        return os.path.join(
            self.cache_dir,
            f"{cache_key}.json",
        )

    # ---------------------------------------------------------
    # Exists
    # ---------------------------------------------------------

    def exists(
        self,
        cache_key: str,
    ) -> bool:
        """
        Check whether cache exists.
        """

        path = self.get_cache_path(
            cache_key
        )

        if not os.path.exists(path):
            return False

        # -----------------------------------------------------
        # TTL validation
        # -----------------------------------------------------

        if self.ttl_seconds is not None:

            modified = os.path.getmtime(
                path
            )

            age = (
                time.time()
                - modified
            )

            if age > self.ttl_seconds:

                try:
                    os.remove(path)

                except Exception:
                    pass

                return False

        return True

    # ---------------------------------------------------------
    # Read
    # ---------------------------------------------------------

    def load(
        self,
        cache_key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Load cached object.
        """

        if not self.exists(
            cache_key
        ):

            return None

        path = self.get_cache_path(
            cache_key
        )

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as file:

                return json.load(file)

        except Exception:

            return None

    # ---------------------------------------------------------
    # Write
    # ---------------------------------------------------------

    def save(
        self,
        cache_key: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Save object to cache.
        """

        path = self.get_cache_path(
            cache_key
        )

        try:

            with open(
                path,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    data,
                    file,
                    ensure_ascii=False,
                    separators=(",", ":")
                )

        except Exception as error:

            print(
                f"[CACHE] Save failed: "
                f"{error}"
            )

    # ---------------------------------------------------------
    # Delete
    # ---------------------------------------------------------

    def delete(
        self,
        cache_key: str,
    ) -> None:
        """
        Delete cache entry.
        """

        path = self.get_cache_path(
            cache_key
        )

        if os.path.exists(path):

            try:
                os.remove(path)

            except Exception:
                pass

    # ---------------------------------------------------------
    # Clear
    # ---------------------------------------------------------

    def clear(self) -> None:
        """
        Remove all cache files.
        """

        for filename in os.listdir(
            self.cache_dir
        ):

            path = os.path.join(
                self.cache_dir,
                filename,
            )

            try:

                if os.path.isfile(path):

                    os.remove(path)

            except Exception:
                pass
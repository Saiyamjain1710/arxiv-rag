import json
from pathlib import Path

# Path to the directory containing all individual metadata JSON files
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


def _load_trusted_ids() -> set[str]:
    if not METADATA_DIR.exists() or not METADATA_DIR.is_dir():
        print(f"Warning: Metadata directory not found at '{METADATA_DIR}'. Returning empty trusted set.")
        return set()

    trusted_ids = set()
    # Find all .json files in the directory
    json_files = list(METADATA_DIR.glob("*.json"))

    if not json_files:
        print(f"Warning: No .json files found in '{METADATA_DIR}'. Returning empty trusted set.")
        return set()

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both list of dicts or single dict per file
            records = data if isinstance(data, list) else [data]

            for record in records:
                if isinstance(record, dict) and "arxiv_id" in record:
                    trusted_ids.add(str(record["arxiv_id"]))
        except Exception as e:
            print(f"Warning: Could not read metadata file {json_file.name}: {e}")

    print(f"Loaded {len(trusted_ids)} trusted paper IDs from {len(json_files)} metadata files.")
    return trusted_ids


_trusted_ids_cache = None


def get_trusted_ids() -> set[str]:
    global _trusted_ids_cache
    if _trusted_ids_cache is None:
        _trusted_ids_cache = _load_trusted_ids()
    return _trusted_ids_cache


def filter_trusted(candidates: list[dict], extra_trusted_ids: set[str] | None = None) -> list[dict]:
    """Drops any chunk whose paper_id isn't in our trusted local set."""
    trusted = get_trusted_ids() | (extra_trusted_ids or set())
    return [c for c in candidates if c.get("paper_id") in trusted]
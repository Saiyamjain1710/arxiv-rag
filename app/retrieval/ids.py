import uuid

NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")  # fixed, never change this


def deterministic_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))
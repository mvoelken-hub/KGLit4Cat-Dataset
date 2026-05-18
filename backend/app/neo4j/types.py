from neo4j.time import Date as Neo4jDate
from neo4j.time import DateTime as Neo4jDateTime
from neo4j.time import Time as Neo4jTime
from typing import Any


Neo4jDateTimeType = Neo4jDateTime | Neo4jDate | Neo4jTime


def normalize_neo4j_value(value: Any) -> Any:
    """Convert Neo4j driver values into JSON-serializable Python values."""

    if isinstance(value, Neo4jDateTimeType):
        return value.to_native().isoformat()

    if isinstance(value, dict):
        return {
            str(key): normalize_neo4j_value(nested_value)
            for key, nested_value in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [normalize_neo4j_value(item) for item in value]

    return value

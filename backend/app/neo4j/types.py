from neo4j.time import Date as Neo4jDate
from neo4j.time import DateTime as Neo4jDateTime
from neo4j.time import Time as Neo4jTime


Neo4jDateTimeType = Neo4jDateTime | Neo4jDate | Neo4jTime


from asyncio import Lock, sleep
import time

from neo4j import AsyncGraphDatabase
from rdflib import Graph, Namespace
from rdflib_neo4j import HANDLE_VOCAB_URI_STRATEGY, Neo4jStore, Neo4jStoreConfig
from typing_extensions import deprecated

from app.core.config import Settings, settings
from logging import Logger
from app.core.logging import logger


class Neo4jDriver:
    def __init__(self, settings: Settings, logger: Logger):

        self.neo4j_uri = settings.neo4j_uri
        self.neo4j_user = settings.neo4j_user
        self.neo4j_password = settings.neo4j_password
        self.auth = settings.neo4j_auth

        self._driver = self.get_new_driver()

        self._reconnect_lock = Lock()

        self.databases: dict[str, str] = settings.db_names.copy()

        self.logger = logger

    def get_new_driver(self):
        return AsyncGraphDatabase.driver(
            self.neo4j_uri, auth=self.auth
        )

    async def reconnect(self):
        async with self._reconnect_lock:
            old_driver = self._driver
            self._driver = self.get_new_driver()
            await old_driver.close()

    async def verify_connection(self):
        try:
            await self._driver.verify_connectivity()
            self.logger.info("Successfully connected to Neo4j database at %s", self.neo4j_uri)
        except Exception as e:
            raise ConnectionError("Failed to connect to Neo4j database") from e

    async def wait_for_connection(self, timeout_s: float = 30.0, interval_s: float = 1.0) -> None:
        deadline = time.monotonic() + timeout_s
        last_exc: Exception | None = None

        while time.monotonic() < deadline:
            try:
                await self.verify_connection()
                return
            except Exception as exc:
                last_exc = exc
                self.logger.info("Waiting for Neo4j at %s...", self.neo4j_uri)
                await sleep(interval_s)

        raise ConnectionError(
            f"Neo4j not reachable at {self.neo4j_uri} after {timeout_s:.0f}s"
        ) from last_exc

    async def close(self):
        await self._driver.close()

    async def query(self, query: str, parameters: dict | None = None, db_name: str | None = None) -> list[dict]:
        _db_name = self._resolve_db_name(db_name)
        async with self._driver.session(database=_db_name) as session:
            result = await session.run(query, parameters)  # type: ignore[reportArgumentType]
            return await result.data()

    def add_graph(self, graph: Graph) -> None:
        neo4j_store_config = self._get_neo4j_store_config(list(graph.namespace_manager.namespaces()))

        default_graph_store = Graph(store=Neo4jStore(neo4j_store_config))
        for pf, ns in graph.namespace_manager.namespaces():
            default_graph_store.namespace_manager.bind(pf, ns, override=True, replace=True)
        default_graph_store.open(configuration="", create=True)
        default_graph_store += graph
        default_graph_store.close(commit_pending_transaction=True)

    ## Helpers

    def _get_neo4j_store_config(self, namespaces: list[tuple[str, str]], db_name: str | None = None) -> Neo4jStoreConfig:
        db_name = self._resolve_db_name(db_name)
        custom_prefixes = {prefix: Namespace(uri) for prefix, uri in namespaces}
        return Neo4jStoreConfig(
            auth_data={
                "uri": self.neo4j_uri,
                "database": db_name,
                "user": self.neo4j_user,
                "pwd": self.neo4j_password,
            },
            custom_prefixes=custom_prefixes,
            handle_vocab_uri_strategy=HANDLE_VOCAB_URI_STRATEGY.SHORTEN,
        )

    def _resolve_db_name(self, db_name: str | None) -> str:
        if db_name is None:
            return self.databases["default"]
        if db_name not in self.databases.values():
            raise ValueError(f"Database '{db_name}' not registered in driver.")
        return db_name

neo4j_driver = Neo4jDriver(settings=settings, logger=logger)
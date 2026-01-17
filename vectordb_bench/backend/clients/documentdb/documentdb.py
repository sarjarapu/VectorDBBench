import logging
import time
from contextlib import contextmanager

from pymongo import MongoClient

from ..api import VectorDB
from .config import DocumentDBIndexConfig

log = logging.getLogger(__name__)


class DocumentDBError(Exception):
    """Custom exception class for DocumentDB client errors."""


class DocumentDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DocumentDBIndexConfig,
        collection_name: str = "vdb_bench_collection",
        id_field: str = "id",
        vector_field: str = "vector",
        drop_old: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.id_field = id_field
        self.vector_field = vector_field
        self.drop_old = drop_old

        # Extract version for search syntax selection
        self.docdb_version = getattr(db_case_config, 'docdb_version', '5.0')

        # Build DocumentDB vectorOptions with dimensions
        self.index_params = self.case_config.index_param(dim)
        log.info(f"DocumentDB version: {self.docdb_version}")
        log.info(f"vectorOptions: {self.index_params}")

        # Initialize - they'll also be set in init()
        uri = self.db_config["connection_string"]
        self.client = MongoClient(uri)
        self.db = self.client[self.db_config["database"]]
        self.collection = self.db[self.collection_name]
        if self.drop_old and self.collection_name in self.db.list_collection_names():
            log.info(f"DocumentDB client dropping old collection: {self.collection_name}")
            self.db.drop_collection(self.collection_name)
        self.client = None
        self.db = None
        self.collection = None

    @contextmanager
    def init(self):
        """Initialize DocumentDB client and cleanup when done"""
        try:
            uri = self.db_config["connection_string"]
            self.client = MongoClient(uri)
            self.db = self.client[self.db_config["database"]]
            self.collection = self.db[self.collection_name]

            yield
        finally:
            if self.client is not None:
                self.client.close()
                self.client = None
                self.db = None
                self.collection = None

    def _create_index(self) -> None:
        """Create vector search index using DocumentDB runCommand API"""
        # Check if vector index creation should be skipped
        if getattr(self.case_config, 'skip_vector_index', False):
            log.warning("Skipping vector index creation (skip_vector_index=True)")
            self.collection.create_index(self.id_field)
            log.info(f"Created index on {self.id_field} field")
            return

        index_name = "vector_index"
        vector_options = self.index_params
        log.info(f"Creating index with vectorOptions: {vector_options}")

        # Drop existing vector index if it exists
        try:
            existing_indexes = list(self.collection.list_indexes())
            for idx in existing_indexes:
                if idx.get("name") == index_name:
                    log.info(f"Dropping existing index: {index_name}")
                    self.collection.drop_index(index_name)
                    break
        except Exception:
            log.exception(f"Error checking/dropping index {index_name}")

        try:
            # Create vector index using DocumentDB runCommand syntax
            # See: https://docs.aws.amazon.com/documentdb/latest/developerguide/vector-search.html
            result = self.db.command({
                "createIndexes": self.collection_name,
                "indexes": [{
                    "key": {self.vector_field: "vector"},
                    "vectorOptions": vector_options,
                    "name": index_name,
                }]
            })
            log.info(f"Created vector index: {index_name}, result: {result}")
            self._wait_for_index_ready(index_name)

            # Create regular index on id field for faster lookups
            self.collection.create_index(self.id_field)
            log.info(f"Created index on {self.id_field} field")

        except Exception:
            log.exception(f"Error creating index {index_name}")
            raise

    def _wait_for_index_ready(self, index_name: str, check_interval: int = 5, max_wait: int = 600) -> None:
        """Wait for index to be ready"""
        start_time = time.time()
        while True:
            try:
                # Check if index exists in list_indexes
                indexes = list(self.collection.list_indexes())
                for idx in indexes:
                    if idx.get("name") == index_name:
                        log.info(f"Index {index_name} found: {idx}")
                        return
            except Exception:
                log.exception("Error checking index status")

            if time.time() - start_time > max_wait:
                log.warning(f"Timeout waiting for index {index_name}, proceeding anyway")
                return

            log.info(f"Waiting for index {index_name} to be ready...")
            time.sleep(check_interval)

    def need_normalize_cosine(self) -> bool:
        return False

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception | None):
        """Insert embeddings into DocumentDB"""

        # Prepare documents in bulk
        documents = [
            {
                self.id_field: id_,
                self.vector_field: embedding,
            }
            for id_, embedding in zip(metadata, embeddings, strict=False)
        ]

        # Use ordered=False for better insert performance
        try:
            self.collection.insert_many(documents, ordered=False)
        except Exception as e:
            return 0, e
        return len(documents), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        **kwargs,
    ) -> list[int]:
        """Search for similar vectors - version-aware implementation"""
        search_params = self.case_config.search_param()

        # Select search syntax based on DocumentDB version
        if self._is_version_8_or_higher():
            return self._search_v8(query, k, filters, search_params)
        else:
            return self._search_v5(query, k, filters, search_params)

    def _is_version_8_or_higher(self) -> bool:
        """Check if DocumentDB version supports $vectorSearch syntax"""
        try:
            major_version = int(self.docdb_version.split('.')[0])
            return major_version >= 8
        except (ValueError, AttributeError):
            return False

    def _search_v8(
        self,
        query: list[float],
        k: int,
        filters: dict | None,
        search_params: dict,
    ) -> list[int]:
        """DocumentDB 8.0+ search using $vectorSearch aggregation stage (HNSW)"""
        vector_search = {
            "queryVector": query,
            "index": "vector_index",
            "path": self.vector_field,
            "limit": k,
        }

        # Add exact search parameter if specified
        if search_params.get("exact"):
            vector_search["exact"] = True
        else:
            # Set numCandidates based on k value
            num_candidates = min(10000, k * search_params.get("num_candidates_ratio", 10))
            vector_search["numCandidates"] = num_candidates

        # Add filter if specified
        if filters:
            log.info(f"Applying filter: {filters}")
            vector_search["filter"] = {
                "id": {"$gte": filters["id"]},
            }

        pipeline = [
            {"$vectorSearch": vector_search},
            {
                "$project": {
                    "_id": 0,
                    self.id_field: 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        results = list(self.collection.aggregate(pipeline))
        return [doc[self.id_field] for doc in results]

    def _search_v5(
        self,
        query: list[float],
        k: int,
        filters: dict | None,
        search_params: dict,
    ) -> list[int]:
        """DocumentDB 5.0 search using $search with nested vectorSearch (HNSW)"""
        vector_search_query = {
            "vector": query,
            "path": self.vector_field,
            "k": k,
            "similarity": self.case_config.parse_metric(),
        }

        pipeline = [
            {
                "$search": {
                    "vectorSearch": vector_search_query
                }
            },
            {
                "$project": {
                    "_id": 0,
                    self.id_field: 1,
                    "score": {"$meta": "searchScore"},
                }
            },
        ]

        # Add filter as $match stage after $search if specified
        if filters:
            log.info(f"Applying filter: {filters}")
            pipeline.insert(1, {
                "$match": {
                    "id": {"$gte": filters["id"]}
                }
            })

        results = list(self.collection.aggregate(pipeline))
        return [doc[self.id_field] for doc in results]

    def optimize(self, data_size: int | None = None) -> None:
        """Create vector index after data load"""
        log.info("optimize for search - creating vector index")
        self._create_index()

    def ready_to_load(self) -> None:
        """DocumentDB is always ready to load"""
        pass

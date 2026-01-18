from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class DocumentDBConfig(DBConfig, BaseModel):
    connection_string: SecretStr = "mongodb://superuser:supersecret@localhost:27088/?tls=true&tlsAllowInvalidHostnames=true&tlsAllowInvalidCertificates=true&retryWrites=false&directConnection=true"
    database: str = "vdb_bench"

    def to_dict(self) -> dict:
        return {
            "connection_string": self.connection_string.get_secret_value(),
            "database": self.database,
        }


class DocumentDBIndexConfig(BaseModel, DBCaseConfig):
    index: IndexType = IndexType.HNSW
    metric_type: MetricType = MetricType.COSINE

    # Search parameters
    num_candidates_ratio: int = 10  # numCandidates = k * ratio
    ef_search: int = 256  # HNSW search accuracy (v5 only, higher = better recall)

    # HNSW index parameters
    m: int = 16  # Number of connections per layer
    ef_construction: int = 256  # Size of dynamic candidate list for construction

    # Flags
    exact: bool = False  # Use exact search (no ANN)
    skip_vector_index: bool = False  # Skip vector index creation

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "euclidean"
        if self.metric_type == MetricType.IP:
            return "dotProduct"
        return "cosine"

    def index_param(self, num_dimensions: int) -> dict:
        """Return vectorOptions for DocumentDB createIndexes command"""
        return {
            "type": "hnsw",
            "dimensions": num_dimensions,
            "similarity": self.parse_metric(),
            "m": self.m,
            "efConstruction": self.ef_construction,
        }

    def search_param(self) -> dict:
        return {
            "num_candidates_ratio": self.num_candidates_ratio,
            "exact": self.exact,
            "ef_search": self.ef_search,
        }

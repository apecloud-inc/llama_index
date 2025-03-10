# Using Vector Stores

LlamaIndex offers multiple integration points with vector stores / vector databases:

1. LlamaIndex can use a vector store itself as an index. Like any other index, this index can store documents and be used to answer queries.
2. LlamaIndex can load data from vector stores, similar to any other data connector. This data can then be used within LlamaIndex data structures.

(vector-store-index)=

## Using a Vector Store as an Index

LlamaIndex also supports different vector stores
as the storage backend for `VectorStoreIndex`.

- Azure Cognitive Search (`CognitiveSearchVectorStore`). [Quickstart](https://learn.microsoft.com/en-us/azure/search/search-get-started-vector)
- [Apache Cassandra®](https://cassandra.apache.org/) and compatible databases such as [Astra DB](https://www.datastax.com/press-release/datastax-adds-vector-search-to-astra-db-on-google-cloud-for-building-real-time-generative-ai-applications) (`CassandraVectorStore`)
- Chroma (`ChromaVectorStore`) [Installation](https://docs.trychroma.com/getting-started)
- Epsilla (`EpsillaVectorStore`) [Installation/Quickstart](https://epsilla-inc.gitbook.io/epsilladb/quick-start)
- DeepLake (`DeepLakeVectorStore`) [Installation](https://docs.deeplake.ai/en/latest/Installation.html)
- Elasticsearch (`ElasticsearchStore`) [Installation](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html)
- Qdrant (`QdrantVectorStore`) [Installation](https://qdrant.tech/documentation/install/) [Python Client](https://qdrant.tech/documentation/install/#python-client)
- Weaviate (`WeaviateVectorStore`). [Installation](https://weaviate.io/developers/weaviate/installation). [Python Client](https://weaviate.io/developers/weaviate/client-libraries/python).
- Zep (`ZepVectorStore`). [Installation](https://docs.getzep.com/deployment/quickstart/). [Python Client](https://docs.getzep.com/sdk/).
- Pinecone (`PineconeVectorStore`). [Installation/Quickstart](https://docs.pinecone.io/docs/quickstart).
- Faiss (`FaissVectorStore`). [Installation](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).
- Milvus (`MilvusVectorStore`). [Installation](https://milvus.io/docs)
- Zilliz (`MilvusVectorStore`). [Quickstart](https://zilliz.com/doc/quick_start)
- MyScale (`MyScaleVectorStore`). [Quickstart](https://docs.myscale.com/en/quickstart/). [Installation/Python Client](https://docs.myscale.com/en/python-client/).
- Supabase (`SupabaseVectorStore`). [Quickstart](https://supabase.github.io/vecs/api/).
- DocArray (`DocArrayHnswVectorStore`, `DocArrayInMemoryVectorStore`). [Installation/Python Client](https://github.com/docarray/docarray#installation).
- MongoDB Atlas (`MongoDBAtlasVectorSearch`). [Installation/Quickstart](https://www.mongodb.com/atlas/database).
- Redis (`RedisVectorStore`). [Installation](https://redis.io/docs/getting-started/installation/).
- Neo4j (`Neo4jVectorIndex`). [Installation](https://neo4j.com/docs/operations-manual/current/installation/).
- TimeScale (`TimescaleVectorStore`). [Installation](https://github.com/timescale/python-vector).
- DashVector (`DashVectorStore`). [Installation](https://help.aliyun.com/document_detail/2510230.html).
- AstraDB (`AstraDBVectorStore`). [Quickstart](https://docs.datastax.com/en/home/docs/index.html).
- Lantern (`LanternVectorStore`). [Quickstart](https://docs.lantern.dev/get-started/overview).

A detailed API reference is [found here](/api_reference/indices/vector_store.rst).

Similar to any other index within LlamaIndex (tree, keyword table, list), `VectorStoreIndex` can be constructed upon any collection
of documents. We use the vector store within the index to store embeddings for the input text chunks.

Once constructed, the index can be used for querying.

**Default Vector Store Index Construction/Querying**

By default, `VectorStoreIndex` uses a in-memory `SimpleVectorStore`
that's initialized as part of the default storage context.

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader

# Load documents and build index
documents = SimpleDirectoryReader("../paul_graham_essay/data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Query index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
```

**Custom Vector Store Index Construction/Querying**

We can query over a custom vector store as follows:

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores import DeepLakeVectorStore

# construct vector store and customize storage context
storage_context = StorageContext.from_defaults(
    vector_store=DeepLakeVectorStore(dataset_path="<dataset_path>")
)

# Load documents and build index
documents = SimpleDirectoryReader("../paul_graham_essay/data").load_data()
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# Query index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
```

Below we show more examples of how to construct various vector stores we support.

**Elasticsearch**

First, you can start Elasticsearch either locally or on [Elastic cloud](https://cloud.elastic.co/registration?utm_source=llama-index&utm_content=documentation).

To start Elasticsearch locally with docker, run the following command:

```bash
docker run -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "xpack.security.http.ssl.enabled=false" \
  -e "xpack.license.self_generated.type=trial" \
  docker.elastic.co/elasticsearch/elasticsearch:8.9.0
```

Then connect and use Elasticsearch as a vector database with LlamaIndex

```python
from llama_index.vector_stores import ElasticsearchStore

vector_store = ElasticsearchStore(
    index_name="llm-project",
    es_url="http://localhost:9200",
    # Cloud connection options:
    # es_cloud_id="<cloud_id>",
    # es_user="elastic",
    # es_password="<password>",
)
```

This can be used with the `VectorStoreIndex` to provide a query interface for retrieval, querying, deleting, persisting the index, and more.

**Redis**

First, start Redis-Stack (or get url from Redis provider)

```bash
docker run --name redis-vecdb -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

Then connect and use Redis as a vector database with LlamaIndex

```python
from llama_index.vector_stores import RedisVectorStore

vector_store = RedisVectorStore(
    index_name="llm-project",
    redis_url="redis://localhost:6379",
    overwrite=True,
)
```

This can be used with the `VectorStoreIndex` to provide a query interface for retrieval, querying, deleting, persisting the index, and more.

**DeepLake**

```python
import os
import getpath
from llama_index.vector_stores import DeepLakeVectorStore

os.environ["OPENAI_API_KEY"] = getpath.getpath("OPENAI_API_KEY: ")
os.environ["ACTIVELOOP_TOKEN"] = getpath.getpath("ACTIVELOOP_TOKEN: ")
dataset_path = "hub://adilkhan/paul_graham_essay"

# construct vector store
vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
```

**Faiss**

```python
import faiss
from llama_index.vector_stores import FaissVectorStore

# create faiss index
d = 1536
faiss_index = faiss.IndexFlatL2(d)

# construct vector store
vector_store = FaissVectorStore(faiss_index)

...

# NOTE: since faiss index is in-memory, we need to explicitly call
#       vector_store.persist() or storage_context.persist() to save it to disk.
#       persist() takes in optional arg persist_path. If none give, will use default paths.
storage_context.persist()
```

**Weaviate**

```python
import weaviate
from llama_index.vector_stores import WeaviateVectorStore

# creating a Weaviate client
resource_owner_config = weaviate.AuthClientPassword(
    username="<username>",
    password="<password>",
)
client = weaviate.Client(
    "https://<cluster-id>.semi.network/",
    auth_client_secret=resource_owner_config,
)

# construct vector store
vector_store = WeaviateVectorStore(weaviate_client=client)
```

**Zep**

Zep stores texts, metadata, and embeddings. All are returned in search results.

```python
from llama_index.vector_stores.zep import ZepVectorStore

vector_store = ZepVectorStore(
    api_url="<api_url>",
    api_key="<api_key>",
    collection_name="<unique_collection_name>",  # Can either be an existing collection or a new one
    embedding_dimensions=1536,  # Optional, required if creating a new collection
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# Query index using both a text query and metadata filters
filters = MetadataFilters(
    filters=[ExactMatchFilter(key="theme", value="Mafia")]
)
retriever = index.as_retriever(filters=filters)
result = retriever.retrieve("What is inception about?")
```

**Pinecone**

```python
import pinecone
from llama_index.vector_stores import PineconeVectorStore

# Creating a Pinecone index
api_key = "api_key"
pinecone.init(api_key=api_key, environment="us-west1-gcp")
pinecone.create_index(
    "quickstart", dimension=1536, metric="euclidean", pod_type="p1"
)
index = pinecone.Index("quickstart")

# can define filters specific to this vector index (so you can
# reuse pinecone indexes)
metadata_filters = {"title": "paul_graham_essay"}

# construct vector store
vector_store = PineconeVectorStore(
    pinecone_index=index, metadata_filters=metadata_filters
)
```

**Qdrant**

```python
import qdrant_client
from llama_index.vector_stores import QdrantVectorStore

# Creating a Qdrant vector store
client = qdrant_client.QdrantClient(
    host="<qdrant-host>", api_key="<qdrant-api-key>", https=True
)
collection_name = "paul_graham"

# construct vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
)
```

**Cassandra** (covering DataStax Astra DB cloud instances as well)

```python
from llama_index.vector_stores import CassandraVectorStore
import cassio

# For an Astra DB cloud instance:
cassio.init(database_id="1234abcd-...", token="AstraCS:...")

# For a Cassandra cluster:
from cassandra.cluster import Cluster

cluster = Cluster(["127.0.0.1"])
cassio.init(session=cluster.connect(), keyspace="my_keyspace")

# After the above `cassio.init(...)`, create a vector store:
vector_store = CassandraVectorStore(
    table="cass_v_table", embedding_dimension=1536
)
```

**Chroma**

```python
import chromadb
from llama_index.vector_stores import ChromaVectorStore

# Creating a Chroma client
# EphemeralClient operates purely in-memory, PersistentClient will also save to disk
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# construct vector store
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection,
)
```

**Epsilla**

```python
from pyepsilla import vectordb
from llama_index.vector_stores import EpsillaVectorStore

# Creating an Epsilla client
epsilla_client = vectordb.Client()

# Construct vector store
vector_store = EpsillaVectorStore(client=epsilla_client)
```

**Note**: `EpsillaVectorStore` depends on the `pyepsilla` library and a running Epsilla vector database.
Use `pip/pip3 install pyepsilla` if not installed yet.
A running Epsilla vector database could be found through docker image.
For complete instructions, see the following documentation:
https://epsilla-inc.gitbook.io/epsilladb/quick-start

**Milvus**

- Milvus Index offers the ability to store both Documents and their embeddings.

```python
import pymilvus
from llama_index.vector_stores import MilvusVectorStore

# construct vector store
vector_store = MilvusVectorStore(
    uri="https://localhost:19530", overwrite="True"
)
```

**Note**: `MilvusVectorStore` depends on the `pymilvus` library.
Use `pip install pymilvus` if not already installed.
If you get stuck at building wheel for `grpcio`, check if you are using python 3.11
(there's a known issue: https://github.com/milvus-io/pymilvus/issues/1308)
and try downgrading.

**Zilliz**

- Zilliz Cloud (hosted version of Milvus) uses the Milvus Index with some extra arguments.

```python
import pymilvus
from llama_index.vector_stores import MilvusVectorStore


# construct vector store
vector_store = MilvusVectorStore(
    uri="foo.vectordb.zillizcloud.com",
    token="your_token_here",
    overwrite="True",
)
```

**Note**: `MilvusVectorStore` depends on the `pymilvus` library.
Use `pip install pymilvus` if not already installed.
If you get stuck at building wheel for `grpcio`, check if you are using python 3.11
(there's a known issue: https://github.com/milvus-io/pymilvus/issues/1308)
and try downgrading.

**MyScale**

```python
import clickhouse_connect
from llama_index.vector_stores import MyScaleVectorStore

# Creating a MyScale client
client = clickhouse_connect.get_client(
    host="YOUR_CLUSTER_HOST",
    port=8443,
    username="YOUR_USERNAME",
    password="YOUR_CLUSTER_PASSWORD",
)


# construct vector store
vector_store = MyScaleVectorStore(myscale_client=client)
```

**Timescale**

```python
from llama_index.vector_stores import TimescaleVectorStore

vector_store = TimescaleVectorStore.from_params(
    service_url="YOUR TIMESCALE SERVICE URL",
    table_name="paul_graham_essay",
)
```

**SingleStore**

```python
from llama_index.vector_stores import SingleStoreVectorStore
import os

# can set the singlestore db url in env
# or pass it in as an argument to the SingleStoreVectorStore constructor
os.environ["SINGLESTOREDB_URL"] = "PLACEHOLDER URL"
vector_store = SingleStoreVectorStore(
    table_name="embeddings",
    content_field="content",
    metadata_field="metadata",
    vector_field="vector",
    timeout=30,
)
```

**DocArray**

```python
from llama_index.vector_stores import (
    DocArrayHnswVectorStore,
    DocArrayInMemoryVectorStore,
)

# construct vector store
vector_store = DocArrayHnswVectorStore(work_dir="hnsw_index")

# alternatively, construct the in-memory vector store
vector_store = DocArrayInMemoryVectorStore()
```

**MongoDBAtlas**

```python
# Provide URI to constructor, or use environment variable
import pymongo
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.readers.file.base import SimpleDirectoryReader

# mongo_uri = os.environ["MONGO_URI"]
mongo_uri = (
    "mongodb+srv://<username>:<password>@<host>?retryWrites=true&w=majority"
)
mongodb_client = pymongo.MongoClient(mongo_uri)

# construct store
store = MongoDBAtlasVectorSearch(mongodb_client)
storage_context = StorageContext.from_defaults(vector_store=store)
uber_docs = SimpleDirectoryReader(
    input_files=["../data/10k/uber_2021.pdf"]
).load_data()

# construct index
index = VectorStoreIndex.from_documents(
    uber_docs, storage_context=storage_context
)
```

**Neo4j**

- Neo4j stores texts, metadata, and embeddings and can be customized to return graph data in the form of metadata.

```python
from llama_index.vector_stores import Neo4jVectorStore

# construct vector store
neo4j_vector = Neo4jVectorStore(
    username="neo4j",
    password="pleaseletmein",
    url="bolt://localhost:7687",
    embed_dim=1536,
)
```

**Azure Cognitive Search**

```python
from azure.search.documents import SearchClient
from llama_index.vector_stores import ChromaVectorStore
from azure.core.credentials import AzureKeyCredential

service_endpoint = f"https://{search_service_name}.search.windows.net"
index_name = "quickstart"
cognitive_search_credential = AzureKeyCredential("<API key>")

search_client = SearchClient(
    endpoint=service_endpoint,
    index_name=index_name,
    credential=cognitive_search_credential,
)

# construct vector store
vector_store = CognitiveSearchVectorStore(
    search_client,
    id_field_key="id",
    chunk_field_key="content",
    embedding_field_key="embedding",
    metadata_field_key="li_jsonMetadata",
    doc_id_field_key="li_doc_id",
)
```

**DashVector**

```python
import dashvector
from llama_index.vector_stores import DashVectorStore

# init dashvector client
client = dashvector.Client(api_key="your-dashvector-api-key")

# creating a DashVector collection
client.create("quickstart", dimension=1536)
collection = client.get("quickstart")

# construct vector store
vector_store = DashVectorStore(collection)
```

[Example notebooks can be found here](https://github.com/jerryjliu/llama_index/tree/main/docs/examples/vector_stores).

## Loading Data from Vector Stores using Data Connector

LlamaIndex supports loading data from a huge number of sources. See [Data Connectors](/module_guides/loading/connector/modules.md) for more details and API documentation.

Chroma stores both documents and vectors. This is an example of how to use Chroma:

```python
from llama_index.readers.chroma import ChromaReader
from llama_index.indices import SummaryIndex

# The chroma reader loads data from a persisted Chroma collection.
# This requires a collection name and a persist directory.
reader = ChromaReader(
    collection_name="chroma_collection",
    persist_directory="examples/data_connectors/chroma_collection",
)

query_vector = [n1, n2, n3, ...]

documents = reader.load_data(
    collection_name="demo", query_vector=query_vector, limit=5
)
index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")
display(Markdown(f"<b>{response}</b>"))
```

Qdrant also stores both documents and vectors. This is an example of how to use Qdrant:

```python
from llama_index.readers.qdrant import QdrantReader

reader = QdrantReader(host="localhost")

# the query_vector is an embedding representation of your query_vector
# Example query_vector
# query_vector = [0.3, 0.3, 0.3, 0.3, ...]

query_vector = [n1, n2, n3, ...]

# NOTE: Required args are collection_name, query_vector.
# See the Python client: https;//github.com/qdrant/qdrant_client
# for more details

documents = reader.load_data(
    collection_name="demo", query_vector=query_vector, limit=5
)
```

NOTE: Since Weaviate can store a hybrid of document and vector objects, the user may either choose to explicitly specify `class_name` and `properties` in order to query documents, or they may choose to specify a raw GraphQL query. See below for usage.

```python
# option 1: specify class_name and properties

# 1) load data using class_name and properties
documents = reader.load_data(
    class_name="<class_name>",
    properties=["property1", "property2", "..."],
    separate_documents=True,
)

# 2) example GraphQL query
query = """
{
    Get {
        <class_name> {
            <property1>
            <property2>
        }
    }
}
"""

documents = reader.load_data(graphql_query=query, separate_documents=True)
```

NOTE: Both Pinecone and Faiss data loaders assume that the respective data sources only store vectors; text content is stored elsewhere. Therefore, both data loaders require that the user specifies an `id_to_text_map` in the load_data call.

For instance, this is an example usage of the Pinecone data loader `PineconeReader`:

```python
from llama_index.readers.pinecone import PineconeReader

reader = PineconeReader(api_key=api_key, environment="us-west1-gcp")

id_to_text_map = {
    "id1": "text blob 1",
    "id2": "text blob 2",
}

query_vector = [n1, n2, n3, ...]

documents = reader.load_data(
    index_name="quickstart",
    id_to_text_map=id_to_text_map,
    top_k=3,
    vector=query_vector,
    separate_documents=True,
)
```

[Example notebooks can be found here](https://github.com/jerryjliu/llama_index/tree/main/docs/examples/data_connectors).

```{toctree}
---
caption: Examples
maxdepth: 1
---
../../examples/vector_stores/Elasticsearch_demo.ipynb
../../examples/vector_stores/SimpleIndexDemo.ipynb
../../examples/vector_stores/SimpleIndexDemoMMR.ipynb
../../examples/vector_stores/RedisIndexDemo.ipynb
../../examples/vector_stores/QdrantIndexDemo.ipynb
../../examples/vector_stores/FaissIndexDemo.ipynb
../../examples/vector_stores/DeepLakeIndexDemo.ipynb
../../examples/vector_stores/MyScaleIndexDemo.ipynb
../../examples/vector_stores/MetalIndexDemo.ipynb
../../examples/vector_stores/WeaviateIndexDemo.ipynb
../../examples/vector_stores/ZepIndexDemo.ipynb
../../examples/vector_stores/OpensearchDemo.ipynb
../../examples/vector_stores/PineconeIndexDemo.ipynb
../../examples/vector_stores/CassandraIndexDemo.ipynb
../../examples/vector_stores/ChromaIndexDemo.ipynb
../../examples/vector_stores/EpsillaIndexDemo.ipynb
../../examples/vector_stores/LanceDBIndexDemo.ipynb
../../examples/vector_stores/MilvusIndexDemo.ipynb
../../examples/vector_stores/WeaviateIndexDemo-Hybrid.ipynb
../../examples/vector_stores/PineconeIndexDemo-Hybrid.ipynb
../../examples/vector_stores/AsyncIndexCreationDemo.ipynb
../../examples/vector_stores/SupabaseVectorIndexDemo.ipynb
../../examples/vector_stores/DocArrayHnswIndexDemo.ipynb
../../examples/vector_stores/DocArrayInMemoryIndexDemo.ipynb
../../examples/vector_stores/MongoDBAtlasVectorSearch.ipynb
../../examples/vector_stores/postgres.ipynb
../../examples/vector_stores/AwadbDemo.ipynb
../../examples/vector_stores/Neo4jVectorDemo.ipynb
../../examples/vector_stores/CognitiveSearchIndexDemo.ipynb
../../examples/vector_stores/Timescalevector.ipynb
../../examples/vector_stores/SingleStoreDemo.ipynb
../../examples/vector_stores/DashvectorIndexDemo.ipynb
```

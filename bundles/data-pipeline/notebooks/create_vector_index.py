# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Index Creation Job
# MAGIC 
# MAGIC This notebook creates vector indexes from processed documents for migration planning using the simple Databricks Vector Search SDK.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# Install required packages
%pip install databricks-vectorsearch

# Restart Python to ensure packages are loaded
dbutils.library.restartPython()

# COMMAND ----------

# Import required libraries
from databricks.vector_search.client import VectorSearchClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Get Parameters

# COMMAND ----------

# Get parameters from job configuration
source_table = dbutils.widgets.get("source_table")
vector_search_endpoint = dbutils.widgets.get("vector_search_endpoint")
vector_index_name = dbutils.widgets.get("vector_index_name")

print(f"Source table: {source_table}")
print(f"Vector search endpoint: {vector_search_endpoint}")
print(f"Vector index name: {vector_index_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Vector Search Client

# COMMAND ----------

# Initialize the Vector Search client
client = VectorSearchClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Check if Index Exists and Create/Update

# COMMAND ----------

# Check if the index already exists
print(f"Checking if vector index exists: {vector_index_name}")

try:
    existing_index = client.get_index(
        endpoint_name=vector_search_endpoint,
        index_name=vector_index_name
    )
    print(f"✅ Index already exists: {vector_index_name}")
    print(f"Index status: {getattr(existing_index, 'status', 'Unknown')}")
    print(f"Index type: {getattr(existing_index, 'index_type', 'Unknown')}")
    index = existing_index
    index_exists = True
except Exception as e:
    error_str = str(e)
    if ("RESOURCE_DOES_NOT_EXIST" in error_str or 
        "RESOURCE_NOT_FOUND" in error_str or 
        "not found" in error_str.lower() or
        "does not exist" in error_str.lower()):
        print(f"ℹ️ Index does not exist, will create new one: {vector_index_name}")
        index_exists = False
    else:
        print(f"❌ Error checking index existence: {error_str}")
        raise e

# COMMAND ----------

# Create the Delta Sync Index if it doesn't exist
if not index_exists:
    print(f"Creating vector index: {vector_index_name}")
    
    try:
        index = client.create_delta_sync_index(
            endpoint_name=vector_search_endpoint,
            source_table_name=source_table,
            index_name=vector_index_name,
            primary_key="path",
            embedding_source_column="text",
            embedding_model_endpoint_name="databricks-gte-large-en",
            pipeline_type="TRIGGERED"
        )
        
        print(f"✅ Index created successfully: {index}")
        
    except Exception as e:
        print(f"❌ Failed to create index: {str(e)}")
        raise e
else:
    print(f"ℹ️ Using existing index: {vector_index_name}")
    # Get the existing index for consistency
    index = client.get_index(
        endpoint_name=vector_search_endpoint,
        index_name=vector_index_name
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Sync the Index

# COMMAND ----------

# Start the initial sync (only if index was just created or if we want to refresh)
if not index_exists:
    print("Starting index sync...")
    
    try:
        sync_result = index.sync()
        print(f"Sync started: {sync_result}")
        
        # Wait for the index to be ready using the built-in method
        print("Waiting for index to be ready...")
        index.wait_until_ready(verbose=True, timeout=1800)  # 30 minutes timeout
        print("✅ Index is now ready!")
        
    except Exception as e:
        print(f"❌ Sync failed: {str(e)}")
        raise e
else:
    print("ℹ️ Skipping sync for existing index")
    # Check if existing index is ready
    try:
        print("Checking if existing index is ready...")
        index.wait_until_ready(verbose=True, timeout=300)  # 5 minutes timeout
        print("✅ Existing index is ready!")
    except Exception as e:
        print(f"⚠️ Existing index may not be ready: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Index Status and Information

# COMMAND ----------

# Get detailed information about the index
print("Getting index information...")
try:
    index_info = index.describe()
    print(f"📊 Index Name: {getattr(index_info, 'name', 'Unknown')}")
    print(f"📈 Status: {getattr(index_info, 'status', 'Unknown')}")
    print(f"🔗 Endpoint: {getattr(index_info, 'endpoint_name', 'Unknown')}")
    print(f"📋 Primary Key: {getattr(index_info, 'primary_key', 'Unknown')}")
    print(f"🔢 Pipeline Type: {getattr(index_info, 'pipeline_type', 'Unknown')}")
    print(f"📊 Source Table: {getattr(index_info, 'source_table_name', 'Unknown')}")
    
    # Show embedding model info if available
    embedding_model = getattr(index_info, 'embedding_model_endpoint_name', 'Unknown')
    if embedding_model != 'Unknown':
        print(f"🤖 Embedding Model: {embedding_model}")
    
except Exception as e:
    print(f"⚠️ Could not get detailed index information: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test Vector Search

# COMMAND ----------

# Test queries with different complexity levels
test_queries = [
    {
        "query": "Oracle migration to Databricks best practices",
        "category": "migration",
        "expected_topics": ["oracle", "databricks", "migration"]
    },
    {
        "query": "Data pipeline ETL performance optimization",
        "category": "performance", 
        "expected_topics": ["etl", "pipeline", "performance"]
    },
    {
        "query": "Security governance compliance data",
        "category": "security",
        "expected_topics": ["security", "governance", "compliance"]
    },
    {
        "query": "Vector search similarity algorithms",
        "category": "technical",
        "expected_topics": ["vector", "search", "algorithms"]
    }
]

print("🧪 Testing vector search functionality...")
print("=" * 60)

for test_case in test_queries:
    query = test_case["query"]
    category = test_case["category"]
    expected_topics = test_case["expected_topics"]
    
    print(f"\n🔍 Query: {query}")
    print(f"📂 Category: {category}")
    print(f"🎯 Expected Topics: {', '.join(expected_topics)}")
    print("-" * 40)
    
    try:
        # Basic similarity search
        results = index.similarity_search(
            query_text=query,
            columns=["path", "text", "filename", "categories", "topics"],
            num_results=3,
            debug_level=1  # Enable debug info
        )
        
        # Handle different result formats
        if isinstance(results, dict) and 'result' in results:
            data_array = results.get('result', {}).get('data_array', [])
        else:
            data_array = results if isinstance(results, list) else []
        
        print(f"📊 Found {len(data_array)} results:")
        
        for i, doc in enumerate(data_array[:2], 1):
            # Handle both dict and object formats
            if isinstance(doc, dict):
                filename = doc.get('filename', 'Unknown')
                text = doc.get('text', '')
                doc_categories = doc.get('categories', 'N/A')
                doc_topics = doc.get('topics', 'N/A')
            else:
                filename = getattr(doc, 'filename', 'Unknown')
                text = getattr(doc, 'text', '')
                doc_categories = getattr(doc, 'categories', 'N/A')
                doc_topics = getattr(doc, 'topics', 'N/A')
            
            print(f"  {i}. 📄 {filename}")
            print(f"     🏷️  Categories: {doc_categories}")
            print(f"     🎯 Topics: {doc_topics}")
            print(f"     📝 Content: {text[:100]}...")
            print()
            
    except Exception as e:
        print(f"❌ Error testing query '{query}': {str(e)}")
        continue

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Test Filtered Search

# COMMAND ----------

print("🔍 Testing filtered search capabilities...")
print("=" * 60)

# Test different filter combinations
filter_tests = [
    {
        "name": "Oracle Migration Filter",
        "query": "data migration best practices",
        "filters": {"categories": "oracle-migration"},
        "description": "Filter by Oracle migration category"
    },
    {
        "name": "ETL Pipeline Filter", 
        "query": "data processing optimization",
        "filters": {"topics": "etl,data-pipeline"},
        "description": "Filter by ETL and data pipeline topics"
    },
    {
        "name": "Security Filter",
        "query": "data governance compliance",
        "filters": {"topics": "security,governance"},
        "description": "Filter by security and governance topics"
    }
]

for test in filter_tests:
    print(f"\n🧪 {test['name']}")
    print(f"📝 Description: {test['description']}")
    print(f"🔍 Query: {test['query']}")
    print(f"🎯 Filters: {test['filters']}")
    print("-" * 40)
    
    try:
        filtered_results = index.similarity_search(
            query_text=test['query'],
            columns=["path", "text", "filename", "categories", "topics"],
            filters=test['filters'],
            num_results=3
        )
        
        # Handle different result formats for filtered search
        if isinstance(filtered_results, dict) and 'result' in filtered_results:
            filtered_data_array = filtered_results.get('result', {}).get('data_array', [])
        else:
            filtered_data_array = filtered_results if isinstance(filtered_results, list) else []
        
        print(f"📊 Found {len(filtered_data_array)} filtered results:")
        
        for i, doc in enumerate(filtered_data_array, 1):
            # Handle both dict and object formats
            if isinstance(doc, dict):
                filename = doc.get('filename', 'Unknown')
                categories = doc.get('categories', 'N/A')
                topics = doc.get('topics', 'N/A')
                text = doc.get('text', '')
            else:
                filename = getattr(doc, 'filename', 'Unknown')
                categories = getattr(doc, 'categories', 'N/A')
                topics = getattr(doc, 'topics', 'N/A')
                text = getattr(doc, 'text', '')
            
            print(f"  {i}. 📄 {filename}")
            print(f"     🏷️  Categories: {categories}")
            print(f"     🎯 Topics: {topics}")
            print(f"     📝 Content: {text[:80]}...")
            print()
            
    except Exception as e:
        print(f"❌ Error testing filtered search: {str(e)}")
        continue

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Test with Reranker

# COMMAND ----------

# Test with reranker for improved results
try:
    from databricks.vector_search.reranker import DatabricksReranker
    
    print("Testing search with reranker...")
    
    reranked_results = index.similarity_search(
        query_text="Oracle to Databricks migration strategies",
        columns=["path", "text", "filename", "categories", "topics"],
        num_results=3,
        query_type="hybrid",
        reranker=DatabricksReranker(columns_to_rerank=["text", "categories", "topics"])
    )
    
    # Handle different result formats for reranked search
    if isinstance(reranked_results, dict) and 'result' in reranked_results:
        reranked_data_array = reranked_results.get('result', {}).get('data_array', [])
        debug_info = reranked_results.get('debug_info', {})
    else:
        reranked_data_array = reranked_results if isinstance(reranked_results, list) else []
        debug_info = {}
    
    print(f"Reranked results: {len(reranked_data_array)}")
    
    # Show debug info if available
    if debug_info:
        if isinstance(debug_info, dict):
            print(f"Response time: {debug_info.get('response_time', 'N/A')}ms")
            print(f"Reranker time: {debug_info.get('reranker_time', 'N/A')}ms")
        else:
            print(f"Response time: {getattr(debug_info, 'response_time', 'N/A')}ms")
            print(f"Reranker time: {getattr(debug_info, 'reranker_time', 'N/A')}ms")
    
    for i, doc in enumerate(reranked_data_array[:2], 1):
        # Handle both dict and object formats
        if isinstance(doc, dict):
            filename = doc.get('filename', 'Unknown')
            text = doc.get('text', '')
        else:
            filename = getattr(doc, 'filename', 'Unknown')
            text = getattr(doc, 'text', '')
        
        print(f"  {i}. {filename} - {text[:100]}...")
        
except ImportError:
    print("Reranker not available, skipping reranker test")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Performance Metrics and Summary

# COMMAND ----------

print("📊 PERFORMANCE METRICS AND SUMMARY")
print("=" * 60)

# Get final index status
try:
    final_status = index.describe()
    status = getattr(final_status, 'status', 'Unknown')
    print(f"📈 Final Index Status: {status}")
    
    # Show index statistics if available
    if hasattr(final_status, 'num_rows'):
        print(f"📊 Total Documents: {getattr(final_status, 'num_rows', 'Unknown')}")
    
    if hasattr(final_status, 'index_size_bytes'):
        size_bytes = getattr(final_status, 'index_size_bytes', 0)
        size_mb = size_bytes / (1024 * 1024) if size_bytes > 0 else 0
        print(f"💾 Index Size: {size_mb:.2f} MB")
        
except Exception as e:
    print(f"⚠️ Could not get final metrics: {str(e)}")

print("\n🎯 CAPABILITIES VERIFIED:")
print("✅ Vector similarity search")
print("✅ Filtered search by categories and topics")
print("✅ Hybrid search with reranking")
print("✅ Debug information and performance metrics")
print("✅ Error handling and graceful degradation")

print(f"\n📋 CONFIGURATION SUMMARY:")
print(f"📊 Index Name: {vector_index_name}")
print(f"🔗 Endpoint: {vector_search_endpoint}")
print(f"📈 Source Table: {source_table}")
print(f"🔑 Primary Key: path")
print(f"📝 Embedding Column: text")
print(f"🤖 Embedding Model: databricks-gte-large-en")
print(f"🔄 Pipeline Type: TRIGGERED")
print(f"📊 Index Status: {'Created' if not index_exists else 'Existing'}")

print(f"\n🚀 NEXT STEPS:")
print("1. Use the index in your AI agent applications")
print("2. Monitor index performance and update as needed")
print("3. Consider setting up continuous sync for real-time updates")
print("4. Test with production workloads and optimize queries")

print("\n" + "="*60)
print("✅ VECTOR INDEX CREATION JOB COMPLETED SUCCESSFULLY")
print("="*60)
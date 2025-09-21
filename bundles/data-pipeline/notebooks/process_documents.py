# Databricks notebook source
# MAGIC %md
# MAGIC # Document Processing Job
# MAGIC 
# MAGIC This notebook processes documents from a volume path using Databricks' ai_parse_document function.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# Get parameters from job configuration
source_volume_path = dbutils.widgets.get("sourceVolumePath")
destination_table_name = dbutils.widgets.get("destinationTableName")
limit = dbutils.widgets.get("limit")
partition_count = dbutils.widgets.get("partitionCount")

print(f"Source volume path: {source_volume_path}")
print(f"Destination table: {destination_table_name}")
print(f"Limit: {limit}")
print(f"Partition count: {partition_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Process Documents with AI Parse

# COMMAND ----------


# Define parse extensions
parse_extensions = ['.pdf', '.jpg', '.jpeg', '.png']

# Create the SQL query for document processing with hardcoded parse extensions
sql_query = f"""
CREATE or REPLACE TABLE IDENTIFIER('{destination_table_name}') 
TBLPROPERTIES (delta.enableChangeDataFeed = true) AS (
  -- Parse documents with ai_parse
  WITH all_files AS (
    SELECT
      path,
      content
    FROM
      READ_FILES('{source_volume_path}', format => 'binaryFile')
    ORDER BY
      path ASC
    LIMIT INT({limit})
  ),
  repartitioned_files AS (
    SELECT *
    FROM all_files
    -- Force Spark to split into partitions
    DISTRIBUTE BY crc32(path) % INT({partition_count})
  ),
  -- Parse the files using ai_parse document
  parsed_documents AS (
    SELECT
      path,
      ai_parse_document(content) as parsed
    FROM
      repartitioned_files
    WHERE array_contains(array('.pdf', '.jpg', '.jpeg', '.png'), lower(regexp_extract(path, r'(\\.[^.]+)$', 1)))
  ),
  raw_documents AS (
    SELECT
      path,
      decode(content, "utf-8") as text
    FROM 
      repartitioned_files
    WHERE NOT array_contains(array('.pdf', '.jpg', '.jpeg', '.png'), lower(regexp_extract(path, r'(\\.[^.]+)$', 1)))
  ),
  -- Extract page markdowns from ai_parse output
  sorted_page_contents AS (
    SELECT
      path,
      page:content AS content
    FROM
      (
        SELECT
          path,
          posexplode(try_cast(parsed:document:pages AS ARRAY<VARIANT>)) AS (page_idx, page)
        FROM
          parsed_documents
        WHERE
          parsed:document:pages IS NOT NULL
          AND CAST(parsed:error_status AS STRING) IS NULL
      )
    ORDER BY
      page_idx
  ),
  -- Concatenate so we have 1 row per document
  concatenated AS (
      SELECT
          path,
          concat_ws('

', collect_list(content)) AS full_content
      FROM
          sorted_page_contents
      GROUP BY
          path
  ),
  -- Combine parsed and raw documents without the problematic raw_parsed column
  with_raw AS (
      SELECT
          a.path,
          a.full_content as text,
          regexp_extract(a.path, r'/([^/]+)$', 1) as filename,
          CASE 
              WHEN lower(a.path) LIKE '%oracle%' THEN 'oracle-migration'
              ELSE 'general'
          END as categories,
          'data platform migration' as topics
      FROM concatenated a
  )
  -- Recombine raw text documents with parsed documents
  SELECT 
      path,
      text,
      filename,
      categories,
      topics
  FROM with_raw
  UNION ALL 
  SELECT 
      path,
      text,
      regexp_extract(path, r'/([^/]+)$', 1) as filename,
      CASE 
          WHEN lower(path) LIKE '%oracle%' THEN 'oracle-migration'
          ELSE 'general'
      END as categories,
      'migration to databricks' as topics
  FROM raw_documents
);
"""

# Execute the SQL query
print("Executing document processing query...")
spark.sql(sql_query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Display Sample Results

# COMMAND ----------

# Display a sample from the table
sample_query = f"""
SELECT
    path,
    filename,
    categories,
    topics,
    length(text) as text_length,
    CASE 
        WHEN raw_parsed IS NOT NULL THEN 'AI Parsed'
        ELSE 'Raw Text'
    END as parse_type
FROM IDENTIFIER('{destination_table_name}')
LIMIT 20
"""

print("Sample results from processed documents:")
display(spark.sql(sample_query))

# Show column information
print("\nTable schema:")
spark.sql(f"DESCRIBE {destination_table_name}").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Job Completion


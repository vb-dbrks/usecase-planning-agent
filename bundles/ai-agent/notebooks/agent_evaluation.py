# Databricks notebook source
# MAGIC %md
# MAGIC # Agent Evaluation
# MAGIC 
# MAGIC This notebook evaluates the performance of the Migration Planning Agent using various metrics and test cases.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# Install required packages
%pip install databricks-vectorsearch databricks-genai-inference

# Restart Python to ensure packages are loaded
dbutils.library.restartPython()

# COMMAND ----------

# Import required libraries
from databricks.vector_search.client import VectorSearchClient
from databricks_genai_inference import ChatCompletion
import json
import pandas as pd
import time
from typing import List, Dict, Any
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Get Parameters

# COMMAND ----------

# Get parameters from job configuration
vector_search_endpoint = dbutils.widgets.get("vector_search_endpoint")
vector_index_name = dbutils.widgets.get("vector_index_name")
migration_documents_table = dbutils.widgets.get("migration_documents_table")
agent_model = dbutils.widgets.get("agent_model")
temperature = float(dbutils.widgets.get("temperature"))
max_tokens = int(dbutils.widgets.get("max_tokens"))

print(f"Vector search endpoint: {vector_search_endpoint}")
print(f"Vector index name: {vector_index_name}")
print(f"Migration documents table: {migration_documents_table}")
print(f"Agent model: {agent_model}")
print(f"Temperature: {temperature}")
print(f"Max tokens: {max_tokens}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initialize Vector Search Client

# COMMAND ----------

# Initialize the Vector Search client
client = VectorSearchClient()

# Get the index for searching
index = client.get_index(
    endpoint_name=vector_search_endpoint,
    index_name=vector_index_name
)

print(f"Vector search index loaded: {vector_index_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluation Framework

# COMMAND ----------

class AgentEvaluator:
    def __init__(self, vector_index, model_name, temperature=0.1, max_tokens=2048):
        self.vector_index = vector_index
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def search_relevant_documents(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search for relevant documents using vector search"""
        try:
            results = self.vector_index.similarity_search(
                query_text=query,
                columns=["path", "text", "filename"],
                num_results=num_results
            )
            
            documents = []
            if results and 'result' in results and 'data_array' in results['result']:
                for doc in results['result']['data_array']:
                    documents.append({
                        'path': doc.get('path', ''),
                        'text': doc.get('text', ''),
                        'filename': doc.get('filename', ''),
                        'score': doc.get('score', 0.0)
                    })
            
            return documents
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []
    
    def generate_response(self, query: str, context_documents: List[Dict]) -> str:
        """Generate a response using the LLM with context documents"""
        
        # Prepare context from documents
        context_text = "\n\n".join([
            f"Document: {doc.get('filename', 'Unknown')}\nContent: {doc.get('text', '')[:1000]}..."
            for doc in context_documents
        ])
        
        # Create the prompt
        system_prompt = """You are an expert migration planning agent specializing in data platform migrations, particularly Oracle to Databricks migrations. 
        
Your role is to provide comprehensive, actionable migration plans based on the provided context documents and user requirements.

Key responsibilities:
1. Analyze the current system architecture and requirements
2. Identify migration challenges and risks
3. Provide step-by-step migration strategies
4. Recommend best practices and tools
5. Suggest timeline and resource planning
6. Address security, governance, and compliance considerations

Always provide specific, actionable recommendations with clear next steps."""

        user_prompt = f"""Based on the following context documents about migration planning, please provide a comprehensive migration plan for:

**User Query:** {query}

**Context Documents:**
{context_text}

Please provide:
1. **Executive Summary** - High-level overview of the migration approach
2. **Current State Analysis** - Assessment of existing systems
3. **Migration Strategy** - Detailed step-by-step approach
4. **Technical Recommendations** - Specific tools, patterns, and best practices
5. **Risk Assessment** - Potential challenges and mitigation strategies
6. **Timeline & Resources** - Estimated timeline and resource requirements
7. **Next Steps** - Immediate actionable items

Format your response in a clear, structured manner with headings and bullet points where appropriate."""

        try:
            response = ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def evaluate_response_quality(self, query: str, response: str) -> Dict[str, float]:
        """Evaluate the quality of the response using various metrics"""
        
        # Basic metrics
        word_count = len(response.split())
        char_count = len(response)
        
        # Check for key sections
        key_sections = [
            "executive summary", "current state", "migration strategy", 
            "technical recommendations", "risk assessment", "timeline", "next steps"
        ]
        
        sections_found = sum(1 for section in key_sections if section.lower() in response.lower())
        section_coverage = sections_found / len(key_sections)
        
        # Check for actionable items (bullet points, numbered lists)
        actionable_indicators = ["•", "-", "1.", "2.", "3.", "step", "action", "recommend"]
        actionable_count = sum(1 for indicator in actionable_indicators if indicator.lower() in response.lower())
        
        # Check for technical terms
        technical_terms = ["databricks", "oracle", "migration", "etl", "data", "warehouse", "lakehouse"]
        technical_count = sum(1 for term in technical_terms if term.lower() in response.lower())
        
        # Calculate scores
        completeness_score = min(section_coverage * 1.2, 1.0)  # Cap at 1.0
        actionability_score = min(actionable_count / 10, 1.0)  # Cap at 1.0
        technical_relevance_score = min(technical_count / len(technical_terms), 1.0)
        
        # Overall quality score (weighted average)
        overall_score = (
            completeness_score * 0.4 +
            actionability_score * 0.3 +
            technical_relevance_score * 0.3
        )
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sections_found': sections_found,
            'section_coverage': section_coverage,
            'actionable_count': actionable_count,
            'technical_count': technical_count,
            'completeness_score': completeness_score,
            'actionability_score': actionability_score,
            'technical_relevance_score': technical_relevance_score,
            'overall_score': overall_score
        }
    
    def evaluate_query(self, query: str) -> Dict[str, Any]:
        """Evaluate a single query and return comprehensive metrics"""
        start_time = time.time()
        
        # Search for relevant documents
        relevant_docs = self.search_relevant_documents(query, num_results=5)
        
        # Generate response
        response = self.generate_response(query, relevant_docs)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Evaluate response quality
        quality_metrics = self.evaluate_response_quality(query, response)
        
        return {
            'query': query,
            'response': response,
            'response_time': response_time,
            'num_documents_used': len(relevant_docs),
            'quality_metrics': quality_metrics,
            'metadata': {
                'model_used': self.model_name,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Initialize Evaluator

# COMMAND ----------

# Initialize the evaluator
evaluator = AgentEvaluator(
    vector_index=index,
    model_name=agent_model,
    temperature=temperature,
    max_tokens=max_tokens
)

print("Agent Evaluator initialized successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test Cases

# COMMAND ----------

# Define comprehensive test cases for evaluation
test_cases = [
    {
        'query': 'How do I migrate my Oracle data warehouse to Databricks?',
        'category': 'general_migration',
        'expected_sections': ['executive summary', 'migration strategy', 'technical recommendations']
    },
    {
        'query': 'What are the best practices for migrating ETL processes from Oracle to Databricks?',
        'category': 'etl_migration',
        'expected_sections': ['technical recommendations', 'migration strategy', 'best practices']
    },
    {
        'query': 'How should I handle data security and governance during migration?',
        'category': 'security_governance',
        'expected_sections': ['risk assessment', 'security', 'governance', 'compliance']
    },
    {
        'query': 'What tools and technologies should I use for Oracle to Databricks migration?',
        'category': 'tools_technology',
        'expected_sections': ['technical recommendations', 'tools', 'technologies']
    },
    {
        'query': 'How do I estimate the timeline and resources needed for migration?',
        'category': 'planning',
        'expected_sections': ['timeline', 'resources', 'planning', 'estimation']
    },
    {
        'query': 'What are the common challenges in Oracle to Databricks migration?',
        'category': 'challenges',
        'expected_sections': ['risk assessment', 'challenges', 'mitigation']
    },
    {
        'query': 'How do I migrate my Oracle stored procedures to Databricks?',
        'category': 'stored_procedures',
        'expected_sections': ['technical recommendations', 'migration strategy', 'stored procedures']
    },
    {
        'query': 'What is the recommended approach for data validation during migration?',
        'category': 'data_validation',
        'expected_sections': ['data validation', 'testing', 'quality assurance']
    }
]

print(f"Defined {len(test_cases)} test cases for evaluation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Run Evaluation

# COMMAND ----------

print("Starting comprehensive evaluation...")
print("="*80)

evaluation_results = []

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'='*20} TEST CASE {i}/{len(test_cases)} {'='*20}")
    print(f"Query: {test_case['query']}")
    print(f"Category: {test_case['category']}")
    print("-" * 60)
    
    # Evaluate the query
    result = evaluator.evaluate_query(test_case['query'])
    
    # Add test case metadata
    result['test_case'] = test_case
    evaluation_results.append(result)
    
    # Display results
    print(f"Response Time: {result['response_time']:.2f} seconds")
    print(f"Documents Used: {result['num_documents_used']}")
    print(f"Overall Score: {result['quality_metrics']['overall_score']:.3f}")
    print(f"Completeness: {result['quality_metrics']['completeness_score']:.3f}")
    print(f"Actionability: {result['quality_metrics']['actionability_score']:.3f}")
    print(f"Technical Relevance: {result['quality_metrics']['technical_relevance_score']:.3f}")
    
    print("\nResponse Preview:")
    print(result['response'][:300] + "..." if len(result['response']) > 300 else result['response'])
    print("\n" + "="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Analysis and Reporting

# COMMAND ----------

# Convert results to DataFrame for analysis
results_df = pd.DataFrame([
    {
        'query': result['query'],
        'category': result['test_case']['category'],
        'response_time': result['response_time'],
        'num_documents': result['num_documents_used'],
        'overall_score': result['quality_metrics']['overall_score'],
        'completeness_score': result['quality_metrics']['completeness_score'],
        'actionability_score': result['quality_metrics']['actionability_score'],
        'technical_relevance_score': result['quality_metrics']['technical_relevance_score'],
        'word_count': result['quality_metrics']['word_count'],
        'sections_found': result['quality_metrics']['sections_found']
    }
    for result in evaluation_results
])

print("Evaluation Results Summary:")
print("="*50)
display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Performance Metrics

# COMMAND ----------

# Calculate overall performance metrics
overall_metrics = {
    'total_queries': len(evaluation_results),
    'avg_response_time': np.mean([r['response_time'] for r in evaluation_results]),
    'avg_overall_score': np.mean([r['quality_metrics']['overall_score'] for r in evaluation_results]),
    'avg_completeness_score': np.mean([r['quality_metrics']['completeness_score'] for r in evaluation_results]),
    'avg_actionability_score': np.mean([r['quality_metrics']['actionability_score'] for r in evaluation_results]),
    'avg_technical_relevance_score': np.mean([r['quality_metrics']['technical_relevance_score'] for r in evaluation_results]),
    'avg_documents_used': np.mean([r['num_documents_used'] for r in evaluation_results]),
    'avg_word_count': np.mean([r['quality_metrics']['word_count'] for r in evaluation_results])
}

print("Overall Performance Metrics:")
print("="*40)
for metric, value in overall_metrics.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.3f}")
    else:
        print(f"{metric}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Category-wise Analysis

# COMMAND ----------

# Analyze performance by category
category_analysis = results_df.groupby('category').agg({
    'overall_score': 'mean',
    'response_time': 'mean',
    'completeness_score': 'mean',
    'actionability_score': 'mean',
    'technical_relevance_score': 'mean'
}).round(3)

print("Performance by Category:")
print("="*30)
display(category_analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Detailed Results

# COMMAND ----------

# Display detailed results for each test case
print("Detailed Evaluation Results:")
print("="*50)

for i, result in enumerate(evaluation_results, 1):
    print(f"\n{'='*20} DETAILED RESULT {i} {'='*20}")
    print(f"Query: {result['query']}")
    print(f"Category: {result['test_case']['category']}")
    print(f"Response Time: {result['response_time']:.2f} seconds")
    print(f"Documents Used: {result['num_documents_used']}")
    print(f"Word Count: {result['quality_metrics']['word_count']}")
    print(f"Sections Found: {result['quality_metrics']['sections_found']}")
    print(f"Overall Score: {result['quality_metrics']['overall_score']:.3f}")
    print(f"Completeness: {result['quality_metrics']['completeness_score']:.3f}")
    print(f"Actionability: {result['quality_metrics']['actionability_score']:.3f}")
    print(f"Technical Relevance: {result['quality_metrics']['technical_relevance_score']:.3f}")
    
    print("\nFull Response:")
    print("-" * 40)
    print(result['response'])
    print("\n" + "="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Job Completion

# COMMAND ----------

print("="*60)
print("✅ AGENT EVALUATION JOB COMPLETED SUCCESSFULLY")
print("="*60)
print(f"🤖 Agent Model: {agent_model}")
print(f"🔍 Vector Index: {vector_index_name}")
print(f"📊 Test Cases Evaluated: {len(test_cases)}")
print(f"⏱️ Average Response Time: {overall_metrics['avg_response_time']:.2f} seconds")
print(f"📈 Average Overall Score: {overall_metrics['avg_overall_score']:.3f}")
print(f"📝 Average Word Count: {overall_metrics['avg_word_count']:.0f}")
print(f"🔧 Temperature: {temperature}")
print(f"📝 Max Tokens: {max_tokens}")
print("="*60)

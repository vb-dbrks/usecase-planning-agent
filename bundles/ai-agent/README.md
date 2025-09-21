# AI Agent Bundle

This bundle contains AI agent-related jobs for migration planning using Databricks Vector Search and LLM capabilities.

## Overview

The AI Agent bundle provides intelligent migration planning capabilities by combining:
- **Vector Search** for document retrieval and context
- **Large Language Models** for generating comprehensive migration plans
- **Evaluation Framework** for assessing agent performance

## Jobs

### 1. Migration Planning Agent Job (`migration_planning_agent_job`)
- **Purpose**: Main AI agent for generating migration plans
- **Notebook**: `notebooks/migration_planning_agent.py`
- **Features**:
  - Vector search for relevant documents
  - LLM-powered migration plan generation
  - Interactive query interface
  - Comprehensive test cases

### 2. Agent Evaluation Job (`agent_evaluation_job`)
- **Purpose**: Evaluate agent performance and quality
- **Notebook**: `notebooks/agent_evaluation.py`
- **Features**:
  - Comprehensive evaluation metrics
  - Performance analysis by category
  - Response quality assessment
  - Detailed reporting

## Configuration

### Variables
- `catalog_name`: Catalog name for data storage
- `schema_name`: Schema name for data organization
- `vector_search_endpoint`: Vector search endpoint name
- `vector_index_name`: Name of the vector index
- `migration_documents_table`: Table containing migration documents
- `agent_model`: AI model to use (default: databricks-dbrx-instruct)
- `temperature`: Model temperature for response generation
- `max_tokens`: Maximum tokens for responses

### Targets
- **default**: Production configuration
- **dev**: Development configuration with reduced limits
- **prod**: Production configuration with higher limits

## Usage

### Deploy the Bundle
```bash
# Deploy to development
databricks bundle deploy --target dev --profile dev

# Deploy to production
databricks bundle deploy --target prod --profile prod
```

### Run Jobs
```bash
# Run migration planning agent
databricks jobs run-now --job-id <migration_planning_agent_job_id>

# Run agent evaluation
databricks jobs run-now --job-id <agent_evaluation_job_id>
```

## Features

### Migration Planning Agent
- **Document Search**: Uses vector search to find relevant migration documents
- **Plan Generation**: Creates comprehensive migration plans with:
  - Executive Summary
  - Current State Analysis
  - Migration Strategy
  - Technical Recommendations
  - Risk Assessment
  - Timeline & Resources
  - Next Steps
- **Interactive Mode**: Supports real-time querying
- **Test Cases**: Pre-defined test scenarios for validation

### Agent Evaluation
- **Quality Metrics**: Comprehensive evaluation of response quality
- **Performance Analysis**: Response time and efficiency metrics
- **Category Analysis**: Performance breakdown by query category
- **Detailed Reporting**: Full evaluation results and recommendations

## Dependencies

- `databricks-vectorsearch`: Vector search capabilities
- `databricks-genai-inference`: LLM inference
- `pandas`: Data manipulation
- `numpy`: Numerical operations

## Architecture

```
AI Agent Bundle
├── databricks.yml          # Bundle configuration
├── notebooks/
│   ├── migration_planning_agent.py  # Main agent notebook
│   └── agent_evaluation.py         # Evaluation notebook
└── README.md               # This file
```

## Integration

This bundle integrates with the `data-pipeline` bundle:
- Uses the same vector search endpoint
- References the same migration documents table
- Shares the same catalog and schema structure

## Monitoring

The evaluation job provides comprehensive monitoring capabilities:
- Response quality scores
- Performance metrics
- Category-wise analysis
- Detailed logging and reporting

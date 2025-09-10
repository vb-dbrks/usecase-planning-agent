# Usecase Delivery Agent

A comprehensive Databricks Asset Bundle for document processing and AI-powered delivery planning. This project combines document parsing, vector indexing, and intelligent project planning using DSPy and Databricks AI services.

## 🚀 Features

### 1. Document Processing Pipeline
- **AI-Powered PDF Parsing**: Uses Databricks `ai_parse_document` function for intelligent document processing
- **Vector Indexing**: Creates searchable vector embeddings for RAG applications
- **Flexible Chunking**: Optional text chunking for large context window models (272k+ tokens)
- **Unity Catalog Integration**: Seamlessly works with Unity Catalog volumes and managed tables

### 2. Delivery Planning Agent
- **DSPy-Powered RAG**: Intelligent question generation and document retrieval
- **Structured Planning**: Covers Resource, Scope, Customer Background, and Process Maturity
- **Vector Search Integration**: Retrieves relevant information from indexed documents
- **Risk Assessment**: Identifies potential risks and mitigation strategies
- **Interactive Interface**: User-friendly planning session management

## 📁 Project Structure

```
use-case-delivery-agent/
├── notebooks/
│   ├── ai_parse_processor.ipynb          # Document processing and vector indexing
│   ├── check_table.ipynb                 # Table verification and status checking
│   └── usecase_delivery_planning_agent.ipynb  # Main planning agent
├── databricks.yml                        # Databricks Asset Bundle configuration
└── README.md                            # This file
```

## 🛠️ Setup and Configuration

### Prerequisites
- Databricks workspace with Unity Catalog enabled
- Vector Search endpoint configured
- Databricks CLI configured with appropriate permissions

### Environment Variables
```bash
# Set your Databricks profile
export DATABRICKS_PROFILE=dev

# Set your workspace URL (if not using default)
export DATABRICKS_HOST=https://your-workspace.databricks.com
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd use-case-delivery-agent

# Deploy the Databricks Asset Bundle
databricks bundle deploy --profile dev --target development
```

## 📊 Data Flow

### Document Processing Pipeline
1. **Input**: PDF documents stored in Unity Catalog volume
2. **Processing**: AI parse documents using `ai_parse_document` function
3. **Chunking**: Optional text chunking based on configuration
4. **Storage**: Store parsed content in Delta tables
5. **Indexing**: Create vector embeddings for search

### Delivery Planning Agent
1. **Context Gathering**: Collect project context and requirements
2. **Question Generation**: Generate relevant questions by category
3. **Document Retrieval**: Search indexed documents for relevant information
4. **Analysis**: Extract insights from answers and documents
5. **Planning**: Generate comprehensive project plans
6. **Risk Assessment**: Identify risks and mitigation strategies

## 🎯 Usage

### Document Processing
```bash
# Process documents and create vector index
databricks bundle run ai_parse_processor_job --profile dev --target development

# Check processing status
databricks bundle run check_table_job --profile dev --target development
```

### Delivery Planning Agent
```python
# Initialize the planning agent
agent = DeliveryPlanningAgent("your-vector-search-endpoint")

# Start a planning session
agent.start_planning_session(
    project_context="Your project description",
    timeline_requirements="6 months deadline"
)

# Answer questions interactively
agent.answer_question(question, answer, category)

# Generate final project plan
final_plan = agent.generate_project_plan()
```

## 📋 Planning Categories

The delivery planning agent covers comprehensive project planning across four key categories:

### Resource Planning
- Team size and composition
- Skills assessment and training needs
- Resource availability and allocation
- Ownership and accountability

### Scope Definition
- Data volume and complexity
- Pipeline migration approach
- Testing and validation requirements
- Monitoring and optimization needs

### Customer Background
- Timeline constraints and deadlines
- Cloud and platform experience
- Security and compliance requirements
- Business drivers and objectives

### Process Maturity
- Change management capabilities
- Project methodology preferences
- Historical performance patterns
- Risk tolerance and mitigation

## 🔧 Configuration

### Document Processing Settings
```python
# Configuration in ai_parse_processor.ipynb
SOURCE_VOLUME_PATH = "/Volumes/vbdemos/dbdemos_autoloader/raw_data/usecase-planning-agent-pdf/"
DESTINATION_TABLE_NAME = "vbdemos.usecase_agent.usecase_planning_agent_pdf_parsed"
CHUNKS_TABLE_NAME = "vbdemos.usecase_agent.pdf_chunks"
ENABLE_CHUNKING = False  # Set to True for chunking
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks
```

### Planning Agent Settings
```python
# Configuration in usecase_delivery_planning_agent.ipynb
VECTOR_SEARCH_ENDPOINT_NAME = "usecase-planning-agent-index"
CATALOG_NAME = "vbdemos"
SCHEMA_NAME = "usecase_agent"
DOCUMENTS_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.pdf_chunks"
```

## 🚀 Deployment

### Development Environment
```bash
# Deploy to development
databricks bundle deploy --profile dev --target development

# Run all jobs
databricks bundle run --profile dev --target development
```

### Production Environment
```bash
# Deploy to production
databricks bundle deploy --profile dev --target production

# Run specific job
databricks bundle run ai_parse_processor_job --profile dev --target production
```

## 📈 Monitoring and Maintenance

### Job Monitoring
- Monitor job runs in Databricks Jobs UI
- Check logs for processing status
- Verify table creation and data quality

### Vector Search Maintenance
- Monitor vector search endpoint performance
- Update embeddings when new documents are added
- Optimize search parameters based on usage patterns

## 🔍 Troubleshooting

### Common Issues
1. **Vector Search Endpoint Not Found**: Ensure the endpoint is created and accessible
2. **Document Processing Failures**: Check Unity Catalog permissions and file paths
3. **Planning Agent Errors**: Verify DSPy configuration and model access

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

**Last Updated**: December 2024
**Version**: 1.0.0
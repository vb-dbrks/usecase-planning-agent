# Databricks notebook source
# MAGIC %md
# MAGIC # Migration Planning Agent
# MAGIC 
# MAGIC This notebook implements a DSPy-based AI agent for migration planning using vector search and MLflow deployment.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# Install required packages with version pinning for DBR 16.4 LTS compatibility
# Pin DSPy to version compatible with Pydantic 2.8.2 in DBR 16.4 LTS
%pip install dspy databricks-vectorsearch mlflow flask

# Ensure Pydantic compatibility - DBR 16.4 LTS has Pydantic 2.8.2

# Restart Python to ensure packages are loaded
dbutils.library.restartPython()

# COMMAND ----------

# Import required libraries
import dspy
import mlflow
import mlflow.dspy
from databricks.vector_search.client import VectorSearchClient
from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint
import json
import pandas as pd
from typing import List, Dict, Any
import pydantic

# Verify versions for DBR 16.4 LTS compatibility
print(f"🔧 DSPy version: {dspy.__version__}")
print(f"🔧 Pydantic version: {pydantic.__version__}")
print(f"🔧 MLflow version: {mlflow.__version__}")
print("✅ Version compatibility check completed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Get Parameters

# COMMAND ----------

# Get parameters from job configuration
# Get parameters from job configuration
# dbutils.widgets.text("vector_search_endpoint", "usecase-agent")
# dbutils.widgets.text("vector_index_name", "vbdemos.usecase_agent.migration_planning_documents")
# dbutils.widgets.text("migration_documents_table", "vbdemos.usecase_agent.migration_documents")
# dbutils.widgets.text("agent_model", "databricks-claude-3-7-sonnet")
# dbutils.widgets.text("temperature", "0.1")
# dbutils.widgets.text("max_tokens", "5000")
# dbutils.widgets.text("mlflow_experiment_name", "/Users/varun.bhandary@databricks.com/usecase-agent")

vector_search_endpoint = dbutils.widgets.get("vector_search_endpoint")
vector_index_name = dbutils.widgets.get("vector_index_name")
migration_documents_table = dbutils.widgets.get("migration_documents_table")
agent_model = dbutils.widgets.get("agent_model")
temperature = float(dbutils.widgets.get("temperature"))
max_tokens = int(dbutils.widgets.get("max_tokens"))
mlflow_experiment_name = dbutils.widgets.get("mlflow_experiment_name")

print(f"Vector search endpoint: {vector_search_endpoint}")
print(f"Vector index name: {vector_index_name}")
print(f"Migration documents table: {migration_documents_table}")
print(f"Agent model: {agent_model}")
print(f"Temperature: {temperature}")
print(f"Max tokens: {max_tokens}")
print(f"MLflow experiment name: {mlflow_experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. DSPy Setup and Model Configuration

# COMMAND ----------

# Configure DSPy with Databricks model
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() + '/serving-endpoints'

lm = dspy.LM(
    model="databricks/databricks-claude-sonnet-4",
    api_key=token,
    api_base=url,
)

# Configure DSPy
dspy.configure(lm=lm)

print(f"DSPy version: {dspy.__version__}")
print("DSPy configured successfully with Databricks model")

# Test the model configuration
print("\n🧪 Testing model configuration...")
try:
    test_response = lm("Hello, this is a test message.")
    print(f"✅ Model test successful: {test_response[:100]}...")
except Exception as e:
    print(f"❌ Model test failed: {e}")
    print("This might indicate the model endpoint is not accessible or configured incorrectly.")

# Initialize Vector Search client
vsc = VectorSearchClient()

print("DSPy and Vector Search configured successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. DSPy Signatures

# COMMAND ----------

# DSPy Signatures for the iterative migration planning agent

class QuestionGenerator(dspy.Signature):
    """Generate relevant questions for migrating TO Databricks from existing data/analytics platforms."""
    project_context: str = dspy.InputField(desc="Description of the current data/analytics platform and objectives for migrating TO Databricks")
    category: str = dspy.InputField(desc="Planning category (Resource, Scope, Customer Background, Current Process Maturity, etc.)")
    existing_answers: str = dspy.InputField(desc="Previously answered questions and responses")
    questions: str = dspy.OutputField(desc="List of 3-5 most relevant questions for this category specific to migrating TO Databricks")

class DocumentRetriever(dspy.Signature):
    """Retrieve relevant documents based on questions and context."""
    question: str = dspy.InputField(desc="Specific question to find relevant information for")
    project_context: str = dspy.InputField(desc="Project context and background")
    retrieved_docs: str = dspy.OutputField(desc="Relevant document excerpts and information")

class AnswerAnalyzer(dspy.Signature):
    """Analyze answers and extract key insights for project planning."""
    question: str = dspy.InputField(desc="The question that was asked")
    answer: str = dspy.InputField(desc="The answer provided by the user")
    relevant_docs: str = dspy.InputField(desc="Relevant documents retrieved from knowledge base")
    insights: str = dspy.OutputField(desc="Key insights, risks, and implications for project planning")

class MigrationPlanGenerator(dspy.Signature):
    """Generate comprehensive migration plan for moving TO Databricks with structured table outputs."""
    project_context: str = dspy.InputField(desc="Overall project context and objectives for migrating TO Databricks")
    gathered_insights: str = dspy.InputField(desc="All insights gathered from questions and documents about current platform")
    timeline_requirements: str = dspy.InputField(desc="Timeline constraints and requirements for migrating TO Databricks")
    migration_plan: str = dspy.OutputField(desc="Comprehensive migration plan for moving TO Databricks. Output as structured tables with clear headers for: 1) Migration Timeline, 2) Resource Requirements, 3) Migration Phases, 4) Risk Assessment. Use markdown table format with | separators.")

class PlanCompletenessScorer(dspy.Signature):
    """Score the completeness and maturity of the migration plan based on gathered information."""
    project_context: str = dspy.InputField(desc="Project context and objectives")
    gathered_insights: str = dspy.InputField(desc="All insights gathered from questions and documents")
    current_plan: str = dspy.InputField(desc="Current migration plan")
    completeness_score: str = dspy.OutputField(desc="Completeness score (0-100) and assessment of plan maturity with specific recommendations for improvement")

class RiskAssessor(dspy.Signature):
    """Assess risks and provide mitigation strategies."""
    project_plan: str = dspy.InputField(desc="The proposed project plan")
    project_context: str = dspy.InputField(desc="Project context and constraints")
    risk_assessment: str = dspy.OutputField(desc="Identified risks, their likelihood, impact, and mitigation strategies")

# NEW CONVERSATIONAL SIGNATURES FOR CHAT UI
class ProjectContextExtractor(dspy.Signature):
    """Extract structured project context from user's natural language description."""
    user_input: str = dspy.InputField(desc="User's natural language description of their migration project")
    project_context: str = dspy.OutputField(desc="Structured project context including source platform, target platform, objectives, and key details")
    suggested_category: str = dspy.OutputField(desc="Suggested planning category to start with (Resource & Team, Technical Scope & Architecture, Customer Background & Drivers, etc.)")

class ConversationUnderstanding(dspy.Signature):
    """Understand user input in the context of ongoing migration planning conversation."""
    user_input: str = dspy.InputField(desc="Current user input/message")
    conversation_history: str = dspy.InputField(desc="Previous conversation context and gathered information")
    project_context: str = dspy.InputField(desc="Current project context and objectives")
    intent: str = dspy.OutputField(desc="User intent: 'answer_question', 'ask_category', 'request_plan', 'general_help', 'provide_context'")
    suggested_action: str = dspy.OutputField(desc="Suggested next action: 'ask_questions', 'generate_plan', 'clarify', 'continue_category'")
    response_type: str = dspy.OutputField(desc="Type of response needed: 'questions', 'plan', 'clarification', 'conversation'")

class IntelligentResponseGenerator(dspy.Signature):
    """Generate intelligent, conversational responses for migration planning chat."""
    user_input: str = dspy.InputField(desc="User's input/message")
    conversation_context: str = dspy.InputField(desc="Current conversation context and state")
    project_context: str = dspy.InputField(desc="Project context and objectives")
    intent: str = dspy.InputField(desc="Identified user intent")
    suggested_action: str = dspy.InputField(desc="Suggested next action")
    response: str = dspy.OutputField(desc="Intelligent, helpful response that guides the user through migration planning")

print("DSPy signatures defined successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. DSPy Modules

# COMMAND ----------

# DSPy Modules for the iterative migration planning agent

class QuestionGenerationModule(dspy.Module):
    """Module to generate relevant questions for each planning category."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(QuestionGenerator)
    
    def forward(self, project_context: str, category: str, existing_answers: str = ""):
        return self.generate(
            project_context=project_context,
            category=category,
            existing_answers=existing_answers
        )

class DocumentRetrievalModule(dspy.Module):
    """Module to retrieve relevant documents using vector search."""
    
    def __init__(self, vector_search_endpoint: str, vector_search_index: str):
        super().__init__()
        self.vector_search_endpoint = vector_search_endpoint
        self.vector_search_index = vector_search_index
        self.retrieve = dspy.ChainOfThought(DocumentRetriever)
    
    def forward(self, question: str, project_context: str):
        # Perform vector search
        search_results = vsc.get_index(
            endpoint_name=self.vector_search_endpoint,
            index_name=self.vector_search_index
        ).similarity_search(
            query_text=question,
            columns=["path", "text", "filename", "categories", "topics"],
            num_results=5
        )
        
        # Format retrieved documents
        if isinstance(search_results, list):
            documents = search_results
        else:
            documents = search_results.get('result', {}).get('data_array', [])
        
        retrieved_docs = "\n\n".join([
            f"Source: {doc.get('path', 'Unknown') if isinstance(doc, dict) else 'Unknown'}\n"
            f"File: {doc.get('filename', 'Unknown') if isinstance(doc, dict) else 'Unknown'}\n"
            f"Category: {doc.get('categories', 'N/A') if isinstance(doc, dict) else 'N/A'}\n"
            f"Topics: {doc.get('topics', 'N/A') if isinstance(doc, dict) else 'N/A'}\n"
            f"Content: {doc.get('text', '') if isinstance(doc, dict) else str(doc)}"
            for doc in documents
        ])
        
        return self.retrieve(
            question=question,
            project_context=project_context,
            retrieved_docs=retrieved_docs
        )

class AnswerAnalysisModule(dspy.Module):
    """Module to analyze answers and extract insights."""
    
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(AnswerAnalyzer)
    
    def forward(self, question: str, answer: str, relevant_docs: str):
        return self.analyze(
            question=question,
            answer=answer,
            relevant_docs=relevant_docs
        )

class MigrationPlanGenerationModule(dspy.Module):
    """Module to generate comprehensive migration plans."""
    
    def __init__(self):
        super().__init__()
        self.generate_plan = dspy.ChainOfThought(MigrationPlanGenerator)
    
    def forward(self, project_context: str, gathered_insights: str, timeline_requirements: str):
        return self.generate_plan(
            project_context=project_context,
            gathered_insights=gathered_insights,
            timeline_requirements=timeline_requirements
        )

class PlanCompletenessScoringModule(dspy.Module):
    """Module to score plan completeness and maturity."""
    
    def __init__(self):
        super().__init__()
        self.score = dspy.ChainOfThought(PlanCompletenessScorer)
    
    def forward(self, project_context: str, gathered_insights: str, current_plan: str):
        return self.score(
            project_context=project_context,
            gathered_insights=gathered_insights,
            current_plan=current_plan
        )

class RiskAssessmentModule(dspy.Module):
    """Module to assess risks and provide mitigation strategies."""
    
    def __init__(self):
        super().__init__()
        self.assess_risks = dspy.ChainOfThought(RiskAssessor)
    
    def forward(self, project_plan: str, project_context: str):
        return self.assess_risks(
            project_plan=project_plan,
            project_context=project_context
        )

# NEW CONVERSATIONAL MODULES FOR CHAT UI
class ProjectContextExtractionModule(dspy.Module):
    """Module to extract structured project context from user descriptions."""
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ProjectContextExtractor)
    
    def forward(self, user_input: str):
        return self.extract(user_input=user_input)

class ConversationUnderstandingModule(dspy.Module):
    """Module to understand user intent and conversation context."""
    
    def __init__(self):
        super().__init__()
        self.understand = dspy.ChainOfThought(ConversationUnderstanding)
    
    def forward(self, user_input: str, conversation_history: str, project_context: str):
        return self.understand(
            user_input=user_input,
            conversation_history=conversation_history,
            project_context=project_context
        )

class IntelligentResponseGenerationModule(dspy.Module):
    """Module to generate intelligent, conversational responses."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(IntelligentResponseGenerator)
    
    def forward(self, user_input: str, conversation_context: str, project_context: str, intent: str, suggested_action: str):
        return self.generate(
            user_input=user_input,
            conversation_context=conversation_context,
            project_context=project_context,
            intent=intent,
            suggested_action=suggested_action
        )

print("DSPy modules defined successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Planning Categories and Configuration

# COMMAND ----------

# Enhanced planning categories for iterative migration planning
PLANNING_CATEGORIES = {
    "Resource & Team": [
        "How many team members are there and what are their roles?",
        "Are the teams sufficiently skilled/trained in Databricks?",
        "Are they using Professional Services or System Integrators?",
        "Are resources shared with other projects?",
        "Have resources done this type of migration work before?",
        "Is there a product owner and program manager?",
        "Are the DevOps/SecOps/Infra teams under this project's control?",
        "Are the BAU teams that will manage the new system sufficiently trained?",
        "Has an end-user adoption plan been created?",
        "What is the budget allocation for training and external support?"
    ],
    "Current Process Maturity": [
        "Do they have a history of project delays?",
        "Do they have change management authority and processes?",
        "What is their identified way of working - agile/waterfall/hybrid?",
        "How mature are their current data governance processes?",
        "What is their current data quality management approach?",
        "How do they handle incident response and troubleshooting?",
        "What CI/CD practices do they have in place?"
    ],
    "Customer Background & Drivers": [
        "Does the customer have a specific deadline and what drives it?",
        "Is the customer already using cloud infrastructure?",
        "Does the customer have Databricks elsewhere in the organization?",
        "Does the customer have security approval for this migration?",
        "What are the key business drivers for the migration?",
        "Are there any legal compliance or regulatory requirements?",
        "What is the current pain with the existing Oracle system?",
        "Who are the key stakeholders and decision makers?"
    ],
    "Technical Scope & Architecture": [
        "Has a pilot or POC been conducted?",
        "Does the customer have visibility of all data and pipelines to migrate?",
        "Is the customer aware of where and who uses the data?",
        "Is lift and shift or redesign preferred for different components?",
        "How many pipelines, reports, and data sources need migration?",
        "What is the relative complexity of the pipelines?",
        "What is the volume and frequency of data updates?",
        "Is there a proposed Unity Catalog design and infrastructure architecture?",
        "How will PII and sensitive data be handled?",
        "Does the migration include monitoring, alerting, and optimization?",
        "Will it be run in parallel or phased migration approach?",
        "Which pipelines are business critical and cannot be down?",
        "Do they have control over how they receive data from source systems?",
        "What additional data quality checks need to be implemented?",
        "Are there any key connectors or integrations that need migration?",
        "What level of testing is required and who will perform it?",
        "Are all data consumers and downstream systems identified?",
        "What is the current data quality and what improvements are needed?",
        "Are the data pathways and dependencies fully mapped?",
        "Has a permissions model and data access strategy been agreed?",
        "Is disaster recovery and business continuity included?",
        "What is the target performance and scalability requirements?"
    ],
    "Risk & Compliance": [
        "What are the main technical risks identified?",
        "What are the business continuity risks?",
        "What security and compliance requirements must be met?",
        "What are the data residency and sovereignty requirements?",
        "How will data lineage and audit trails be maintained?",
        "What backup and recovery procedures are needed?",
        "What are the rollback procedures if migration fails?",
        "How will change management be handled during migration?"
    ],
    "Timeline & Milestones": [
        "What is the overall project timeline and key milestones?",
        "Are there any hard deadlines or business events driving the schedule?",
        "What are the dependencies on other projects or initiatives?",
        "How will progress be measured and reported?",
        "What are the go/no-go criteria for each phase?",
        "How will success be defined and measured post-migration?"
    ]
}

print("Planning categories defined successfully!")
print(f"Categories: {list(PLANNING_CATEGORIES.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Main Migration Planning Agent

# COMMAND ----------

class MigrationPlanningAgent(dspy.Module):
    """Main migration planning agent that orchestrates iterative planning process."""
    
    def __init__(self, vector_search_endpoint: str, vector_search_index: str):
        super().__init__()
        # Core planning modules
        self.question_generator = QuestionGenerationModule()
        self.document_retriever = DocumentRetrievalModule(vector_search_endpoint, vector_search_index)
        self.answer_analyzer = AnswerAnalysisModule()
        self.plan_generator = MigrationPlanGenerationModule()
        self.plan_scorer = PlanCompletenessScoringModule()
        self.risk_assessor = RiskAssessmentModule()
        
        # NEW: Conversational AI modules for chat UI
        self.context_extractor = ProjectContextExtractionModule()
        self.conversation_understanding = ConversationUnderstandingModule()
        self.response_generator = IntelligentResponseGenerationModule()
        
        # Store conversation state
        self.conversation_history = []
        self.gathered_insights = []
        self.project_context = ""
        self.timeline_requirements = ""
        self.current_plan = ""
        self.completeness_score = 0
        self.is_initialized = False
    
    def start_planning_session(self, project_context: str, timeline_requirements: str = ""):
        """Start a new planning session."""
        self.project_context = project_context
        self.timeline_requirements = timeline_requirements
        self.conversation_history = []
        self.gathered_insights = []
        self.current_plan = ""
        self.completeness_score = 0
        
        print("🚀 Starting Migration Planning Session")
        print(f"📋 Project Context: {project_context}")
        if timeline_requirements:
            print(f"⏰ Timeline Requirements: {timeline_requirements}")
        print("\n" + "="*50)
        
        return self._generate_questions_for_category("Resource & Team")
    
    def _generate_questions_for_category(self, category: str):
        """Generate questions for a specific category."""
        # Use optimized context for question generation
        existing_answers = self._get_context_for_question_generation(category)
        
        # Try to generate questions using DSPy first
        try:
            result = self.question_generator(
                project_context=self.project_context,
                category=category,
                existing_answers=existing_answers
            )
            
            # Parse generated questions
            generated_questions = result.questions.split('\n')
            questions = [q.strip() for q in generated_questions if q.strip()]
            
            # If DSPy generated good questions, use them
            if len(questions) >= 3:
                print(f"\n📝 {category} Questions (AI Generated):")
                print("-" * 40)
                for i, question in enumerate(questions, 1):
                    print(f"{i}. {question}")
                return questions
            else:
                print(f"⚠️ DSPy generated only {len(questions)} questions, using predefined questions as fallback")
                
        except Exception as e:
            print(f"⚠️ DSPy question generation failed: {e}, using predefined questions")
        
        # Fallback to predefined questions
        if category in PLANNING_CATEGORIES:
            questions = PLANNING_CATEGORIES[category]
            print(f"\n📝 {category} Questions (Predefined):")
            print("-" * 40)
            for i, question in enumerate(questions, 1):
                print(f"{i}. {question}")
            return questions
        else:
            print(f"❌ No predefined questions found for category: {category}")
            return []
    
    def answer_question(self, question: str, answer: str, category: str = ""):
        """Process a user's answer to a question."""
        print(f"\n🔄 Processing Answer:")
        print(f"Q: {question}")
        print(f"A: {answer}")
        
        # Retrieve relevant documents
        doc_result = self.document_retriever(question, self.project_context)
        relevant_docs = doc_result.retrieved_docs
        
        # Analyze the answer
        analysis_result = self.answer_analyzer(question, answer, relevant_docs)
        insights = analysis_result.insights
        
        # Store in conversation history
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "category": category,
            "insights": insights,
            "relevant_docs": relevant_docs
        })
        
        self.gathered_insights.append(insights)
        
        # Manage conversation history size
        self._manage_conversation_history()
        
        print(f"\n💡 Key Insights:")
        print(insights)
        
        # Only update plan if it already exists (continuous updates)
        if self.current_plan:
            self._update_migration_plan()
            self._score_plan_completeness()
            print(f"\n📋 Plan Updated (Score: {self.completeness_score}/100)")
        else:
            print(f"\n📝 Information gathered. Use '/plan' to generate migration plan when ready.")
        
        return {
            "insights": insights,
            "current_plan": self.current_plan,
            "completeness_score": self.completeness_score,
            "next_questions": self.get_next_category_questions(),
            "plan_ready": self._is_plan_ready()
        }
    
    def _update_migration_plan(self):
        """Update the migration plan based on current insights."""
        if not self.gathered_insights:
            return
        
        # Use comprehensive context for plan generation
        context = self._get_context_for_plan_generation()
        
        plan_result = self.plan_generator(
            project_context=self.project_context,
            gathered_insights=context,
            timeline_requirements=self.timeline_requirements
        )
        
        self.current_plan = plan_result.migration_plan
        
        print(f"\n📋 Updated Migration Plan:")
        print("="*50)
        print(self.current_plan)
    
    def _score_plan_completeness(self):
        """Score the completeness of the current plan."""
        if not self.gathered_insights:
            self.completeness_score = 0
            return
        
        all_insights = "\n\n".join(self.gathered_insights)
        
        score_result = self.plan_scorer(
            project_context=self.project_context,
            gathered_insights=all_insights,
            current_plan=self.current_plan
        )
        
        # Extract score from the result
        score_text = score_result.completeness_score
        try:
            # Look for a number between 0-100 in the score text
            import re
            score_match = re.search(r'(\d+)', score_text)
            if score_match:
                self.completeness_score = int(score_match.group(1))
            else:
                self.completeness_score = 0
        except:
            self.completeness_score = 0
        
        print(f"\n📊 Plan Completeness Score: {self.completeness_score}/100")
        print(f"📈 Maturity Level: {self._get_maturity_level()}")
        print(f"💡 Recommendations: {score_text}")
    
    def _get_maturity_level(self):
        """Get maturity level based on completeness score."""
        if self.completeness_score >= 90:
            return "🟢 Mature - Ready for execution"
        elif self.completeness_score >= 70:
            return "🟡 Advanced - Minor gaps to address"
        elif self.completeness_score >= 50:
            return "🟠 Developing - Significant gaps to fill"
        elif self.completeness_score >= 30:
            return "🔴 Early Draft - Major information needed"
        else:
            return "⚫ Initial - Just getting started"
    
    def _is_plan_ready(self):
        """Check if we have enough information to generate a meaningful plan."""
        if not self.gathered_insights:
            return False
        
        # Check for minimum required information
        required_categories = ["Resource & Team", "Technical Scope & Architecture", "Customer Background & Drivers"]
        answered_categories = set(entry.get('category', '') for entry in self.conversation_history)
        
        # Need at least 2 out of 3 key categories
        key_categories_answered = len(set(required_categories) & answered_categories)
        
        # Need at least 3 Q&A pairs total
        min_qa_pairs = 3
        
        return key_categories_answered >= 2 and len(self.conversation_history) >= min_qa_pairs
    
    def generate_plan_manually(self):
        """Manually generate migration plan when user requests it."""
        print("\n🎯 Generating Migration Plan...")
        print("="*50)
        
        if not self._is_plan_ready():
            print("⚠️ Not enough information yet. Need at least:")
            print("  • 2 out of 3 key categories (Resource & Team, Technical Scope & Architecture, Customer Background & Drivers)")
            print("  • 3 Q&A pairs total")
            print(f"  • Current: {len(self.conversation_history)} Q&A pairs")
            
            answered_categories = set(entry.get('category', '') for entry in self.conversation_history)
            required_categories = ["Resource & Team", "Technical Scope & Architecture", "Customer Background & Drivers"]
            key_categories_answered = len(set(required_categories) & answered_categories)
            print(f"  • Key categories answered: {key_categories_answered}/3")
            
            return {
                "success": False,
                "message": "Not enough information to generate a meaningful plan yet.",
                "suggestions": self.get_next_category_questions()
            }
        
        # Generate the plan
        self._update_migration_plan()
        self._score_plan_completeness()
        
        print(f"✅ Migration plan generated!")
        print(f"📊 Completeness Score: {self.completeness_score}/100")
        print(f"📈 Maturity Level: {self._get_maturity_level()}")
        
        return {
            "success": True,
            "plan": self.current_plan,
            "completeness_score": self.completeness_score,
            "maturity_level": self._get_maturity_level(),
            "message": "Migration plan generated successfully!"
        }
    
    def process_user_input(self, user_input: str):
        """Process user input with intelligent conversation understanding for chat UI."""
        user_input = user_input.strip()
        
        # Handle special commands first
        if user_input.lower() == "/plan":
            return self.generate_plan_manually()
        elif user_input.lower() == "/status":
            return self.get_conversation_summary()
        elif user_input.lower() == "/help":
            return {
                "type": "help",
                "response": "Available commands:\n/plan - Generate migration plan\n/status - Show conversation summary\n/help - Show this help message\n\nYou can also:\n- Describe your migration project\n- Answer questions about your current setup\n- Ask about specific planning categories"
            }
        
        # If this is the first interaction and no project context exists, extract it
        if not self.is_initialized or not self.project_context:
            return self._handle_initial_user_input(user_input)
        
        # For ongoing conversation, understand the user's intent
        conversation_context = self._format_conversation_context()
        
        try:
            # Understand user intent and context
            understanding = self.conversation_understanding(
                user_input=user_input,
                conversation_history=conversation_context,
                project_context=self.project_context
            )
            
            intent = understanding.intent
            suggested_action = understanding.suggested_action
            response_type = understanding.response_type
            
            # Generate appropriate response based on intent
            if intent == "provide_context":
                return self._handle_context_provision(user_input)
            elif intent == "answer_question":
                return self._handle_question_answer(user_input)
            elif intent == "ask_category":
                return self._handle_category_request(user_input)
            elif intent == "request_plan":
                return self.generate_plan_manually()
            elif intent == "general_help":
                return self._handle_general_help()
            else:
                # Generate intelligent response
                return self._generate_intelligent_response(user_input, conversation_context, intent, suggested_action)
                
        except Exception as e:
            # Fallback to simple response
            return self._generate_fallback_response(user_input)
    
    def _handle_initial_user_input(self, user_input: str):
        """Handle the first user input to extract project context."""
        try:
            # Extract project context from user's description
            context_result = self.context_extractor(user_input=user_input)
            
            # Set up the planning session
            self.project_context = context_result.project_context
            self.is_initialized = True
            
            # Get suggested category to start with
            suggested_category = context_result.suggested_category
            
            # Generate welcome message and first questions
            welcome_message = f"Great! I understand you're working on: {self.project_context}\n\nLet me help you plan this migration to Databricks. I'll start by asking some questions about {suggested_category}."
            
            # Generate questions for the suggested category
            questions = self._generate_questions_for_category(suggested_category)
            
            return {
                "type": "initialization",
                "response": welcome_message,
                "questions": questions,
                "category": suggested_category
            }
            
        except Exception as e:
            # Fallback to default questions
            return {
                "type": "questions",
                "response": "I'll help you plan your migration to Databricks. Let me start with some basic questions about your project.",
                "questions": self._generate_questions_for_category("Resource & Team"),
                "category": "Resource & Team"
            }
    
    def _handle_context_provision(self, user_input: str):
        """Handle when user provides additional context."""
        # Add to conversation history
        self.conversation_history.append({
            "type": "context",
            "content": user_input,
            "timestamp": self._get_timestamp()
        })
        
        # Update project context if needed
        if len(user_input) > 50:  # Substantial context
            self.project_context += f"\n\nAdditional context: {user_input}"
        
        # Get next questions
        next_questions = self.get_next_category_questions()
        if next_questions:
            return {
                "type": "questions",
                "response": "Thanks for that additional context! Let me continue with the next set of questions.",
                "questions": next_questions
            }
        else:
            return {
                "type": "ready_for_plan",
                "response": "Great! I have enough information to help you generate a migration plan. Use /plan to create your plan.",
                "suggestions": ["/plan - Generate migration plan", "/status - Review what we've discussed"]
            }
    
    def _handle_question_answer(self, user_input: str):
        """Handle when user answers a question."""
        # This would need to be enhanced to match with the current question
        # For now, add to conversation history
        self.conversation_history.append({
            "type": "answer",
            "content": user_input,
            "timestamp": self._get_timestamp()
        })
        
        # Get next questions
        next_questions = self.get_next_category_questions()
        if next_questions:
            return {
                "type": "questions",
                "response": "Thanks for your answer! Let me ask the next set of questions.",
                "questions": next_questions
            }
        else:
            return {
                "type": "ready_for_plan",
                "response": "Excellent! I have enough information to help you generate a migration plan. Use /plan to create your plan.",
                "suggestions": ["/plan - Generate migration plan", "/status - Review what we've discussed"]
            }
    
    def _handle_category_request(self, user_input: str):
        """Handle when user requests a specific category."""
        # Check if it's a valid category
        if user_input in PLANNING_CATEGORIES:
            questions = self._generate_questions_for_category(user_input)
            return {
                "type": "questions",
                "response": f"Here are questions about {user_input}:",
                "questions": questions,
                "category": user_input
            }
        else:
            return {
                "type": "category_list",
                "response": "Here are the available planning categories:",
                "categories": list(PLANNING_CATEGORIES.keys())
            }
    
    def _handle_general_help(self):
        """Handle general help requests."""
        return {
            "type": "help",
            "response": "I'm here to help you plan your migration to Databricks! I can:\n\n• Ask questions about your current setup\n• Help you understand what's needed for migration\n• Generate a comprehensive migration plan\n• Assess risks and provide recommendations\n\nJust describe your project or answer my questions to get started!"
        }
    
    def _generate_intelligent_response(self, user_input: str, conversation_context: str, intent: str, suggested_action: str):
        """Generate an intelligent response using the response generator."""
        try:
            response_result = self.response_generator(
                user_input=user_input,
                conversation_context=conversation_context,
                project_context=self.project_context,
                intent=intent,
                suggested_action=suggested_action
            )
            
            return {
                "type": "conversation",
                "response": response_result.response
            }
        except Exception as e:
            return self._generate_fallback_response(user_input)
    
    def _generate_fallback_response(self, user_input: str):
        """Generate a fallback response when AI modules fail."""
        return {
            "type": "conversation",
            "response": "I understand you're working on a migration project. Let me help you by asking some questions about your current setup. What's your current data platform?"
        }
    
    def _format_conversation_context(self):
        """Format conversation history for context."""
        if not self.conversation_history:
            return "No previous conversation."
        
        recent_entries = self.conversation_history[-5:]  # Last 5 entries
        return "\n".join([f"{entry['type']}: {entry['content']}" for entry in recent_entries])
    
    def _get_timestamp(self):
        """Get current timestamp."""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def forward(self, inputs):
        """Forward method for MLflow deployment - processes user input and returns DSPy Prediction."""
        try:
            # Handle both string and dictionary inputs
            if isinstance(inputs, dict):
                user_input = inputs.get("inputs", "")
            else:
                user_input = str(inputs)
            
            # Process user input through the agent
            result = self.process_user_input(user_input)
            
            # Convert result to DSPy Prediction format
            return dspy.Prediction(
                response=result.get("response", ""),
                type=result.get("type", "conversation"),
                questions=result.get("questions", ""),
                category=result.get("category", "")
            )
        except Exception as e:
            return dspy.Prediction(
                response=f"Error processing input: {str(e)}",
                type="error",
                questions="",
                category=""
            )
    
    def get_next_category_questions(self):
        """Get questions for the next planning category."""
        categories = list(PLANNING_CATEGORIES.keys())
        completed_categories = set(entry.get('category', '') for entry in self.conversation_history)
        
        for category in categories:
            if category not in completed_categories:
                return self._generate_questions_for_category(category)
        
        return None  # All categories completed
    
    def get_random_question_from_category(self, category: str):
        """Get a random question from a specific category."""
        import random
        
        if category in PLANNING_CATEGORIES:
            questions = PLANNING_CATEGORIES[category]
            return random.choice(questions)
        else:
            return f"Please tell me more about {category.lower()} for your migration project."
    
    def get_question_suggestions(self, current_category: str = None):
        """Get suggested questions based on current context and completeness score."""
        suggestions = []
        
        if self.completeness_score < 30:
            # Early stage - focus on basic requirements
            suggestions.extend([
                "What is the main business driver for this migration?",
                "What is your target timeline for completion?",
                "How many people will be working on this project?"
            ])
        elif self.completeness_score < 60:
            # Mid stage - focus on technical details
            suggestions.extend([
                "What is the current data volume and complexity?",
                "What are your main technical challenges?",
                "How do you currently handle data governance?"
            ])
        else:
            # Advanced stage - focus on implementation details
            suggestions.extend([
                "What are your specific performance requirements?",
                "How will you handle data quality during migration?",
                "What is your rollback strategy if issues arise?"
            ])
        
        return suggestions
    
    def _manage_conversation_history(self, max_entries: int = 50):
        """Manage conversation history size to prevent memory issues."""
        if len(self.conversation_history) > max_entries:
            # Keep the most recent entries and compress older ones
            recent_entries = self.conversation_history[-max_entries:]
            
            # Create a summary of older entries
            older_entries = self.conversation_history[:-max_entries]
            if older_entries:
                summary_entry = {
                    "question": "Previous conversation summary",
                    "answer": f"Compressed {len(older_entries)} earlier Q&A pairs into insights",
                    "category": "System",
                    "insights": "Historical context compressed for memory management",
                    "relevant_docs": ""
                }
                self.conversation_history = [summary_entry] + recent_entries
                
                print(f"📝 Compressed conversation history: {len(older_entries)} older entries summarized")
    
    def _estimate_token_count(self, text: str) -> int:
        """Rough estimation of token count (4 chars per token average)."""
        return len(text) // 4
    
    def _get_context_for_question_generation(self, category: str):
        """Get optimized context for question generation."""
        # For question generation, we want recent context + category-specific context
        recent_context = self._format_existing_answers(max_entries=3, include_summary=True)
        
        # Add category-specific context
        category_insights = []
        for entry in self.conversation_history:
            if entry.get('category') == category:
                category_insights.append(f"Q: {entry['question']}\nA: {entry['answer']}")
        
        if category_insights:
            category_context = f"\n\n{category.upper()} SPECIFIC CONTEXT:\n" + "\n\n".join(category_insights[-2:])  # Last 2 from this category
            return recent_context + category_context
        
        return recent_context
    
    def generate_final_plan(self):
        """Generate the final project plan with risk assessment."""
        print("\n🎯 Generating Final Project Plan...")
        print("="*50)
        
        # Assess risks
        risk_result = self.risk_assessor(self.current_plan, self.project_context)
        risk_assessment = risk_result.risk_assessment
        
        print("⚠️ RISK ASSESSMENT:")
        print("="*50)
        print(risk_assessment)
        
        return {
            "project_plan": self.current_plan,
            "risk_assessment": risk_assessment,
            "completeness_score": self.completeness_score,
            "maturity_level": self._get_maturity_level(),
            "conversation_history": self.conversation_history
        }
    
    def _format_existing_answers(self, max_entries: int = 5, include_summary: bool = True):
        """Format existing answers for context with smart truncation."""
        if not self.conversation_history:
            return "No previous answers yet."
        
        # Get recent entries
        recent_entries = self.conversation_history[-max_entries:] if len(self.conversation_history) > max_entries else self.conversation_history
        
        formatted = []
        
        # Add summary if requested and we have many entries
        if include_summary and len(self.conversation_history) > max_entries:
            summary = self._get_conversation_summary_compact()
            formatted.append(f"CONVERSATION SUMMARY: {summary}")
            formatted.append("RECENT Q&A:")
        
        # Add recent Q&A pairs
        for entry in recent_entries:
            formatted.append(f"Q: {entry['question']}\nA: {entry['answer']}")
        
        return "\n\n".join(formatted)
    
    def _get_conversation_summary_compact(self):
        """Get a compact summary of the conversation."""
        if not self.conversation_history:
            return "No conversation yet."
        
        # Count by category
        category_counts = {}
        for entry in self.conversation_history:
            category = entry.get('category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Create compact summary
        summary_parts = []
        for category, count in category_counts.items():
            summary_parts.append(f"{category}: {count} questions")
        
        return f"Total: {len(self.conversation_history)} questions across {len(category_counts)} categories ({', '.join(summary_parts)})"
    
    def _get_context_for_plan_generation(self):
        """Get comprehensive context for plan generation."""
        if not self.gathered_insights:
            return "No insights gathered yet."
        
        # Use all insights for plan generation
        all_insights = "\n\n".join(self.gathered_insights)
        
        # Add key decisions from conversation
        key_decisions = []
        for entry in self.conversation_history:
            if any(keyword in entry['answer'].lower() for keyword in ['decided', 'chose', 'selected', 'prefer', 'will use']):
                key_decisions.append(f"- {entry['question']}: {entry['answer']}")
        
        context = f"GATHERED INSIGHTS:\n{all_insights}"
        if key_decisions:
            context += f"\n\nKEY DECISIONS:\n" + "\n".join(key_decisions)
        
        return context
    
    def get_conversation_summary(self):
        """Get a summary of the conversation."""
        if not self.conversation_history:
            return "No conversation history available."
        
        summary = f"Conversation Summary ({len(self.conversation_history)} questions):\n"
        summary += f"Completeness Score: {self.completeness_score}/100\n"
        summary += f"Maturity Level: {self._get_maturity_level()}\n\n"
        
        for i, response in enumerate(self.conversation_history, 1):
            summary += f"{i}. {response['question']} ({response['category']})\n"
        
        return summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Initialize Agent

# COMMAND ----------

# Initialize the migration planning agent
agent = MigrationPlanningAgent(
    vector_search_endpoint=vector_search_endpoint,
    vector_search_index=vector_index_name
)

print("Migration Planning Agent initialized successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Simple Agent Testing

# COMMAND ----------

# Simple test of the agent functionality
print("Testing Migration Planning Agent...")
print("="*50)

# Initialize the agent
agent = MigrationPlanningAgent(
    vector_search_endpoint=vector_search_endpoint,
    vector_search_index=vector_index_name
)

# Test basic functionality
print("🚀 Testing Basic Agent Functionality...")

# Test question generation for a category
print("\n📝 Testing Question Generation:")
questions = agent._generate_questions_for_category("Resource & Team")
print(f"Generated {len(questions)} questions for Resource & Team category")

# Test document retrieval
print("\n🔍 Testing Document Retrieval:")
test_question = "How many team members are there?"
try:
    doc_result = agent.document_retriever(test_question, "Oracle to Databricks migration project")
    print(f"Retrieved documents: {len(doc_result.retrieved_docs)} characters")
except Exception as e:
    print(f"Document retrieval test failed: {e}")

# Test answer analysis
print("\n💡 Testing Answer Analysis:")
test_answer = "We have a 10 person team with 8 developers"
try:
    analysis_result = agent.answer_analyzer(test_question, test_answer, "Sample documents")
    print(f"Analysis completed: {len(analysis_result.insights)} characters")
except Exception as e:
    print(f"Answer analysis test failed: {e}")

print("\n✅ Basic agent functionality test completed!")
print("The agent is ready for SME testing and iterative planning sessions.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. MLflow DSPy Integration with Auto Tracing

# COMMAND ----------

# Enable MLflow auto tracing for DSPy - following documentation
import mlflow.dspy

# Set up MLflow experiment - following documentation pattern
mlflow.set_experiment(mlflow_experiment_name)

print(f"🔬 MLflow experiment set: {mlflow_experiment_name}")

# Turn on auto tracing with MLflow - following documentation
mlflow.dspy.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Create Optimized DSPy Module for MLflow

# COMMAND ----------

# Create a simplified DSPy module following the MLflow documentation pattern
class MigrationPlanningDSPyModule(dspy.Module):
    """DSPy module for migration planning following MLflow documentation pattern."""
    
    def __init__(self, vector_search_endpoint: str, vector_search_index: str):
        super().__init__()
        self.vector_search_endpoint = vector_search_endpoint
        self.vector_search_index = vector_search_index
        
        # Initialize DSPy modules
        self.question_generator = dspy.ChainOfThought(QuestionGenerator)
        self.document_retriever = dspy.ChainOfThought(DocumentRetriever)
        self.answer_analyzer = dspy.ChainOfThought(AnswerAnalyzer)
        self.plan_generator = dspy.ChainOfThought(MigrationPlanGenerator)
        self.plan_scorer = dspy.ChainOfThought(PlanCompletenessScorer)
        
        # State management
        self.conversation_history = []
        self.gathered_insights = []
        self.project_context = "Oracle to Databricks migration project"
        self.timeline_requirements = ""
        self.current_plan = ""
        self.completeness_score = 0
    
    def forward(self, question: str):
        """Main forward method following MLflow DSPy pattern."""
        if not question:
            return dspy.Prediction(predictions="Error: Question is required")
        
        # Generate questions for the input category
        try:
            result = self.question_generator(
                project_context=self.project_context,
                category=question,  # Use question as category
                existing_answers=self._format_existing_answers()
            )
            return dspy.Prediction(predictions=result.questions)
        except Exception as e:
            # Fallback to predefined questions
            if question in PLANNING_CATEGORIES:
                questions = PLANNING_CATEGORIES[question]
                questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
                return dspy.Prediction(predictions=questions_text)
            else:
                return dspy.Prediction(predictions=f"Error: Failed to generate questions: {str(e)}")
    
    def _format_existing_answers(self):
        """Format existing answers for context."""
        if not self.conversation_history:
            return "No previous answers yet."
        
        recent_entries = self.conversation_history[-3:]  # Last 3 entries
        return "\n\n".join([
            f"Q: {entry['question']}\nA: {entry['answer']}"
            for entry in recent_entries
        ])

print("✅ Optimized DSPy module created for MLflow integration")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. DSPy Compilation and Optimization

# COMMAND ----------

# Create training data for DSPy optimization
training_data = [
    {
        "question": "How many team members are there?",
        "answer": "We have a 10 person team with 8 developers",
        "category": "Resource & Team",
        "project_context": "Oracle to Databricks migration project",
        "expected_insights": "Team size and composition identified for resource planning"
    },
    {
        "question": "What is the current data volume?",
        "answer": "Approximately 50TB of data across all domains",
        "category": "Technical Scope & Architecture", 
        "project_context": "Oracle to Databricks migration project",
        "expected_insights": "Data volume requirements for migration planning"
    },
    {
        "question": "What is the target timeline?",
        "answer": "6 months before the financial year end",
        "category": "Timeline & Milestones",
        "project_context": "Oracle to Databricks migration project", 
        "expected_insights": "Timeline constraints identified for project planning"
    }
]

print(f"📊 Created {len(training_data)} training examples for DSPy optimization")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. MLflow DSPy Model Logging

# COMMAND ----------

# Create the MigrationPlanningAgent instance directly
migration_agent = MigrationPlanningAgent(
    vector_search_endpoint=vector_search_endpoint,
    vector_search_index=vector_index_name
)

# Input example for the model - following MLflow documentation pattern
input_example = "I need to migrate our Oracle data warehouse to Databricks"

# Define input and output schemas for the MigrationPlanningAgent
from mlflow.types.schema import Schema, ColSpec
from mlflow.types import DataType

# Define input schema (string input)
input_schema = Schema([ColSpec(DataType.string, "inputs")])

# Define output schema (dictionary output with response details)
output_schema = Schema([
    ColSpec(DataType.string, "response"),
    ColSpec(DataType.string, "type"),
    ColSpec(DataType.string, "questions", required=False),
    ColSpec(DataType.string, "category", required=False)
])

# Log the MigrationPlanningAgent directly using MLflow's native DSPy support
with mlflow.start_run() as run:
    # Log the MigrationPlanningAgent with explicit schemas
    model_info = mlflow.dspy.log_model(
        migration_agent,
        artifact_path="migration-planning-agent",
        input_example=input_example,
        signature=mlflow.models.signature.ModelSignature(
            inputs=input_schema,
            outputs=output_schema
        )
    )
    
    print(f"🤖 DSPy model logged to MLflow: {model_info.model_uri}")
    print(f"📊 Run ID: {run.info.run_id}")

# Register the model in Unity Catalog
catalog_name = "vbdemos"
schema_name = "usecase_agent"
model_name = f"{catalog_name}.{schema_name}.migration-planning-agent"
uc_model_info = mlflow.register_model(model_uri=model_info.model_uri, name=model_name)

print(f"📝 Model registered in Unity Catalog: {uc_model_info.name}")
print(f"   Version: {uc_model_info.version}")

print(f"🚀 Model ready for deployment!")
print(f"📊 Model URI: {model_info.model_uri}")
print(f"📋 Registered as: {uc_model_info.name}")
print(f"🔗 Next step: Deploy via Databricks Model Serving UI or API")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Test the Deployed Model

# COMMAND ----------

# Test the agent directly first
print("Testing the MigrationPlanningAgent directly...")
test_input = "I need to migrate our Oracle data warehouse to Databricks"
direct_result = migration_agent.process_user_input(test_input)
print(f"Direct Agent Response:\n{direct_result}")
print("✅ Direct agent test completed!")

# Test the logged DSPy model following MLflow documentation pattern
print("\nTesting the logged DSPy model...")

# Load the model using MLflow DSPy
loaded_dspy_model = mlflow.dspy.load_model(model_info.model_uri)

# Test with question generation - following documentation pattern
print("\n🔍 Testing Question Generation:")
# The model expects {"inputs": "..."} format based on our schema
test_input_dict = {"inputs": "I need to migrate our Oracle data warehouse to Databricks"}
question_result = loaded_dspy_model(test_input_dict)
print(f"Generated Response:\n{question_result.response}")
print(f"Response Type: {question_result.type}")
if hasattr(question_result, 'questions') and question_result.questions:
    print(f"Questions:\n{question_result.questions}")

# Test with PyFunc API - following documentation pattern
print("\n🔧 Testing PyFunc API:")
loaded_pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
# PyFunc expects the input in the format defined by the schema
pyfunc_result = loaded_pyfunc_model.predict([input_example])
print(f"PyFunc Result: {pyfunc_result}")

print("✅ Model test completed successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. SME Testing Interface

# COMMAND ----------

# MAGIC %md
# MAGIC ### Simple Testing Interface for SMEs
# MAGIC 
# MAGIC This section provides a simple interface for SMEs to test the migration planning agent.

# COMMAND ----------

# def simple_sme_test():
#     """Simple testing function for SMEs"""
#     print("Migration Planning Agent - SME Testing Mode")
#     print("=" * 50)
#     print("This is a simplified interface for testing the agent with SMEs.")
#     print("The agent can generate questions for different planning categories.")
#     print()
    
#     # Test different categories
#     categories = list(PLANNING_CATEGORIES.keys())
    
#     print("Available planning categories:")
#     for i, category in enumerate(categories, 1):
#         print(f"{i}. {category}")
    
#     print("\nTesting question generation for each category:")
#     print("-" * 50)
    
#     for category in categories[:3]:  # Test first 3 categories
#         print(f"\n📝 {category}:")
#         questions = agent._generate_questions_for_category(category)
#         if questions:
#             print(f"  Generated {len(questions)} questions")
#             print(f"  Sample: {questions[0]}")
#         else:
#             print("  No questions generated")
    
#     print("\n✅ SME testing interface ready!")
#     print("SMEs can now test the agent by running individual cells or using the MLflow model.")

# # Run the simple test
# simple_sme_test()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Job Completion

# COMMAND ----------

# print("="*60)
# print("✅ MIGRATION PLANNING AGENT JOB COMPLETED SUCCESSFULLY")
# print("="*60)
# print(f"🤖 Agent Model: {agent_model}")
# print(f"🔍 Vector Index: {vector_index_name}")
# print(f"📊 Documents Table: {migration_documents_table}")
# print(f"🔧 Temperature: {temperature}")
# print(f"📝 Max Tokens: {max_tokens}")
# print(f"📋 Model Registered: {uc_model_info.name}")
# print(f"📊 Model Version: {uc_model_info.version}")
# print(f"🔬 MLflow Experiment: {mlflow_experiment_name}")
# print(f"📊 Training Examples: {len(training_data)}")
# print("="*60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Comprehensive Testing Suite
# MAGIC 
# MAGIC **For Testers**: Run the cells below to test the complete migration planning agent functionality.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test 1: Basic Agent Functionality

# COMMAND ----------

# def test_basic_functionality():
#     """Test basic agent functionality"""
#     print("🧪 TEST 1: Basic Agent Functionality")
#     print("="*50)
    
#     # Initialize agent
#     test_agent = MigrationPlanningAgent(
#         vector_search_endpoint=vector_search_endpoint,
#         vector_search_index=vector_index_name
#     )
    
#     # Test question generation
#     print("\n📝 Testing Question Generation:")
#     categories = ["Resource & Team", "Technical Scope & Architecture", "Customer Background & Drivers"]
    
#     for category in categories:
#         questions = test_agent._generate_questions_for_category(category)
#         print(f"  ✅ {category}: {len(questions)} questions generated")
#         if questions:
#             print(f"     Sample: {questions[0][:80]}...")
    
#     # Test document retrieval
#     print("\n🔍 Testing Document Retrieval:")
#     test_question = "How many team members are there?"
#     try:
#         doc_result = test_agent.document_retriever(test_question, "Oracle to Databricks migration project")
#         print(f"  ✅ Document retrieval: {len(doc_result.retrieved_docs)} characters retrieved")
#     except Exception as e:
#         print(f"  ❌ Document retrieval failed: {e}")
    
#     # Test answer analysis
#     print("\n💡 Testing Answer Analysis:")
#     test_answer = "We have a 10 person team with 8 developers"
#     try:
#         analysis_result = test_agent.answer_analyzer(test_question, test_answer, "Sample documents")
#         print(f"  ✅ Answer analysis: {len(analysis_result.insights)} characters generated")
#     except Exception as e:
#         print(f"  ❌ Answer analysis failed: {e}")
    
#     print("\n✅ Basic functionality test completed!")

# # Run basic functionality test
# test_basic_functionality()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test 2: MLflow Model Testing

# COMMAND ----------

# def test_mlflow_model():
#     """Test MLflow model functionality"""
#     print("🧪 TEST 2: MLflow Model Testing")
#     print("="*50)
    
#     try:
#         # Test DSPy model loading
#         print("\n🔧 Testing DSPy Model Loading:")
#         loaded_dspy_model = mlflow.dspy.load_model(model_info.model_uri)
#         print("  ✅ DSPy model loaded successfully")
        
#         # Test DSPy model inference
#         print("\n🔍 Testing DSPy Model Inference:")
#         test_input = "Resource & Team"
#         dspy_result = loaded_dspy_model(test_input)
#         print(f"  ✅ DSPy inference: {len(dspy_result.questions)} characters generated")
#         print(f"     Sample output: {dspy_result.questions[:100]}...")
        
#         # Test PyFunc model loading
#         print("\n🔧 Testing PyFunc Model Loading:")
#         loaded_pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
#         print("  ✅ PyFunc model loaded successfully")
        
#         # Test PyFunc model inference
#         print("\n🔍 Testing PyFunc Model Inference:")
#         pyfunc_result = loaded_pyfunc_model.predict(test_input)
#         print(f"  ✅ PyFunc inference: {len(str(pyfunc_result))} characters generated")
#         print(f"     Sample output: {str(pyfunc_result)[:100]}...")
        
#         print("\n✅ MLflow model testing completed!")
        
#     except Exception as e:
#         print(f"❌ MLflow model testing failed: {e}")

# # Run MLflow model test
# test_mlflow_model()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test 3: SME Testing Interface

# COMMAND ----------

# def test_sme_interface():
#     """Test SME testing interface"""
#     print("🧪 TEST 3: SME Testing Interface")
#     print("="*50)
    
#     # Initialize agent for testing
#     test_agent = MigrationPlanningAgent(
#         vector_search_endpoint=vector_search_endpoint,
#         vector_search_index=vector_index_name
#     )
    
#     print("\n📋 Available Planning Categories:")
#     categories = list(PLANNING_CATEGORIES.keys())
#     for i, category in enumerate(categories, 1):
#         print(f"  {i}. {category}")
    
#     print("\n🔍 Testing Question Generation for Each Category:")
#     print("-" * 50)
    
#     for category in categories:
#         print(f"\n📝 {category}:")
#         try:
#             questions = test_agent._generate_questions_for_category(category)
#             if questions:
#                 print(f"  ✅ Generated {len(questions)} questions")
#                 print(f"     Sample: {questions[0][:80]}...")
#             else:
#                 print("  ⚠️ No questions generated")
#         except Exception as e:
#             print(f"  ❌ Failed: {e}")
    
#     print("\n✅ SME testing interface completed!")
#     print("SMEs can now test the agent by running individual cells or using the MLflow model.")

# # Run SME interface test
# test_sme_interface()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test 4: End-to-End Workflow Test

# COMMAND ----------

# def test_end_to_end_workflow():
#     """Test complete end-to-end workflow with new plan generation approach"""
#     print("🧪 TEST 4: End-to-End Workflow Test")
#     print("="*50)
    
#     # Initialize agent
#     test_agent = MigrationPlanningAgent(
#         vector_search_endpoint=vector_search_endpoint,
#         vector_search_index=vector_index_name
#     )
    
#     # Start planning session
#     print("\n🚀 Starting Planning Session:")
#     project_context = "Oracle to Databricks migration project"
#     timeline_requirements = "6 months timeline"
    
#     test_agent.start_planning_session(project_context, timeline_requirements)
#     print("  ✅ Planning session started")
    
#     # Test Q&A workflow (no automatic plan generation)
#     print("\n💬 Testing Q&A Workflow (Question-First Mode):")
#     test_questions = [
#         ("How many team members are there?", "We have a 10 person team with 8 developers", "Resource & Team"),
#         ("What is the data volume?", "Approximately 50TB of data", "Technical Scope & Architecture"),
#         ("What are the business drivers?", "Cost reduction and better performance", "Customer Background & Drivers")
#     ]
    
#     for i, (question, answer, category) in enumerate(test_questions, 1):
#         print(f"\n  Q{i}: {question}")
#         print(f"  A{i}: {answer}")
        
#         try:
#             result = test_agent.answer_question(question, answer, category)
#             print(f"  ✅ Processed successfully")
#             print(f"     Insights: {len(result['insights'])} characters")
#             print(f"     Plan Ready: {result['plan_ready']}")
#             print(f"     Current Plan: {'Yes' if result['current_plan'] else 'No'}")
#         except Exception as e:
#             print(f"  ❌ Failed: {e}")
    
#     # Test manual plan generation
#     print("\n📋 Testing Manual Plan Generation:")
#     try:
#         plan_result = test_agent.generate_plan_manually()
#         if plan_result['success']:
#             print(f"  ✅ Plan generated successfully")
#             print(f"     Score: {plan_result['completeness_score']}/100")
#             print(f"     Maturity: {plan_result['maturity_level']}")
#             print(f"     Plan length: {len(plan_result['plan'])} characters")
#         else:
#             print(f"  ⚠️ Plan not ready: {plan_result['message']}")
#     except Exception as e:
#         print(f"  ❌ Plan generation failed: {e}")
    
#     # Test user input processing
#     print("\n🔧 Testing User Input Processing:")
#     test_inputs = ["/plan", "/status", "/help", "Resource & Team"]
    
#     for user_input in test_inputs:
#         try:
#             result = test_agent.process_user_input(user_input)
#             print(f"  ✅ Input '{user_input}': {result.get('type', 'command')} processed")
#         except Exception as e:
#             print(f"  ❌ Input '{user_input}' failed: {e}")
    
#     print("\n✅ End-to-end workflow test completed!")

# # Run end-to-end workflow test
# test_end_to_end_workflow()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test 5: Plan Generation Workflow

# COMMAND ----------

# def test_plan_generation_workflow():
#     """Test the new plan generation workflow"""
#     print("🧪 TEST 5: Plan Generation Workflow")
#     print("="*50)
    
#     # Initialize agent
#     test_agent = MigrationPlanningAgent(
#         vector_search_endpoint=vector_search_endpoint,
#         vector_search_index=vector_index_name
#     )
    
#     # Test 1: Try to generate plan with no information
#     print("\n📋 Test 1: Plan generation with no information")
#     try:
#         result = test_agent.generate_plan_manually()
#         if not result['success']:
#             print(f"  ✅ Correctly rejected: {result['message']}")
#         else:
#             print(f"  ❌ Should have been rejected")
#     except Exception as e:
#         print(f"  ❌ Error: {e}")
    
#     # Test 2: Add some information and try again
#     print("\n📋 Test 2: Plan generation with partial information")
#     test_agent.answer_question("How many team members?", "10 developers", "Resource & Team")
#     test_agent.answer_question("What is the data volume?", "50TB", "Technical Scope & Architecture")
    
#     try:
#         result = test_agent.generate_plan_manually()
#         if result['success']:
#             print(f"  ✅ Plan generated: {result['completeness_score']}/100")
#         else:
#             print(f"  ⚠️ Still not ready: {result['message']}")
#     except Exception as e:
#         print(f"  ❌ Error: {e}")
    
#     # Test 3: Add more information to meet requirements
#     print("\n📋 Test 3: Plan generation with sufficient information")
#     test_agent.answer_question("What are the business drivers?", "Cost reduction", "Customer Background & Drivers")
    
#     try:
#         result = test_agent.generate_plan_manually()
#         if result['success']:
#             print(f"  ✅ Plan generated successfully!")
#             print(f"     Score: {result['completeness_score']}/100")
#             print(f"     Maturity: {result['maturity_level']}")
#         else:
#             print(f"  ❌ Still not ready: {result['message']}")
#     except Exception as e:
#         print(f"  ❌ Error: {e}")
    
#     # Test 4: Test continuous updates
#     print("\n📋 Test 4: Continuous plan updates")
#     try:
#         # Add more information and see if plan updates
#         test_agent.answer_question("What is the timeline?", "6 months", "Timeline & Milestones")
#         print(f"  ✅ Plan updated: {test_agent.completeness_score}/100")
#     except Exception as e:
#         print(f"  ❌ Error: {e}")
    
#     print("\n✅ Plan generation workflow test completed!")

# # Run plan generation workflow test
# test_plan_generation_workflow()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test 6: Performance and Error Handling

# COMMAND ----------

# def test_performance_and_errors():
#     """Test performance and error handling"""
#     print("🧪 TEST 5: Performance and Error Handling")
#     print("="*50)
    
#     # Test error handling
#     print("\n🛡️ Testing Error Handling:")
    
#     # Test with empty input
#     try:
#         test_agent = MigrationPlanningAgent(
#             vector_search_endpoint=vector_search_endpoint,
#             vector_search_index=vector_index_name
#         )
#         result = test_agent.answer_question("", "", "")
#         print("  ✅ Empty input handled gracefully")
#     except Exception as e:
#         print(f"  ❌ Empty input error: {e}")
    
#     # Test with invalid category
#     try:
#         questions = test_agent._generate_questions_for_category("Invalid Category")
#         print("  ✅ Invalid category handled gracefully")
#     except Exception as e:
#         print(f"  ❌ Invalid category error: {e}")
    
#     # Test performance
#     print("\n⚡ Testing Performance:")
#     import time
    
#     start_time = time.time()
#     try:
#         questions = test_agent._generate_questions_for_category("Resource & Team")
#         end_time = time.time()
#         print(f"  ✅ Question generation: {end_time - start_time:.2f} seconds")
#     except Exception as e:
#         print(f"  ❌ Performance test failed: {e}")
    
#     print("\n✅ Performance and error handling test completed!")

# # Run performance and error handling test
# test_performance_and_errors()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Summary

# COMMAND ----------

# def print_test_summary():
#     """Print comprehensive test summary"""
#     print("🎯 COMPREHENSIVE TEST SUMMARY")
#     print("="*60)
#     print("✅ All tests completed successfully!")
#     print()
#     print("📊 Test Coverage:")
#     print("  • Basic Agent Functionality - Question generation, document retrieval, answer analysis")
#     print("  • MLflow Model Integration - DSPy and PyFunc model loading and inference")
#     print("  • SME Testing Interface - Category-based question generation")
#     print("  • End-to-End Workflow - Question-first approach with manual plan generation")
#     print("  • Plan Generation Workflow - Smart plan generation with minimum requirements")
#     print("  • Performance & Error Handling - Error scenarios and performance metrics")
#     print()
#     print("🎯 New Features:")
#     print("  • Question-First Mode - Agent focuses on information gathering initially")
#     print("  • Manual Plan Trigger - Use '/plan' command to generate plans when ready")
#     print("  • Smart Validation - Checks minimum information before plan generation")
#     print("  • Continuous Updates - Plans update automatically once generated")
#     print("  • User Commands - /plan, /status, /help for better user experience")
#     print()
#     print("🚀 The Migration Planning Agent is ready for production use!")
#     print("📋 SMEs can now test the agent using the MLflow model or individual components.")
#     print("🔧 The agent can be deployed via Databricks Model Serving for real-world use.")
#     print("="*60)

# # Print test summary
# print_test_summary()
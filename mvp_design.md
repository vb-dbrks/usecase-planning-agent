# MVP Simplified Agent Architecture

## Current vs. Proposed

### Current (Complex)
```
7 Agents: UserInteraction → InformationCollector → Quality → QuestionGenerator → Planner → MigrationPlanning → Response
Complex Signatures with 5-6 fields each
Mem0 Memory System
Complex Handoffs
```

### Proposed (Simple MVP)
```
2 Core Components:
1. ConversationManager (handles all flows)
2. PlanGenerator (handles /plan trigger)

Simple In-Memory Storage:
- conversation_history: List[str]
- information_summary: str
```

## User Flows

### Flow 1: Greeting
```
User: "Hi, what can you help with?"
Agent: "I'm a Use Case Planning Agent for Databricks Account teams. I help create use-case plans based on internal knowledge, past plans, and proposals."
```

### Flow 2: Information Collection Loop
```
User: "I want to migrate from Snowflake to Databricks"
Agent: "Great! Let me ask you 3 questions to understand your project better:

1. What's the size of your current Snowflake data warehouse?
2. What are your main use cases (analytics, ML, real-time)?
3. What's your timeline for this migration?

Please answer these questions, and I'll ask 3 more based on your responses."
```

### Flow 3: Feedback on Gaps
```
User: "How's the information collection going?"
Agent: "Based on what you've shared, I have good information about:
- Data size: 50TB
- Use cases: Analytics and ML
- Timeline: 6 months

I still need more details about:
- Security requirements
- Performance expectations
- Team structure"
```

### Flow 4: Plan Generation
```
User: "/plan" or "Generate a plan"
Agent: "Based on your information and our knowledge base, here's your migration plan:

[Generated plan with assumptions and risks]

Assumptions made:
- Standard security requirements
- 8-hour maintenance windows

Risks identified:
- Data volume might require longer migration time
- Team training needs not specified"
```

## Simplified Components

### 1. ConversationManager
- Handles all 4 flows
- Manages conversation_history and information_summary
- Simple state machine

### 2. PlanGenerator
- Triggered by /plan or planning intent
- Combines information_summary + vector search
- Generates plan with assumptions/risks

### 3. Simple Storage
```python
class SimpleStorage:
    def __init__(self):
        self.conversation_history = []
        self.information_summary = ""
    
    def add_conversation(self, user_input, agent_response):
        self.conversation_history.append({
            "user": user_input,
            "agent": agent_response,
            "timestamp": datetime.now()
        })
    
    def update_summary(self, new_info):
        # Simple concatenation or basic summarization
        self.information_summary += f"\n{new_info}"
```

## Benefits of MVP Approach

### ✅ Simplicity
- 2 components instead of 7 agents
- Simple in-memory storage
- Clear user flows

### ✅ Maintainability
- Easy to debug and modify
- Clear separation of concerns
- No complex handoffs

### ✅ Performance
- Faster execution
- Less memory usage
- Simpler testing

### ✅ User Experience
- Clear conversation flow
- Predictable behavior
- Easy to understand

## Implementation Plan

### Phase 1: Core Components
1. Create ConversationManager
2. Create SimpleStorage
3. Implement 4 user flows

### Phase 2: Integration
1. Keep existing MLflow registration
2. Keep existing endpoint deployment
3. Replace complex agents with simple components

### Phase 3: Enhancement
1. Add few-shot examples for questions
2. Improve summarization
3. Add vector search integration

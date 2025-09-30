const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Serve static files from the React app build directory
app.use(express.static(path.join(__dirname, '../ui-frontend/build')));

// Mock data (same as before)
const accounts = [
  {
    id: '1',
    name: 'Acme Corporation',
    industry: 'Technology',
    region: 'North America',
    status: 'active',
    lastContact: '2024-01-15T10:30:00Z',
    useCaseCount: 2,
  },
  {
    id: '2',
    name: 'Global Manufacturing Ltd',
    industry: 'Manufacturing',
    region: 'Europe',
    status: 'active',
    lastContact: '2024-01-12T09:15:00Z',
    useCaseCount: 1,
  },
  {
    id: '3',
    name: 'HealthTech Solutions',
    industry: 'Healthcare',
    region: 'Asia Pacific',
    status: 'prospect',
    lastContact: '2024-01-08T16:45:00Z',
    useCaseCount: 1,
  },
];

const useCases = [
  {
    id: '1',
    title: 'Oracle Migration',
    description: 'Asset Management System Migration',
    accountId: '1',
    status: 'active',
    lastUpdated: '2024-01-15T10:30:00Z',
    messageCount: 24,
  },
  {
    id: '2',
    title: 'Sybase Migration',
    description: 'Research Analytics Platform',
    accountId: '1',
    status: 'completed',
    lastUpdated: '2024-01-10T14:20:00Z',
    messageCount: 18,
  },
  {
    id: '3',
    title: 'Data Warehouse Modernization',
    description: 'Customer Data Platform Upgrade',
    accountId: '2',
    status: 'active',
    lastUpdated: '2024-01-12T09:15:00Z',
    messageCount: 31,
  },
  {
    id: '4',
    title: 'Clinical Trial Analytics',
    description: 'Real-time Data Processing System',
    accountId: '3',
    status: 'draft',
    lastUpdated: '2024-01-08T16:45:00Z',
    messageCount: 8,
  },
];

// API Routes
app.get('/api/accounts', (req, res) => {
  res.json(accounts);
});

app.get('/api/accounts/:id', (req, res) => {
  const account = accounts.find(acc => acc.id === req.params.id);
  if (!account) {
    return res.status(404).json({ error: 'Account not found' });
  }
  res.json(account);
});

app.get('/api/usecases', (req, res) => {
  const { accountId } = req.query;
  let filteredUseCases = useCases;
  
  if (accountId) {
    filteredUseCases = useCases.filter(uc => uc.accountId === accountId);
  }
  
  res.json(filteredUseCases);
});

app.get('/api/usecases/:id', (req, res) => {
  const useCase = useCases.find(uc => uc.id === req.params.id);
  if (!useCase) {
    return res.status(404).json({ error: 'Use case not found' });
  }
  res.json(useCase);
});

app.post('/api/usecases', (req, res) => {
  const { title, description, accountId } = req.body;
  
  const newUseCase = {
    id: (useCases.length + 1).toString(),
    title,
    description,
    accountId,
    status: 'draft',
    lastUpdated: new Date().toISOString(),
    messageCount: 0,
  };
  
  useCases.push(newUseCase);
  res.status(201).json(newUseCase);
});

// Simplified chat endpoint using MLflow's built-in context management
app.post('/api/chat', async (req, res) => {
  try {
    const {
      message,
      conversationId,
      conversation_id,
      userId,
      user_id,
    } = req.body;
    console.log('🤖 Received chat request:', { 
      message: message?.substring(0, 100) + '...',
      conversation_id: incomingConversationId || 'none',
      user_id: incomingUserId || 'none'
    });
    
    // Get Databricks token from environment variable
    const databricksToken = process.env.DATABRICKS_TOKEN;
    if (!databricksToken) {
      console.error('❌ DATABRICKS_TOKEN not set');
      return res.status(500).json({ 
        error: 'DATABRICKS_TOKEN environment variable not set' 
      });
    }
    
    console.log('✅ DATABRICKS_TOKEN is set, calling agent...');

    const incomingConversationId = conversation_id || conversationId;
    const incomingUserId = user_id || userId;

    // Generate conversation ID if not provided
    const currentConversationId = incomingConversationId || `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const currentUserId = incomingUserId || currentConversationId;
    
    // Create request using MLflow's standard format
    const agentRequest = {
      input: [
        {
          role: "user",
          content: message
        }
      ],
      context: {
        conversation_id: currentConversationId,
        user_id: currentUserId
      },
      metadata: {
        conversation_id: currentConversationId,
        user_id: currentUserId
      }
    };

    console.log('🤖 Calling Databricks agent endpoint...');
    console.log('🤖 Request context:', agentRequest.context);
    
    const agentResponse = await fetch(
      'https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/simplified-migration-planning-agent/invocations',
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${databricksToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(agentRequest)
      }
    );

    if (!agentResponse.ok) {
      const errorText = await agentResponse.text();
      console.error('❌ Agent request failed:', errorText);
      throw new Error(`Databricks agent request failed with status ${agentResponse.status}: ${errorText}`);
    }

    const agentData = await agentResponse.json();
    let responseText = "I'm here to help with your migration planning. Could you tell me more about your project?";
    
    if (agentData && agentData.output && agentData.output.length > 0) {
      const firstOutput = agentData.output[0];
      if (firstOutput.content && firstOutput.content.length > 0) {
        const textContent = firstOutput.content.find(item => item.type === 'output_text');
        if (textContent && textContent.text) {
          responseText = textContent.text;
        }
      }
    } else {
      throw new Error('No response received from Databricks agent');
    }

    res.json({ 
      response: responseText,
      timestamp: new Date().toISOString(),
      conversation_id: currentConversationId,
      user_id: currentUserId
    });

  } catch (error) {
    console.error('Error calling Databricks agent:', error);
    res.status(500).json({ 
      error: 'Failed to get response from planning agent',
      details: error.message 
    });
  }
});

// Catch all handler: send back React's index.html file for any non-API routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../ui-frontend/build', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`📱 Frontend available at http://localhost:${PORT}`);
  console.log(`🔗 API available at http://localhost:${PORT}/api`);
  console.log(`💬 Chat endpoint: http://localhost:${PORT}/api/chat`);
});

module.exports = app;

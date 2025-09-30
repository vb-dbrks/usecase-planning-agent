const express = require('express');
const cors = require('cors');
const path = require('path');
const http = require('http');
const socketIo = require('socket.io');
const fetch = require('node-fetch');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "http://localhost:3000",
    methods: ["GET", "POST"]
  }
});

const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Mock data
const accounts = [
  {
    id: '1',
    name: 'UBS',
    industry: 'Investment Banking',
    description: 'Swiss multinational investment bank and financial services company',
    color: '#1E3A8A',
  },
  {
    id: '2',
    name: 'Leeds Building Society',
    industry: 'Banking',
    description: 'UK-based building society providing financial services',
    color: '#10B981',
  },
  {
    id: '3',
    name: 'Astellas',
    industry: 'Pharmaceutical',
    description: 'Japanese multinational pharmaceutical company',
    color: '#F59E0B',
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

// Chat endpoint to call Databricks agent
app.post('/api/chat', async (req, res) => {
  try {
    const {
      message,
      conversationId,
      conversation_id,
      userId,
      user_id,
      endpointKey,
      endpoint_key,
    } = req.body;

    const selectedEndpointKey = endpoint_key || endpointKey || 'simplified';
    const incomingConversationId = conversation_id || conversationId;
    const incomingUserId = user_id || userId;

    console.log('🤖 Received chat request:', { 
      message: message?.substring(0, 100) + '...',
      conversation_id: incomingConversationId || 'none',
      user_id: incomingUserId || 'none',
      endpoint_key: selectedEndpointKey
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

    // Prepare the request for the Databricks agent (correct format)
    const currentConversationId = incomingConversationId || `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const currentUserId = incomingUserId || currentConversationId;
    const agentRequest = {
      input: [
        {
          role: "user",
          content: message
        }
      ],
      context: {
        user_id: currentUserId,
        conversation_id: currentConversationId
      },
      metadata: {
        user_id: currentUserId,
        conversation_id: currentConversationId
      }
    };

    // Call the Databricks agent endpoint
    console.log('🤖 Calling Databricks agent endpoint...');
    console.log('🤖 Request context:', agentRequest.context);
    const endpointUrl = selectedEndpointKey === 'mvp'
      ? 'https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/migration-planning-agent/invocations'
      : 'https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/simplified-migration-planning-agent/invocations';

    const agentResponse = await fetch(
      endpointUrl,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${databricksToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(agentRequest)
      }
    );

    console.log('🤖 Agent response status:', agentResponse.status);
    if (!agentResponse.ok) {
      const errorText = await agentResponse.text();
      console.error('❌ Agent request failed:', errorText);
      throw new Error(`Databricks agent request failed with status ${agentResponse.status}: ${errorText}`);
    }

    const agentData = await agentResponse.json();
    console.log('🤖 Agent response received, parsing...');
    
    // Extract the response text from the agent's output (correct format)
    let responseText = "I'm here to help with your migration planning. Could you tell me more about your project?";
    let rawSections = [];
    
    if (agentData && agentData.output && agentData.output.length > 0) {
      const firstOutput = agentData.output[0];
      if (firstOutput.content && firstOutput.content.length > 0) {
        // Extract text from content array
        const textContent = firstOutput.content.find(item => item.type === 'output_text');
        if (textContent && textContent.text) {
          responseText = textContent.text;
          rawSections = firstOutput.content
            .filter(item => item.type === 'output_text' && item.text)
            .map(item => item.text);
        }
      }
    } else {
      throw new Error('No response received from Databricks agent');
    }

    res.json({ 
      response: responseText,
      sections: rawSections,
      timestamp: new Date().toISOString(),
      conversation_id: currentConversationId,
      user_id: currentUserId,
      endpoint_key: selectedEndpointKey
    });

  } catch (error) {
    console.error('Error calling Databricks agent:', error);
    res.status(500).json({ 
      error: 'Failed to get response from planning agent',
      details: error.message 
    });
  }
});

// Socket.io for real-time chat
io.on('connection', (socket) => {
  console.log('User connected:', socket.id);

  socket.on('join-chat', (data) => {
    socket.join(data.chatId);
    console.log(`User ${socket.id} joined chat ${data.chatId}`);
  });

  socket.on('send-message', (data) => {
    // Broadcast message to all users in the chat room
    socket.to(data.chatId).emit('receive-message', {
      id: Date.now().toString(),
      text: data.message,
      sender: 'agent',
      timestamp: new Date().toISOString(),
    });
  });

  socket.on('disconnect', () => {
    console.log('User disconnected:', socket.id);
  });
});

// Serve static files from React build
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '../ui-frontend/build')));
  
  app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../ui-frontend/build/index.html'));
  });
}

server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

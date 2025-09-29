import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  IconButton,
  AppBar,
  Toolbar,
  Paper,
  Avatar,
  CircularProgress,
} from '@mui/material';
import {
  ArrowBack,
  Menu,
  AccountCircle,
  Send,
  SmartToy,
  Person,
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import { useNavigate, useLocation } from 'react-router-dom';
import { Message, Account, UseCase } from '../types';

const ChatScreen: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const account = location.state?.account as Account;
  const useCase = location.state?.useCase as UseCase;
  const isNewChat = !useCase;

  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Initialize with welcome message
    if (isNewChat) {
      setMessages([
        {
          id: '1',
          text: `Hello! I'm your **Databricks Migration Planning Agent**. I'll help you create comprehensive use case plans for customer migrations and greenfield scenarios.

**What I can help you with:**
• **Migration Planning**: Developing detailed plans for customers moving from legacy data platforms (like Oracle, Snowflake, Teradata, etc.) to Databricks
• **Greenfield Use Cases**: Creating implementation strategies for new data initiatives and modern analytics projects
• **Technical Architecture**: Recommending optimal Databricks configurations, cluster sizing, and feature selection
• **Timeline & Milestones**: Establishing realistic project phases, dependencies, and success criteria
• **Resource Planning**: Identifying required skills, training needs, and support resources
• **ROI Analysis**: Helping quantify business value and cost optimization opportunities

**Available Commands:**
• \`/plan\` - Generate your comprehensive delivery plan
• \`/status\` - View current progress and data gathered
• \`/help\` - Show help information

Let's start by understanding your ${account?.name} project requirements. What customer scenario would you like to work on today?`,
          sender: 'agent',
          timestamp: new Date().toISOString(),
          type: 'system',
        },
      ]);
    } else {
      setMessages([
        {
          id: '1',
          text: `Welcome back to your ${useCase?.title} conversation. I'm here to help you continue planning your migration project.`,
          sender: 'agent',
          timestamp: new Date().toISOString(),
          type: 'system',
        },
      ]);
    }
  }, [isNewChat, account, useCase]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Call the real Databricks agent
    try {
      const agentResponseText = await generateAgentResponse(inputValue);
      const agentResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: agentResponseText,
        sender: 'agent',
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, agentResponse]);
    } catch (error) {
      console.error('Error calling Databricks agent:', error);
      const errorResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: "I'm having trouble connecting to the Databricks planning agent. Please check your connection and try again.",
        sender: 'agent',
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorResponse]);
    } finally {
      setIsLoading(false);
    }
  };

  const generateAgentResponse = async (userInput: string): Promise<string> => {
    console.log('🤖 Calling Databricks agent with message:', userInput);
    
    // Call the deployed Databricks agent with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
    
    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userInput,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('🤖 Agent response received:', data);
      return data.response || "I'm here to help with your migration planning. Could you tell me more about your project?";
    } catch (error) {
      clearTimeout(timeoutId);
      console.error('🤖 Error calling agent:', error);
      throw error;
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  const handleBackClick = () => {
    if (isNewChat) {
      navigate('/');
    } else {
      navigate(`/usecases/${account?.id}`, { state: { account } });
    }
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <Box sx={{ flexGrow: 1, backgroundColor: '#F8FAFC', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <AppBar position="static" elevation={0} sx={{ backgroundColor: 'white', borderBottom: '1px solid #E2E8F0' }}>
        <Toolbar>
          <IconButton edge="start" color="inherit" onClick={handleBackClick} sx={{ mr: 2, color: '#64748B' }}>
            <ArrowBack />
          </IconButton>
          <IconButton edge="start" color="inherit" sx={{ mr: 2, color: '#64748B' }}>
            <Menu />
          </IconButton>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, color: '#1E293B', fontWeight: 600 }}>
            {isNewChat ? `${account?.name} - New Use Case Chat` : `${account?.name} - ${useCase?.title}`}
          </Typography>
          <IconButton color="inherit" sx={{ color: '#64748B' }}>
            <AccountCircle />
          </IconButton>
        </Toolbar>
      </AppBar>

      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Messages Area */}
        <Box
          sx={{
            flexGrow: 1,
            overflowY: 'auto',
            p: 2,
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
          }}
        >
          {messages.map((message) => (
            <Box
              key={message.id}
              sx={{
                display: 'flex',
                justifyContent: message.sender === 'user' ? 'flex-end' : 'flex-start',
                alignItems: 'flex-start',
                gap: 1,
              }}
            >
              {message.sender === 'agent' && (
                <Avatar sx={{ bgcolor: '#FF6B35', width: 32, height: 32 }}>
                  <SmartToy />
                </Avatar>
              )}
              
              <Box
                sx={{
                  maxWidth: '70%',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: message.sender === 'user' ? 'flex-end' : 'flex-start',
                }}
              >
                <Paper
                  sx={{
                    p: 2,
                    backgroundColor: message.sender === 'user' ? '#FF6B35' : 'white',
                    color: message.sender === 'user' ? 'white' : '#1E293B',
                    borderRadius: message.sender === 'user' ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
                    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
                  }}
                >
                  <Typography variant="body1" sx={{ lineHeight: 1.6 }}>
                    <ReactMarkdown>{message.text}</ReactMarkdown>
                  </Typography>
                </Paper>
                <Typography
                  variant="caption"
                  sx={{
                    color: '#64748B',
                    mt: 0.5,
                    px: 1,
                  }}
                >
                  {formatTime(message.timestamp)}
                </Typography>
              </Box>

              {message.sender === 'user' && (
                <Avatar sx={{ bgcolor: '#1E3A8A', width: 32, height: 32 }}>
                  <Person />
                </Avatar>
              )}
            </Box>
          ))}

          {isLoading && (
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'flex-start',
                alignItems: 'flex-start',
                gap: 1,
              }}
            >
              <Avatar sx={{ bgcolor: '#FF6B35', width: 32, height: 32 }}>
                <SmartToy />
              </Avatar>
              <Paper
                sx={{
                  p: 2,
                  backgroundColor: 'white',
                  borderRadius: '18px 18px 18px 4px',
                  boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                }}
              >
                <CircularProgress size={16} />
                <Typography variant="body2" sx={{ color: '#64748B' }}>
                  Agent is typing...
                </Typography>
              </Paper>
            </Box>
          )}

          <div ref={messagesEndRef} />
        </Box>

        {/* Input Area */}
        <Box
          sx={{
            p: 2,
            backgroundColor: 'white',
            borderTop: '1px solid #E2E8F0',
          }}
        >
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
            <TextField
              fullWidth
              multiline
              maxRows={4}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message here..."
              variant="outlined"
              disabled={isLoading}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: '24px',
                  backgroundColor: '#F8FAFC',
                  '& fieldset': {
                    borderColor: '#E2E8F0',
                  },
                  '&:hover fieldset': {
                    borderColor: '#CBD5E1',
                  },
                  '&.Mui-focused fieldset': {
                    borderColor: '#FF6B35',
                  },
                },
              }}
            />
            <IconButton
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              sx={{
                bgcolor: '#FF6B35',
                color: 'white',
                width: 48,
                height: 48,
                '&:hover': {
                  bgcolor: '#E64A19',
                },
                '&:disabled': {
                  bgcolor: '#CBD5E1',
                  color: '#94A3B8',
                },
              }}
            >
              <Send />
            </IconButton>
          </Box>
        </Box>
      </Box>
    </Box>
  );
};

export default ChatScreen;

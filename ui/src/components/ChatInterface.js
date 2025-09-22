import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import Message from './Message';
import './ChatInterface.css';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const examplePrompts = [
    "Help me plan a migration from on-premises to cloud",
    "What are the key considerations for data migration?",
    "Analyze the current system architecture and suggest improvements",
    "Create a migration timeline for our data warehouse",
    "What security measures should we implement during migration?"
  ];

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage.content,
          conversation_history: messages.map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.response,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Error sending message:', err);
      setError('Failed to send message. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleExampleClick = (prompt) => {
    setInputValue(prompt);
  };

  return (
    <div className="chat-container">
      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <h2>Welcome to the Usecase Delivery Planning Agent</h2>
            <p>I'm here to help you plan and execute your data migration projects. Ask me anything about:</p>
            <div className="example-prompts">
              {examplePrompts.map((prompt, index) => (
                <button
                  key={index}
                  className="example-prompt"
                  onClick={() => handleExampleClick(prompt)}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <Message key={message.id} message={message} />
          ))
        )}
        
        {isLoading && (
          <div className="message assistant">
            <div className="message-avatar">AI</div>
            <div className="message-content">
              <div className="loading-indicator">
                <Loader2 className="spinner" />
                <span>Thinking...</span>
              </div>
            </div>
          </div>
        )}
        
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <div className="input-container">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about migration planning, architecture, or any data-related questions..."
          className="input-field"
          disabled={isLoading}
        />
        <button
          onClick={handleSendMessage}
          disabled={!inputValue.trim() || isLoading}
          className="send-button"
        >
          {isLoading ? (
            <Loader2 className="spinner" />
          ) : (
            <Send size={20} />
          )}
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;

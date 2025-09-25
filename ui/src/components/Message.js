import React from 'react';
import { User, Bot } from 'lucide-react';
import './Message.css';

const Message = ({ message }) => {
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';

  const formatContent = (content) => {
    // Simple formatting for code blocks and lists
    const lines = content.split('\n');
    const formattedLines = lines.map((line, index) => {
      // Check for code blocks
      if (line.startsWith('```')) {
        return <div key={index} className="code-block-marker">{line}</div>;
      }
      
      // Check for bullet points
      if (line.startsWith('- ') || line.startsWith('* ')) {
        return <div key={index} className="bullet-point">{line}</div>;
      }
      
      // Check for numbered lists
      if (/^\d+\.\s/.test(line)) {
        return <div key={index} className="numbered-point">{line}</div>;
      }
      
      // Check for headers
      if (line.startsWith('## ')) {
        return <h3 key={index} className="message-header">{line.substring(3)}</h3>;
      }
      
      if (line.startsWith('# ')) {
        return <h2 key={index} className="message-header">{line.substring(2)}</h2>;
      }
      
      // Regular text
      return <div key={index} className="message-text">{line}</div>;
    });
    
    return formattedLines;
  };

  return (
    <div className={`message ${message.role}`}>
      <div className="message-avatar">
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>
      <div className="message-content">
        <div className="message-body">
          {formatContent(message.content)}
        </div>
        <div className="message-timestamp">
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
};

export default Message;

import React from 'react';
import { Bot, Database } from 'lucide-react';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="header-content">
        <div className="header-left">
          <div className="logo">
            <Bot className="logo-icon" />
            <span className="logo-text">Usecase Delivery Planning Agent</span>
          </div>
        </div>
        <div className="header-right">
          <div className="status-indicator">
            <Database className="status-icon" />
            <span className="status-text">Connected to Databricks</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;

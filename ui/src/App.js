import React, { useState, useRef, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
import Header from './components/Header';
import './App.css';

function App() {
  return (
    <div className="App">
      <Header />
      <ChatInterface />
    </div>
  );
}

export default App;

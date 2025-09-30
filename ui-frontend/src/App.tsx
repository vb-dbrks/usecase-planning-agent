import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import databricksTheme from './theme/databricksTheme';
import AccountScreen from './components/AccountScreen';
import UseCaseScreen from './components/UseCaseScreen';
import ChatScreen from './components/ChatScreen';

function App() {
  return (
    <ThemeProvider theme={databricksTheme}>
      <CssBaseline />
      <Router>
        <Routes>
          <Route path="/" element={<AccountScreen />} />
          <Route path="/usecases/:accountId" element={<UseCaseScreen />} />
          <Route path="/chat/:useCaseId" element={<ChatScreen />} />
          <Route path="/new-chat" element={<ChatScreen />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
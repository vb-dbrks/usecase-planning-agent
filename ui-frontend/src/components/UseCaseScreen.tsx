import React from 'react';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
  CardActionArea,
  Chip,
  Avatar,
  IconButton,
  AppBar,
  Toolbar,
} from '@mui/material';
import {
  ArrowBack,
  Menu,
  AccountCircle,
  Storage,
  Analytics,
  Add,
  Chat,
  Schedule,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { mockUseCases } from '../data/mockData';
import { Account, UseCase } from '../types';

const UseCaseScreen: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const account = location.state?.account as Account;

  const getUseCaseIcon = (title: string) => {
    if (title.toLowerCase().includes('oracle') || title.toLowerCase().includes('sybase')) {
      return <Storage />;
    }
    return <Analytics />;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return '#10B981';
      case 'completed':
        return '#64748B';
      case 'draft':
        return '#F59E0B';
      default:
        return '#64748B';
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'active':
        return 'Active';
      case 'completed':
        return 'Completed';
      case 'draft':
        return 'Draft';
      default:
        return status;
    }
  };

  const handleUseCaseClick = (useCase: UseCase) => {
    navigate(`/chat/${useCase.id}`, { state: { useCase, account } });
  };

  const handleNewUseCaseClick = () => {
    navigate('/new-chat', { state: { account } });
  };

  const handleBackClick = () => {
    navigate('/');
  };

  const accountUseCases = mockUseCases.filter(uc => uc.accountId === account?.id);

  return (
    <Box sx={{ flexGrow: 1, backgroundColor: '#F8FAFC', minHeight: '100vh' }}>
      <AppBar position="static" elevation={0} sx={{ backgroundColor: 'white', borderBottom: '1px solid #E2E8F0' }}>
        <Toolbar>
          <IconButton edge="start" color="inherit" onClick={handleBackClick} sx={{ mr: 2, color: '#64748B' }}>
            <ArrowBack />
          </IconButton>
          <IconButton edge="start" color="inherit" sx={{ mr: 2, color: '#64748B' }}>
            <Menu />
          </IconButton>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, color: '#1E293B', fontWeight: 600 }}>
            {account?.name} - Use Cases
          </Typography>
          <IconButton color="inherit" sx={{ color: '#64748B' }}>
            <AccountCircle />
          </IconButton>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box sx={{ mb: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom sx={{ color: '#1E293B', fontWeight: 700 }}>
            Use Cases
          </Typography>
          <Typography variant="body1" sx={{ color: '#64748B', fontSize: '1.125rem' }}>
            Select a use case to continue your conversation or create a new one
          </Typography>
        </Box>

        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: {
              xs: '1fr',
              sm: 'repeat(2, 1fr)',
              md: 'repeat(3, 1fr)',
            },
            gap: 3,
          }}
        >
          {accountUseCases.map((useCase) => (
            <Box key={useCase.id}>
              <Card
                sx={{
                  height: 280,
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'all 0.3s ease-in-out',
                  cursor: 'pointer',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
                  },
                }}
              >
                <CardActionArea
                  onClick={() => handleUseCaseClick(useCase)}
                  sx={{ height: '100%', p: 0 }}
                >
                  <CardContent sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <Box sx={{ display: 'flex', alignItems: 'flex-start', mb: 2 }}>
                      <Avatar
                        sx={{
                          bgcolor: '#FF6B35',
                          width: 48,
                          height: 48,
                          mr: 2,
                        }}
                      >
                        {getUseCaseIcon(useCase.title)}
                      </Avatar>
                      <Box sx={{ flexGrow: 1 }}>
                        <Typography variant="h6" component="h2" sx={{ fontWeight: 600, color: '#1E293B', mb: 1 }}>
                          {useCase.title}
                        </Typography>
                        <Chip
                          label={getStatusLabel(useCase.status)}
                          size="small"
                          sx={{
                            bgcolor: `${getStatusColor(useCase.status)}20`,
                            color: getStatusColor(useCase.status),
                            fontWeight: 500,
                          }}
                        />
                      </Box>
                    </Box>
                    
                    <Typography
                      variant="body2"
                      sx={{
                        color: '#64748B',
                        lineHeight: 1.6,
                        flexGrow: 1,
                        mb: 2,
                      }}
                    >
                      {useCase.description}
                    </Typography>

                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mt: 'auto' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Chat sx={{ fontSize: 16, color: '#64748B' }} />
                        <Typography variant="body2" sx={{ color: '#64748B' }}>
                          {useCase.messageCount} messages
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Schedule sx={{ fontSize: 16, color: '#64748B' }} />
                        <Typography variant="body2" sx={{ color: '#64748B' }}>
                          {new Date(useCase.lastUpdated).toLocaleDateString()}
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </CardActionArea>
              </Card>
            </Box>
          ))}

          {/* Add New Use Case Card */}
          <Box>
            <Card
              sx={{
                height: 280,
                display: 'flex',
                flexDirection: 'column',
                border: '2px dashed #CBD5E1',
                backgroundColor: '#F8FAFC',
                transition: 'all 0.3s ease-in-out',
                cursor: 'pointer',
                '&:hover': {
                  borderColor: '#FF6B35',
                  backgroundColor: '#FFF7ED',
                },
              }}
            >
              <CardActionArea
                onClick={handleNewUseCaseClick}
                sx={{ height: '100%', p: 0 }}
              >
                <CardContent
                  sx={{
                    p: 3,
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    textAlign: 'center',
                  }}
                >
                  <Avatar
                    sx={{
                      bgcolor: '#FF6B35',
                      width: 64,
                      height: 64,
                      mb: 2,
                    }}
                  >
                    <Add sx={{ fontSize: 32 }} />
                  </Avatar>
                  <Typography variant="h6" sx={{ fontWeight: 600, color: '#1E293B', mb: 1 }}>
                    New Use Case
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#64748B' }}>
                    Start a new conversation with the planning agent
                  </Typography>
                </CardContent>
              </CardActionArea>
            </Card>
            </Box>
        </Box>
      </Container>
    </Box>
  );
};

export default UseCaseScreen;

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
  AccountBalance,
  Business,
  Science,
  Add,
  Menu,
  AccountCircle,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { mockAccounts } from '../data/mockData';
import { Account } from '../types';

const AccountScreen: React.FC = () => {
  const navigate = useNavigate();

  const getIndustryIcon = (industry: string) => {
    switch (industry.toLowerCase()) {
      case 'investment banking':
        return <AccountBalance />;
      case 'banking':
        return <Business />;
      case 'pharmaceutical':
        return <Science />;
      default:
        return <Business />;
    }
  };

  const getIndustryColor = (industry: string) => {
    switch (industry.toLowerCase()) {
      case 'investment banking':
        return '#1E3A8A';
      case 'banking':
        return '#10B981';
      case 'pharmaceutical':
        return '#F59E0B';
      default:
        return '#64748B';
    }
  };

  const handleAccountClick = (account: Account) => {
    navigate(`/usecases/${account.id}`, { state: { account } });
  };

  const handleNewAccountClick = () => {
    // For demo purposes, we'll just show an alert
    alert('New account creation would be implemented here');
  };

  return (
    <Box sx={{ flexGrow: 1, backgroundColor: '#F8FAFC', minHeight: '100vh' }}>
      <AppBar position="static" elevation={0} sx={{ backgroundColor: 'white', borderBottom: '1px solid #E2E8F0' }}>
        <Toolbar>
          <IconButton edge="start" color="inherit" sx={{ mr: 2, color: '#64748B' }}>
            <Menu />
          </IconButton>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, color: '#1E293B', fontWeight: 600 }}>
            Use Case Delivery Agent
          </Typography>
          <IconButton color="inherit" sx={{ color: '#64748B' }}>
            <AccountCircle />
          </IconButton>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box sx={{ mb: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom sx={{ color: '#1E293B', fontWeight: 700 }}>
            Accounts
          </Typography>
          <Typography variant="body1" sx={{ color: '#64748B', fontSize: '1.125rem' }}>
            Select an account to view and manage use cases
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
          {mockAccounts.map((account) => (
            <Box key={account.id}>
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
                  onClick={() => handleAccountClick(account)}
                  sx={{ height: '100%', p: 0 }}
                >
                  <CardContent sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Avatar
                        sx={{
                          bgcolor: getIndustryColor(account.industry),
                          width: 48,
                          height: 48,
                          mr: 2,
                        }}
                      >
                        {getIndustryIcon(account.industry)}
                      </Avatar>
                      <Box sx={{ flexGrow: 1 }}>
                        <Typography variant="h6" component="h2" sx={{ fontWeight: 600, color: '#1E293B' }}>
                          {account.name}
                        </Typography>
                        <Chip
                          label={account.industry}
                          size="small"
                          sx={{
                            bgcolor: `${getIndustryColor(account.industry)}20`,
                            color: getIndustryColor(account.industry),
                            fontWeight: 500,
                            mt: 0.5,
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
                      }}
                    >
                      {account.description}
                    </Typography>
                  </CardContent>
                </CardActionArea>
              </Card>
            </Box>
          ))}

          {/* Add New Account Card */}
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
                onClick={handleNewAccountClick}
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
                    Add New Account
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#64748B' }}>
                    Create a new account to get started
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

export default AccountScreen;

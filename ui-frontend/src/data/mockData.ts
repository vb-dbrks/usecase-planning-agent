import { Account, UseCase } from '../types';

export const mockAccounts: Account[] = [
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

export const mockUseCases: UseCase[] = [
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

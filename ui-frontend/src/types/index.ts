export interface Account {
  id: string;
  name: string;
  industry: string;
  description: string;
  logo?: string;
  color?: string;
}

export interface UseCase {
  id: string;
  title: string;
  description: string;
  accountId: string;
  status: 'active' | 'completed' | 'draft';
  lastUpdated: string;
  messageCount: number;
}

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'agent';
  timestamp: string;
  type?: 'text' | 'system';
}

export interface ChatSession {
  id: string;
  useCaseId: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

# Databricks Use Case Delivery Agent UI

A professional, modern web application built with React and Node.js for managing migration planning conversations with Databricks accounts.

## Features

### 🏢 Account Management
- View all associated Databricks accounts (UBS, Leeds Building Society, Astellas)
- Professional card-based interface with industry-specific branding
- Easy navigation to account-specific use cases

### 📋 Use Case Management
- Browse existing migration use cases for each account
- View conversation status, message counts, and last updated timestamps
- Create new use cases with a single click

### 💬 Intelligent Chat Interface
- **Real-time chat with the Databricks migration planning agent**
- Professional message bubbles with timestamps
- Typing indicators and smooth animations
- Responsive design for all screen sizes
- **Connected to your deployed Databricks agent endpoint**

## Technology Stack

### Frontend
- **React 18** with TypeScript
- **Material-UI (MUI)** for professional components
- **React Router** for navigation
- **Custom Databricks Theme** with official brand colors

### Backend
- **Node.js** with Express
- **Socket.io** for real-time communication
- **RESTful API** for data management
- **Databricks Agent Integration** for intelligent chat responses
- **CORS** enabled for cross-origin requests

## Getting Started

### Prerequisites
- Node.js 16+ 
- npm or yarn
- **Databricks access token** (for agent integration)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd use-case-delivery-agent
   ```

2. **Install frontend dependencies**
   ```bash
   cd ui-frontend
   npm install
   ```

3. **Install backend dependencies**
   ```bash
   cd ../ui-backend
   npm install
   ```

### Running the Application

1. **Set up your Databricks token**
   ```bash
   export DATABRICKS_TOKEN=your_token_here
   ```
   Get your token from: https://adb-984752964297111.11.azuredatabricks.net/settings/access-tokens

2. **Start the backend server**
   ```bash
   cd ui-backend
   npm start
   ```
   The backend will run on `http://localhost:5000`

3. **Start the frontend development server**
   ```bash
   cd ui-frontend
   npm start
   ```
   The frontend will run on `http://localhost:3000`

4. **Open your browser**
   Navigate to `http://localhost:3000` to see the application

### 🤖 **Databricks Agent Integration**

The chat interface is now connected to your deployed Databricks migration planning agent! 

- **Agent Endpoint**: `https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/migration-planning-agent/invocations`
- **Features**: Real-time conversation with the planning agent, intelligent question generation, and comprehensive migration planning
- **Fallback**: If the agent is unavailable, the chat will show helpful fallback responses

## Project Structure

```
use-case-delivery-agent/
├── ui-frontend/                 # React frontend application
│   ├── src/
│   │   ├── components/         # React components
│   │   │   ├── AccountScreen.tsx
│   │   │   ├── UseCaseScreen.tsx
│   │   │   └── ChatScreen.tsx
│   │   ├── theme/             # Material-UI theme
│   │   │   └── databricksTheme.ts
│   │   ├── types/             # TypeScript type definitions
│   │   ├── data/              # Mock data
│   │   └── App.tsx
│   └── package.json
├── ui-backend/                 # Node.js backend API
│   ├── server.js              # Express server with Databricks agent integration
│   └── package.json
└── README.md
```

## Design System

### Databricks Brand Colors
- **Primary Orange**: `#FF6B35` - Databricks signature color
- **Secondary Blue**: `#1E3A8A` - Professional blue
- **Accent Yellow**: `#F59E0B` - Highlight color
- **Success Green**: `#10B981` - Status indicators
- **Background**: `#F8FAFC` - Light gray background

### Typography
- **Font Family**: Inter (Google Fonts)
- **Weights**: 300, 400, 500, 600, 700
- **Responsive sizing** with Material-UI typography scale

### Components
- **Cards**: Rounded corners (12px), subtle shadows, hover effects
- **Buttons**: Rounded (8px), no text transform, professional spacing
- **Chips**: Rounded (6px), color-coded by status/industry
- **Avatars**: Circular, color-coded backgrounds

## API Endpoints

### Accounts
- `GET /api/accounts` - Get all accounts
- `GET /api/accounts/:id` - Get specific account

### Use Cases
- `GET /api/usecases` - Get all use cases (optional accountId query)
- `GET /api/usecases/:id` - Get specific use case
- `POST /api/usecases` - Create new use case

### Chat Integration
- `POST /api/chat` - Send message to Databricks agent
  - **Request**: `{ message: string, account: string, useCase: string }`
  - **Response**: `{ response: string, account: string, useCase: string, timestamp: string }`

### WebSocket Events
- `join-chat` - Join a chat room
- `send-message` - Send a message
- `receive-message` - Receive a message

## Features in Detail

### Account Screen
- Displays 3 sample accounts with industry-specific icons and colors
- Hover animations and professional card design
- "Add New Account" card for future functionality

### Use Case Screen
- Shows existing use cases for the selected account
- Status indicators (Active, Completed, Draft)
- Message counts and last updated timestamps
- "New Use Case" card to start fresh conversations

### Chat Screen
- **Real-time messaging with Databricks agent**
- Agent and user message differentiation
- Typing indicators and loading states
- **Intelligent responses from your deployed migration planning agent**
- Responsive design for mobile and desktop

## Environment Variables

Create a `.env` file in the `ui-backend` directory:

```env
# Databricks Configuration
DATABRICKS_TOKEN=your_databricks_token_here

# Server Configuration
PORT=5000
NODE_ENV=development
```

## Troubleshooting

### Chat Not Working
1. **Check DATABRICKS_TOKEN**: Ensure the environment variable is set
2. **Verify Agent Endpoint**: Check if the Databricks agent is running
3. **Check Network**: Ensure the backend can reach the Databricks endpoint
4. **Check Logs**: Look at the backend console for error messages

### Common Issues
- **"DATABRICKS_TOKEN environment variable not set"**: Set the token as shown above
- **"Failed to get response from planning agent"**: Check your token and network connection
- **Cards not displaying properly**: Ensure all dependencies are installed

## Future Enhancements

- [ ] User authentication and authorization
- [ ] Persistent chat history with database storage
- [ ] File upload and sharing capabilities
- [ ] Advanced search and filtering
- [ ] Export conversation summaries
- [ ] Multi-language support
- [ ] Real-time collaboration features

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the ISC License - see the LICENSE file for details.

## Support

For support and questions, please contact the development team or create an issue in the repository.

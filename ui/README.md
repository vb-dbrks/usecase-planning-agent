# Usecase Delivery Planning Agent

An AI-powered chatbot for migration planning and data architecture assistance, built with React and FastAPI, designed to run on Databricks Apps.

## Features

- 🤖 **AI-Powered Chat Interface**: Interactive chat with your migration planning agent
- 📊 **Databricks Integration**: Seamlessly connects to your Databricks serving endpoints
- 🎨 **Modern React UI**: Clean, responsive interface with real-time messaging
- ⚡ **FastAPI Backend**: High-performance API with automatic documentation
- 🔒 **Secure**: Built-in Databricks authentication and security

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React UI      │    │   FastAPI       │    │  Databricks     │
│   (Frontend)    │◄──►│   (Backend)     │◄──►│  Serving        │
│                 │    │                 │    │  Endpoint       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

- Node.js 16+ and npm
- Python 3.8+
- Databricks workspace with serving endpoint configured
- Databricks CLI configured

## Quick Start

### 1. Install Dependencies

**Backend:**
```bash
pip install -r requirements.txt
```

**Frontend:**
```bash
npm install
```

### 2. Build the React App

```bash
npm run build
```

### 3. Configure Databricks Resources

Update the serving endpoint name in `app.yaml`:

```yaml
env:
  - name: SERVING_ENDPOINT_NAME
    valueFrom:
      databricks:
        resource: serving-endpoint/your-endpoint-name
```

### 4. Deploy to Databricks Apps

```bash
# Create the app
databricks apps create --config-file app.yaml

# Deploy the app
databricks apps deploy --app-id <your-app-id>
```

## Development

### Local Development

1. **Start the React development server:**
   ```bash
   npm start
   ```

2. **Start the FastAPI server:**
   ```bash
   uvicorn app:app --reload --port 8000
   ```

3. **Access the application:**
   - React UI: http://localhost:3000
   - FastAPI docs: http://localhost:8000/docs

### Project Structure

```
ui/
├── app.py                 # FastAPI backend
├── app.yaml              # Databricks Apps configuration
├── requirements.txt      # Python dependencies
├── package.json          # React dependencies
├── public/
│   ├── index.html
│   └── manifest.json
├── src/
│   ├── components/
│   │   ├── ChatInterface.js
│   │   ├── ChatInterface.css
│   │   ├── Header.js
│   │   ├── Header.css
│   │   ├── Message.js
│   │   └── Message.css
│   ├── App.js
│   ├── App.css
│   ├── index.js
│   └── index.css
└── README.md
```

## API Endpoints

### POST /api/chat
Send a message to the AI agent.

**Request:**
```json
{
  "message": "Help me plan a migration to cloud",
  "conversation_history": [
    {
      "role": "user",
      "content": "Previous message"
    }
  ]
}
```

**Response:**
```json
{
  "response": "I'll help you plan your migration...",
  "status": "success"
}
```

### GET /api/health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "usecase-delivery-planning-agent"
}
```

## Configuration

### Environment Variables

- `SERVING_ENDPOINT_NAME`: Name of your Databricks serving endpoint

### Customization

1. **Update the AI model**: Modify the serving endpoint configuration in your Databricks workspace
2. **Customize the UI**: Edit components in `ui/src/components/`
3. **Add new features**: Extend the FastAPI backend in `app.py`

## Troubleshooting

### Common Issues

1. **Build errors**: Ensure Node.js 16+ is installed
2. **API errors**: Check that your Databricks serving endpoint is running
3. **Authentication**: Verify your Databricks CLI is configured correctly

### Logs

Check Databricks Apps logs for debugging:
```bash
databricks apps logs --app-id <your-app-id>
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Check the [Databricks Apps documentation](https://docs.databricks.com/en/dev-tools/databricks-apps/)
- Review the [FastAPI documentation](https://fastapi.tiangolo.com/)
- Consult the [React documentation](https://reactjs.org/)

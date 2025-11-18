# Finsight - AI-Powered Personal Finance Assistant

This project is a comprehensive personal finance management application that leverages AI to provide intelligent insights and analytics. It utilizes a PostgreSQL database (Supabase) to store financial transactions, a FastAPI backend with LangChain AI agents for analytics and natural language processing, and a React + TypeScript frontend for intuitive user interaction.

## Features

### Backend (FastAPI + LangChain)
- **PostgreSQL Database**: Stores detailed transaction information including dates, amounts, categories, merchants, and accounts via Supabase
- **AI-Powered Analytics Agent**: Concurrent execution of 7 analytics tools (monthly summaries, spending trends, income vs expense, top categories, recent transactions, anomaly detection, budget comparison) with comprehensive AI-generated insights
- **Natural Language Query Agent**: Converts natural language questions to SQL queries and returns formatted results (e.g., "How much did I spend on groceries last month?")
- **Transaction Classifier**: Automatically categorizes uploaded transactions using LLM-based intelligent categorization
- **RESTful API**: Provides endpoints for authentication, transactions, analytics, and budget management with JWT Bearer token security
- **Performance Optimized**: ThreadPoolExecutor for parallel tool execution, ~70% latency reduction, 2-4 second dashboard generation

### Frontend (React + TypeScript + Vite)
- **Modern Dark Fintech UI**: Responsive design with custom color palette and smooth animations
- **Interactive Dashboard**: Monthly summary cards and 4 chart types (bar, line, pie) using Recharts
- **AI Chat Assistant**: Real-time chat interface for natural language transaction queries
- **CSV Upload**: Drag-and-drop transaction file upload with automatic processing
- **Budget Management**: Set total and category-specific budget limits with visual comparisons
- **Protected Routes**: React Router with authentication-based route protection
- **Token-based Auth**: Automatic JWT token injection via Axios interceptors

## Prerequisites

Before running this project locally, ensure you have the following installed:

- Python 3.9 or higher
- Node.js 18 or higher
- PostgreSQL database (via Supabase account)
- OpenAI API key
- IDE (VS Code, PyCharm, etc.)

## Installation

### Backend Setup

1. Clone this repository and navigate to the project root:
   ```bash
   cd personal_finance_assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with the following variables:
   ```env
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_anon_key
   OPENAI_API_KEY=your_openai_api_key
   SUPABSE_DB_URL=your_postgresql_connection_string
   SUPABASE_JWT_SECRET=your_jwt_secret_key
   ```

5. Start the FastAPI backend:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd finsight-frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file in the `finsight-frontend` directory:
   ```env
   VITE_SUPABASE_URL=your_supabase_url
   VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

5. Access the application at [http://localhost:5173](http://localhost:5173)

## Usage

### Getting Started
1. Access the frontend application via [http://localhost:5173](http://localhost:5173)
2. Sign up for a new account or log in with existing credentials
3. Verify your email via the Supabase authentication link (if required)

### Uploading Transactions
1. Navigate to the Upload page (`/upload`)
2. Select a CSV file containing your transaction data
3. The backend automatically categorizes transactions using AI

### Viewing Analytics
1. Navigate to the Dashboard (`/dashboard`)
2. View monthly summary cards showing income, expenses, and savings
3. Interact with 4 chart types:
   - Income vs Expense (Bar Chart)
   - Spending Over Time (Line Chart)
   - Top Categories (Pie Chart)
   - Budget Comparison (Bar Chart)
4. Read AI-generated insights for each analytics section
5. See 10 most recent transactions

### Natural Language Queries
1. Use the chat assistant on the dashboard
2. Ask questions in plain English:
   - "How much did I spend on dining last month?"
   - "Show me all transactions over $100"
   - "What are my top 5 expense categories?"
3. Receive formatted results with AI summaries

### Managing Budget
1. Navigate to the Budget page (`/budget`)
2. Enter your total monthly budget
3. Set category-specific limits (Groceries, Dining, Transportation, etc.)
4. Dashboard automatically updates with budget vs actual comparisons

## API Endpoints

Use the provided API endpoints to interact with the backend:

### Authentication (`/auth`)
- `POST /auth/signup` - Register a new user
- `POST /auth/login` - Login and receive JWT access token
- `GET /auth/auth-check` - Check if user is authenticated

### Transactions (`/transactions`)
- `POST /transactions/upload-transactions` - Upload CSV file with transactions
- `GET /transactions/query?nl_query={query}` - Natural language query (e.g., "How much did I spend on groceries last month?")

### Analytics (`/analytics`)
- `GET /analytics/dashboard` - Get complete dashboard analytics with AI insights (includes all 7 analytics tools)

### Budget (`/budget`)
- `POST /budget` - Create or update budget with category-specific limits
- `GET /budget` - Retrieve current budget settings

API documentation (Swagger UI) is available at [http://localhost:8000/docs](http://localhost:8000/docs)

## Project Structure

### Backend
```
backend/
--- agents/          # LangChain AI agents (analytics, SQL NL, classifier)
--- tools/           # Analytics tools (7 tools), anomaly detection
--- api/
-- --- routes/       # FastAPI route handlers
-- --- models/       # Pydantic schemas
--- db/              # Database configuration
--- utils/           # Reusable helper functions
--- main.py         # FastAPI application entry point
```

### Frontend
```
finsight-frontend/
--- src/
-- --- api/         # Axios instance with auth interceptor
-- --- auth/        # Authentication context provider
-- --- components/  # Reusable UI components (Navbar, Charts, Tables)
-- --- pages/       # Route pages (Dashboard, Upload, Budget, Auth)
-- --- types.ts     # TypeScript type definitions
-- --- App.tsx      # Main app with routing
--- package.json
```

## Architecture

The application follows a modern full-stack architecture:

Frontend: (React) -> REST API -> Backend: (FastAPI) -> AI Agents (LangChain) -> Database (Supabase/PostgreSQL)

- Frontend makes authenticated requests using JWT Bearer tokens
- Backend processes requests and invokes AI agents when needed
- Analytics Agent runs 7 tools concurrently for fast dashboard generation
- SQL NL Agent converts natural language to safe SQL queries
- Transaction Classifier uses LLM for intelligent categorization
- All data persists in PostgreSQL via Supabase

## License

This project is for educational and personal use.

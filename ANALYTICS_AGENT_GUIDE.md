# Analytics Agent Guide

## Overview

The analytics agent is built using **LangGraph** (just like your NL2SQL agent) and provides comprehensive financial analytics with **JSON outputs** ready for your React frontend.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Analytics Agent (LangGraph)              │
│                                                              │
│  ┌──────────┐     ┌────────────┐     ┌──────────────┐     │
│  │  Agent   │────▶│   Tools    │────▶│  Insights    │     │
│  │  (LLM)   │◀────│  Executor  │◀────│  Generator   │     │
│  └──────────┘     └────────────┘     └──────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    JSON Output for React
```

## Available Analytics Tools

### 1. **Monthly Summary Tool** (`monthly_summary_tool`)
- **Output**: Income, expenses, and net savings per month
- **Chart Type**: Table or multi-bar chart
- **JSON Format**:
```json
{
  "status": "success",
  "data": [
    {
      "month": "2024-01",
      "income": 5000.00,
      "expenses": 3200.00,
      "net_savings": 1800.00
    }
  ],
  "metadata": {
    "total_months": 6,
    "avg_income": 5200.00,
    "avg_expenses": 3100.00,
    "avg_savings": 2100.00
  }
}
```

### 2. **Spending Over Time Tool** (`spending_over_time_tool`)
- **Output**: Monthly expense trends
- **Chart Type**: Line chart
- **JSON Format**:
```json
{
  "status": "success",
  "chart_type": "line",
  "data": {
    "labels": ["2024-01", "2024-02", "2024-03"],
    "datasets": [{
      "label": "Monthly Expenses",
      "data": [3200.00, 3100.00, 3400.00]
    }]
  },
  "metadata": {
    "total_spending": 9700.00,
    "avg_monthly_spending": 3233.33,
    "max_month": "2024-03",
    "max_amount": 3400.00
  }
}
```

### 3. **Income vs Expense Tool** (`income_vs_expense_tool`)
- **Output**: Dual comparison of income and expenses
- **Chart Type**: Dual bar chart
- **JSON Format**:
```json
{
  "status": "success",
  "chart_type": "bar",
  "data": {
    "labels": ["2024-01", "2024-02", "2024-03"],
    "datasets": [
      {
        "label": "Income",
        "data": [5000.00, 5200.00, 5100.00],
        "type": "bar"
      },
      {
        "label": "Expenses",
        "data": [3200.00, 3100.00, 3400.00],
        "type": "bar"
      }
    ]
  },
  "metadata": {
    "total_income": 15300.00,
    "total_expenses": 9700.00,
    "net_savings": 5600.00
  }
}
```

### 4. **Top Categories Tool** (`top_categories_tool`)
- **Output**: Top spending categories with percentages
- **Chart Type**: Pie chart or bar chart
- **JSON Format**:
```json
{
  "status": "success",
  "chart_type": "pie",
  "data": {
    "labels": ["Groceries", "Dining", "Transportation"],
    "datasets": [{
      "label": "Spending by Category",
      "data": [800.00, 500.00, 300.00]
    }]
  },
  "breakdown": [
    {
      "category": "Groceries",
      "amount": 800.00,
      "percentage": 50.00,
      "transaction_count": 15
    }
  ],
  "metadata": {
    "total_spending": 1600.00,
    "total_categories": 3,
    "top_category": "Groceries",
    "top_category_amount": 800.00
  }
}
```

### 5. **Recent Transactions Tool** (`recent_transactions_tool`)
- **Output**: Last 10 transactions
- **Chart Type**: Table
- **JSON Format**:
```json
{
  "status": "success",
  "data": [
    {
      "id": "txn-123",
      "date": "2024-03-15",
      "description": "Whole Foods Market",
      "merchant": "Whole Foods",
      "category": "Groceries",
      "amount": 85.50,
      "type": "debit",
      "account": "Checking"
    }
  ],
  "metadata": {
    "total_transactions": 10
  }
}
```

### 6. **Anomaly Detection Tool** (`anomaly_detection_tool`)
- **Output**: Unusual transactions detected via Z-score analysis
- **Chart Type**: Table with highlights
- **JSON Format**:
```json
{
  "status": "success",
  "data": [
    {
      "id": "txn-456",
      "date": "2024-03-10",
      "description": "Expensive purchase",
      "merchant": "Electronics Store",
      "category": "Shopping",
      "amount": 1500.00,
      "z_score": 3.2,
      "category_avg": 150.00,
      "deviation": 1350.00
    }
  ],
  "metadata": {
    "total_anomalies": 5,
    "threshold_used": 2.5,
    "total_transactions_analyzed": 250
  }
}
```

### 7. **Budget Comparison Tool** (`budget_comparison_tool`)
- **Output**: Budget vs actual spending comparison
- **Chart Type**: Progress bars or comparison chart
- **JSON Format**:
```json
{
  "status": "success",
  "data": [
    {
      "category": "Groceries",
      "budget": 500.00,
      "actual": 450.00,
      "difference": 50.00,
      "percentage_used": 90.00,
      "status": "under"
    },
    {
      "category": "Dining",
      "budget": 300.00,
      "actual": 350.00,
      "difference": -50.00,
      "percentage_used": 116.67,
      "status": "over"
    }
  ],
  "metadata": {
    "total_budget": 1500.00,
    "total_spent": 1400.00,
    "remaining_budget": 100.00,
    "overall_percentage_used": 93.33,
    "month": "2024-03",
    "categories_over_budget": 1
  }
}
```

### 8. **AI Insights Generation Tool** (`generate_insights_tool`)
- **Output**: Natural language insights about the data
- **JSON Format**:
```json
{
  "status": "success",
  "insights": "Your spending on dining has increased by 20% compared to last month. Consider reducing takeout orders to stay within budget. Your grocery spending is stable and well-managed.",
  "chart_type": "top_categories"
}
```

## How to Use the Analytics Agent

### Basic Usage

```python
from backend.agents.analytics_agent import generate_analytics

# Generate all analytics for a user
result = generate_analytics(
    user_id="user-123",
    analytics_types=None,  # None = all analytics
    budget_data={
        "Groceries": 500.0,
        "Dining": 300.0,
        "Transportation": 200.0
    },
    months=6
)

# Result structure:
{
    "status": "success",
    "user_id": "user-123",
    "analytics": {
        "monthly_summary": {...},
        "spending_trends": {...},
        "income_vs_expense": {...},
        "top_categories": {...},
        "recent_transactions": {...},
        "anomalies": {...},
        "budget_comparison": {...}
    },
    "summary": "AI-generated overall summary",
    "messages": [...]  # Full conversation history
}
```

### Get Dashboard Analytics (All Analytics)

```python
from backend.agents.analytics_agent import get_dashboard_analytics

# One-liner to get everything for a dashboard
result = get_dashboard_analytics(
    user_id="user-123",
    budget_data={"Groceries": 500, "Dining": 300}
)
```

### Get Specific Analytics

```python
from backend.agents.analytics_agent import get_specific_analytics

# Get only spending trends
result = get_specific_analytics(
    user_id="user-123",
    analytics_type="spending_trends",
    months=6
)
```

## Integration with React Frontend

### Example: Using Analytics in React

```typescript
// API call to backend
const fetchAnalytics = async (userId: string) => {
  const response = await fetch('/api/analytics/dashboard', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      budget_data: {
        Groceries: 500,
        Dining: 300,
        Transportation: 200
      }
    })
  });

  const data = await response.json();
  return data.analytics;
};

// React component
const Dashboard = () => {
  const [analytics, setAnalytics] = useState(null);

  useEffect(() => {
    fetchAnalytics(userId).then(setAnalytics);
  }, [userId]);

  return (
    <div>
      {/* Line Chart - Spending Over Time */}
      <LineChart data={analytics?.spending_trends?.data} />

      {/* Bar Chart - Income vs Expense */}
      <BarChart data={analytics?.income_vs_expense?.data} />

      {/* Pie Chart - Top Categories */}
      <PieChart data={analytics?.top_categories?.data} />

      {/* Table - Recent Transactions */}
      <Table data={analytics?.recent_transactions?.data} />

      {/* AI Insights */}
      <InsightsBox insights={analytics?.spending_trends?.insights} />
    </div>
  );
};
```

### Compatible Chart Libraries

The JSON format is compatible with:
- **Chart.js** (React wrapper: `react-chartjs-2`)
- **Recharts**
- **Victory**
- **Nivo**

Example with Chart.js:

```jsx
import { Line } from 'react-chartjs-2';

const SpendingTrendChart = ({ analytics }) => {
  const chartData = analytics.spending_trends.data;

  return (
    <div>
      <Line data={chartData} />
      <p className="insights">{analytics.spending_trends.insights}</p>
    </div>
  );
};
```

## Backend API Endpoint Example

```python
from fastapi import APIRouter
from backend.agents.analytics_agent import get_dashboard_analytics

router = APIRouter()

@router.post("/analytics/dashboard")
async def get_analytics_dashboard(request: AnalyticsRequest):
    """Get all analytics for dashboard"""
    result = get_dashboard_analytics(
        user_id=request.user_id,
        budget_data=request.budget_data
    )
    return result

@router.post("/analytics/specific")
async def get_specific_analytic(request: SpecificAnalyticsRequest):
    """Get specific analytics type"""
    from backend.agents.analytics_agent import get_specific_analytics

    result = get_specific_analytics(
        user_id=request.user_id,
        analytics_type=request.analytics_type,
        budget_data=request.budget_data,
        months=request.months or 6
    )
    return result
```

## LangGraph Workflow

The agent follows this workflow:

1. **START** → Agent receives user request
2. **Agent Node** → LLM decides which tools to call
3. **Tools Node** → Executes analytics tools in parallel/sequence
4. **Should Continue?**
   - If more tools needed → back to Agent
   - If complete → END
5. **END** → Returns comprehensive analytics with insights

## Budget Integration

### Setting Up Budgets

```python
# Budget data structure
budget_data = {
    "Groceries": 500.00,
    "Dining": 300.00,
    "Transportation": 200.00,
    "Entertainment": 150.00,
    "Utilities": 250.00,
    "Shopping": 200.00
}

# Pass to analytics agent
result = generate_analytics(
    user_id="user-123",
    budget_data=budget_data
)
```

### Real-time Budget Updates

The agent automatically:
- Compares current month spending vs budget
- Calculates percentage used for each category
- Flags categories over budget
- Provides budget-aware insights

## Testing the Agent

Run the built-in test:

```bash
cd /Users/pavithrak/PersonalProjects/personal_finance_assistant
python -m backend.agents.analytics_agent
```

This will test the agent with sample data and display all analytics.

## Key Features

1. **LangGraph-based**: Same architecture as your NL2SQL agent
2. **JSON outputs**: All data ready for React charts
3. **AI Insights**: Automatic insights for every chart
4. **Budget-aware**: Real-time budget comparison
5. **Anomaly detection**: Statistical detection of unusual spending
6. **Flexible**: Request all or specific analytics
7. **Memory**: Conversation history persisted per user
8. **Extensible**: Easy to add new analytics tools

## Next Steps

1. Create API endpoints in your FastAPI backend
2. Integrate with React frontend
3. Add budget CRUD operations (create, read, update budgets)
4. Optionally add more analytics:
   - Merchant analysis
   - Payment method breakdown
   - Recurring transaction detection
   - Savings goals tracking

## Architecture & Code Organization

### File Structure
```
backend/
├── agents/
│   └── analytics_agent.py          # LangGraph workflow orchestration
├── tools/
│   ├── analytics_tool.py           # 7 analytics tools (refactored)
│   └── anomaly_tool.py             # Anomaly detection (Z-score)
├── utils/
│   └── analytics_utils.py          # Reusable helper functions
└── api/models/
    └── schemas.py                  # Budget & Transaction schemas
```

### Helper Utilities ([backend/utils/analytics_utils.py](backend/utils/analytics_utils.py))

The analytics tools are built on top of reusable utilities:

**Data Fetching:**
- `get_user_transactions(user_id, months)` - Fetch with automatic date filtering
- `filter_by_type(df, type)` - Filter by transaction type

**Aggregation:**
- `summarize_monthly(df)` - Monthly income/expense/savings
- `aggregate_by_category(df, limit)` - Category spending with percentages

**Formatting:**
- `chart_json(type, labels, datasets, metadata)` - Chart data formatting
- `error_json(message)` - Error responses
- `empty_response_json(type, message)` - Empty data responses
- `format_transactions_list(df, limit)` - Transaction formatting
- `format_category_breakdown(summary)` - Category details

**Statistics:**
- `calculate_statistics(values)` - Mean, median, std, min, max, sum

### Benefits of Refactored Architecture

1. **DRY Principle**: No code duplication across tools
2. **Maintainability**: Change logic once, affects all tools
3. **Testability**: Test utilities independently
4. **Consistency**: All tools use same formatting/error handling
5. **Extensibility**: Easy to add new analytics tools

## Files Created/Modified

- ✓ `backend/tools/analytics_tool.py` - 7 analytics tools (refactored)
- ✓ `backend/agents/analytics_agent.py` - LangGraph agent
- ✓ `backend/api/models/schemas.py` - Added Budget schema
- ✓ `backend/utils/analytics_utils.py` - Reusable helper functions
- ✓ `backend/tools/anomaly_tool.py` - Anomaly detection tool

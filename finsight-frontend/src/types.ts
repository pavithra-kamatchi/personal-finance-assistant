// Type definitions for Finsight frontend

export interface User {
  id: string;
  email: string;
  name?: string;
}

// Transaction data structure
export interface Transaction {
  id: string;
  date: string;
  description: string;
  category: string;
  amount: number;
  type: 'income' | 'expense';
}

// Monthly summary data structure
export interface MonthlySummary {
  total_income: number;
  total_expenses: number;
  net_savings: number;
  month: string;
}

// Chart data point structure
export interface ChartDataPoint {
  name: string;
  value: number;
  income?: number;
  expense?: number;
  actual?: number;
  budget?: number;
  [key: string]: string | number | undefined;
}

// Dashboard data structure
export interface DashboardData {
  monthly_summary: MonthlySummary;
  income_vs_expense: ChartDataPoint[];
  spending_over_time: ChartDataPoint[];
  top_categories: ChartDataPoint[];
  budget_comparison: ChartDataPoint[];
  insights: {
    income_vs_expense?: string;
    spending_trends?: string;
    top_categories?: string;
    budget_comparison?: string;
    overall?: string;
  };
}

// Budget category structure
export interface BudgetCategory {
  category: string;
  limit: number;
}

// Budget request structure
export interface BudgetRequest {
  total_budget: number;
  categories: BudgetCategory[];
}

// Chat message structure
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

// AI Ask request structure
export interface AskRequest {
  query: string;
}

// AI Ask response structure
export interface AskResponse {
  answer: string;
}

// Authentication context structure
export interface AuthContextType {
  user: User | null;
  token: string | null;
  signOut: () => void;
  setUser: (user: User | null) => void;
  setToken: (token: string | null) => void;
}

// Dashboard Page
// Main analytics view with charts, insights, and recent transactions
// Fetches data from GET /dashboard endpoint

import React, { useState, useEffect } from 'react';
import axios from '../api/axios';
import Navbar from '../components/Navbar';
import ChartCard from '../components/ChartCard';
import ChatBar from '../components/ChatBar';
import LoadingSpinner from '../components/LoadingSpinner';
import TransactionsTable from '../components/TransactionsTable';
import type { Transaction } from '../types';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface DashboardData {
  monthly_summary?: {
    total_income: number;
    total_expenses: number;
    net_savings: number;
    month: string;
  };
  income_vs_expense?: any[];
  spending_over_time?: any[];
  top_categories?: any[];
  budget_comparison?: any[];
  recent_transactions?: Transaction[];
  insights?: {
    income_vs_expense?: string;
    spending_trends?: string;
    top_categories?: string;
    budget_comparison?: string;
    overall?: string;
  };
}

const Dashboard: React.FC = () => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchDashboard();
  }, []);

  const fetchDashboard = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/analytics/dashboard');

      // Backend returns { status: "success", data: { analytics: {...} } }
      if (response.data.status === 'success' && response.data.data) {
        const analyticsData = response.data.data.analytics || response.data.data;

        // Helper function to transform chart data to Recharts format
        const transformChartData = (chartData: any) => {
          if (!chartData?.data) return [];
          const { labels, datasets } = chartData.data;
          if (!labels || !datasets) return [];

          // Transform to Recharts format: [{name: "Jan", value1: 100, value2: 200}, ...]
          return labels.map((label: string, index: number) => {
            const point: any = { name: label };
            datasets.forEach((dataset: any) => {
              const key = dataset.label.toLowerCase().replace(/\s+/g, '_');
              point[key] = dataset.data[index];
            });
            return point;
          });
        };

        // Helper function to transform pie chart data to Recharts format
        const transformPieData = (chartData: any) => {
          if (!chartData?.data) return [];
          const { labels, datasets } = chartData.data;
          if (!labels || !datasets || datasets.length === 0) return [];

          // Transform to Recharts format: [{name: "Food", value: 500}, ...]
          return labels.map((label: string, index: number) => ({
            name: label,
            value: datasets[0].data[index]
          }));
        };

        // Extract data from the analytics object
        const processedData: DashboardData = {
          monthly_summary: analyticsData.monthly_summary?.data,
          income_vs_expense: transformChartData(analyticsData.income_vs_expense),
          spending_over_time: transformChartData(analyticsData.spending_trends),
          top_categories: transformPieData(analyticsData.top_categories),
          budget_comparison: transformChartData(analyticsData.budget_comparison),
          recent_transactions: analyticsData.recent_transactions?.data,
          insights: response.data.data.insights,
        };

        setDashboardData(processedData);
      } else {
        setDashboardData(response.data);
      }
      setError('');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load dashboard');
    } finally {
      setLoading(false);
    }
  };

  // Colors for charts using brand palette
  const COLORS = ['#601EF9', '#2A5FFF', '#45E3E0', '#1EB980', '#E5484D', '#0A2540'];

  if (loading) {
    return (
      <>
        <Navbar />
        <LoadingSpinner message="Loading your dashboard..." />
      </>
    );
  }

  if (error) {
    return (
      <>
        <Navbar />
        <div className="dashboard-container">
          <div className="error-message">{error}</div>
          <button className="btn btn-primary" onClick={fetchDashboard}>
            Retry
          </button>
        </div>
      </>
    );
  }

  const summary = dashboardData?.monthly_summary;

  return (
    <>
      <Navbar />
      <div className="dashboard-container">
        <h1>Financial Dashboard</h1>

        {/* Monthly Summary Cards */}
        {summary && summary.total_income !== undefined && (
          <div className="summary-cards">
            <div className="summary-card">
              <h3>Total Income</h3>
              <p className="amount positive">${summary.total_income.toFixed(2)}</p>
            </div>
            <div className="summary-card">
              <h3>Total Expenses</h3>
              <p className="amount negative">${summary.total_expenses.toFixed(2)}</p>
            </div>
            <div className="summary-card">
              <h3>Net Savings</h3>
              <p className={`amount ${summary.net_savings >= 0 ? 'positive' : 'negative'}`}>
                ${summary.net_savings.toFixed(2)}
              </p>
            </div>
          </div>
        )}

        {/* Charts Grid */}
        <div className="charts-grid">
          {/* Income vs Expense Chart */}
          {dashboardData?.income_vs_expense && dashboardData.income_vs_expense.length > 0 && (
            <ChartCard
              title="Income vs Expense"
              insight={dashboardData.insights?.income_vs_expense}
            >
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={dashboardData.income_vs_expense}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#D3D7DF" />
                  <XAxis dataKey="name" stroke="#F5F7FA" />
                  <YAxis stroke="#F5F7FA" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0A2540', border: '1px solid #2A5FFF' }}
                  />
                  <Legend />
                  <Bar dataKey="income" fill="#1EB980" />
                  <Bar dataKey="expenses" fill="#E5484D" />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          )}

          {/* Spending Over Time Chart */}
          {dashboardData?.spending_over_time && dashboardData.spending_over_time.length > 0 && (
            <ChartCard
              title="Spending Over Time"
              insight={dashboardData.insights?.spending_trends}
            >
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={dashboardData.spending_over_time}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#D3D7DF" />
                  <XAxis dataKey="name" stroke="#F5F7FA" />
                  <YAxis stroke="#F5F7FA" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0A2540', border: '1px solid #2A5FFF' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="monthly_expenses" stroke="#601EF9" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          )}

          {/* Top Categories Chart */}
          {dashboardData?.top_categories && dashboardData.top_categories.length > 0 && (
            <ChartCard
              title="Top Spending Categories"
              insight={dashboardData.insights?.top_categories}
            >
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={dashboardData.top_categories}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={(entry) => entry.name}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {dashboardData.top_categories.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0A2540', border: '1px solid #2A5FFF' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </ChartCard>
          )}

          {/* Budget Comparison Chart */}
          {dashboardData?.budget_comparison && dashboardData.budget_comparison.length > 0 && (
            <ChartCard
              title="Budget vs Actual Spending"
              insight={dashboardData.insights?.budget_comparison}
            >
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={dashboardData.budget_comparison}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#D3D7DF" />
                  <XAxis dataKey="name" stroke="#F5F7FA" />
                  <YAxis stroke="#F5F7FA" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0A2540', border: '1px solid #2A5FFF' }}
                  />
                  <Legend />
                  <Bar dataKey="budget" fill="#2A5FFF" />
                  <Bar dataKey="actual" fill="#601EF9" />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          )}
        </div>

        {!dashboardData && !loading && (
          <div className="no-data-message">
            <h3>No data available</h3>
            <p>Upload your transaction CSV to get started!</p>
          </div>
        )}

        {/* Overall Financial Insights */}
        {dashboardData?.insights?.overall && (
          <div className="insights-section">
            <h2>Overall Financial Summary</h2>
            <div className="insights-card">
              <p className="insights-content">{dashboardData.insights.overall}</p>
            </div>
          </div>
        )}

        {/* Recent Transactions Table */}
        {dashboardData?.recent_transactions && dashboardData.recent_transactions.length > 0 && (
          <div className="recent-transactions-section">
            <h2>Recent Transactions</h2>
            <TransactionsTable transactions={dashboardData.recent_transactions} />
          </div>
        )}
      </div>

      <ChatBar />
    </>
  );
};

export default Dashboard;

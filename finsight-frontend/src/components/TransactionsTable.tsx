// Transactions Table Component
// Displays transaction data in a scrollable table with sticky header

import React from 'react';
import type { Transaction } from '../types';

interface TransactionsTableProps {
  transactions: Transaction[];
}

const TransactionsTable: React.FC<TransactionsTableProps> = ({ transactions }) => {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
  };

  const formatAmount = (amount: number, type: string) => {
    const formatted = Math.abs(amount).toFixed(2);
    return type === 'expense' ? `-$${formatted}` : `$${formatted}`;
  };

  if (!transactions || transactions.length === 0) {
    return <p className="no-data">No transactions to display</p>;
  }

  return (
    <div className="transactions-table-container">
      <table className="transactions-table">
        <thead>
          <tr>
            <th>Date</th>
            <th>Description</th>
            <th>Category</th>
            <th>Amount</th>
            <th>Type</th>
          </tr>
        </thead>
        <tbody>
          {transactions.map((transaction) => (
            <tr key={transaction.id}>
              <td>{formatDate(transaction.date)}</td>
              <td>{transaction.description}</td>
              <td>
                <span className="category-badge">{transaction.category}</span>
              </td>
              <td className={transaction.type === 'expense' ? 'expense' : 'income'}>
                {formatAmount(transaction.amount, transaction.type)}
              </td>
              <td>
                <span className={`type-badge ${transaction.type}`}>
                  {transaction.type}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default TransactionsTable;

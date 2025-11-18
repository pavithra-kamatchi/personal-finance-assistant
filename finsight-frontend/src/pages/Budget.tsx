// Budget Page
// Set monthly budget and category-specific limits
// Sends to POST /budget endpoint

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from '../api/axios';
import Navbar from '../components/Navbar';

interface BudgetCategory {
  category: string;
  limit: number;
}

const Budget: React.FC = () => {
  const navigate = useNavigate();
  const [totalBudget, setTotalBudget] = useState('');
  const [categories, setCategories] = useState<BudgetCategory[]>([
    { category: 'Food', limit: 0 },
    { category: 'Transportation', limit: 0 },
    { category: 'Entertainment', limit: 0 },
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleAddCategory = () => {
    setCategories([...categories, { category: '', limit: 0 }]);
  };

  const handleRemoveCategory = (index: number) => {
    setCategories(categories.filter((_, i) => i !== index));
  };

  const handleCategoryChange = (index: number, field: keyof BudgetCategory, value: string) => {
    const updated = [...categories];
    if (field === 'limit') {
      updated[index][field] = parseFloat(value) || 0;
    } else {
      updated[index][field] = value;
    }
    setCategories(updated);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    const budget = parseFloat(totalBudget);
    if (!budget || budget <= 0) {
      setError('Please enter a valid total budget');
      return;
    }

    // Filter out empty categories
    const validCategories = categories.filter((c) => c.category.trim() && c.limit > 0);

    if (validCategories.length === 0) {
      setError('Please add at least one category with a limit');
      return;
    }

    setLoading(true);

    try {
      // Call backend budget endpoint
      const response = await axios.post('/budget', {
        total_budget: budget,
        categories: validCategories,
      });

      if (response.status === 200) {
        setSuccess('Budget settings saved successfully!');

        // Redirect to dashboard after 2 seconds
        setTimeout(() => {
          navigate('/dashboard');
        }, 2000);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save budget. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Navbar />
      <div className="budget-container">
        <div className="budget-card">
          <h2>Set Your Budget</h2>
          <p className="budget-subtitle">Define your monthly budget and category limits</p>

          {error && <div className="error-message">{error}</div>}
          {success && <div className="success-message">{success}</div>}

          <form onSubmit={handleSubmit}>
            {/* Total Budget */}
            <div className="form-group">
              <label htmlFor="totalBudget">Total Monthly Budget</label>
              <div className="input-with-prefix">
                <span className="prefix">$</span>
                <input
                  type="number"
                  id="totalBudget"
                  value={totalBudget}
                  onChange={(e) => setTotalBudget(e.target.value)}
                  placeholder="5000"
                  step="0.01"
                  min="0"
                  disabled={loading}
                  required
                />
              </div>
            </div>

            {/* Category Budgets */}
            <div className="categories-section">
              <h3>Category Budgets</h3>
              {categories.map((cat, index) => (
                <div key={index} className="category-row">
                  <input
                    type="text"
                    placeholder="Category name"
                    value={cat.category}
                    onChange={(e) => handleCategoryChange(index, 'category', e.target.value)}
                    disabled={loading}
                    className="category-input"
                  />
                  <div className="input-with-prefix">
                    <span className="prefix">$</span>
                    <input
                      type="number"
                      placeholder="Limit"
                      value={cat.limit || ''}
                      onChange={(e) => handleCategoryChange(index, 'limit', e.target.value)}
                      step="0.01"
                      min="0"
                      disabled={loading}
                      className="limit-input"
                    />
                  </div>
                  <button
                    type="button"
                    className="btn-remove"
                    onClick={() => handleRemoveCategory(index)}
                    disabled={loading}
                  >
                    ✕
                  </button>
                </div>
              ))}

              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleAddCategory}
                disabled={loading}
              >
                + Add Category
              </button>
            </div>

            <button type="submit" className="btn btn-primary btn-full" disabled={loading}>
              {loading ? 'Saving...' : 'Save Budget'}
            </button>
          </form>
        </div>
      </div>
    </>
  );
};

export default Budget;

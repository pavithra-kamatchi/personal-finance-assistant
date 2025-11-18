// Landing Page
// First page users see with hero section and auth buttons

import React from 'react';
import { useNavigate } from 'react-router-dom';

const Landing: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="landing-page">
      <div className="hero-section">
        <h1 className="hero-title">Finsight</h1>
        <p className="hero-subtitle">Your Personal Finance Intelligence Platform</p>
        <p className="hero-description">
          Track expenses, analyze spending patterns, and make smarter financial decisions with AI-powered insights.
        </p>
        <div className="hero-buttons">
          <button className="btn btn-primary" onClick={() => navigate('/signup')}>
            Get Started
          </button>
          <button className="btn btn-secondary" onClick={() => navigate('/login')}>
            Login
          </button>
        </div>
      </div>
    </div>
  );
};

export default Landing;

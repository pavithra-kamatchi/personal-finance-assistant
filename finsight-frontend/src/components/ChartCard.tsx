// Chart Card Component
// Wrapper for charts with title and insights display

import React from 'react';

interface ChartCardProps {
  title: string;
  children: React.ReactNode;
  insight?: string;
}

const ChartCard: React.FC<ChartCardProps> = ({ title, children, insight }) => {
  return (
    <div className="chart-card">
      <h3 className="chart-title">{title}</h3>
      <div className="chart-content">{children}</div>
      {insight && (
        <div className="chart-insight">
          <p>{insight}</p>
        </div>
      )}
    </div>
  );
};

export default ChartCard;

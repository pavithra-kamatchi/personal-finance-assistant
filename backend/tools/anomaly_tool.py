from langchain_core.tools import tool
from backend.tools.retriever_tool import fetch_user_transactions
import json
import pandas as pd


#detect anomalous transactions using Z-score to flag transactions that are unusually high compared to user's typical spending
@tool
def anomaly_detection_tool(user_id: str, std_threshold: float = 2.5) -> str:
    try:
        df = fetch_user_transactions(user_id)

        if df.empty or len(df) < 10:
            return json.dumps({
                "status": "success",
                "data": [],
                "message": "Insufficient data for anomaly detection"
            })

        # Filter expenses only
        expense_df = df[df['type'] == 'debit'].copy()

        if expense_df.empty:
            return json.dumps({
                "status": "success",
                "data": []
            })

        # Calculate Z-scores for each category
        anomalies = []

        for category in expense_df['category'].unique():
            cat_df = expense_df[expense_df['category'] == category]

            if len(cat_df) < 3:
                continue

            amounts = cat_df['transaction_amount']
            mean = amounts.mean()
            std = amounts.std()

            if std == 0:
                continue

            # Calculate Z-scores
            cat_df['z_score'] = (amounts - mean) / std

            # Flag anomalies
            anomalous = cat_df[cat_df['z_score'].abs() > std_threshold]

            for _, row in anomalous.iterrows():
                anomalies.append({
                    "id": row['id'],
                    "date": row['transaction_date'].strftime('%Y-%m-%d'),
                    "description": row['description'],
                    "merchant": row['merchant'],
                    "category": row['category'],
                    "amount": round(float(row['transaction_amount']), 2),
                    "z_score": round(float(row['z_score']), 2),
                    "category_avg": round(float(mean), 2),
                    "deviation": round(float(row['transaction_amount'] - mean), 2)
                })

        # Sort by z_score (highest first)
        anomalies.sort(key=lambda x: abs(x['z_score']), reverse=True)

        return json.dumps({
            "status": "success",
            "data": anomalies,
            "metadata": {
                "total_anomalies": len(anomalies),
                "threshold_used": std_threshold,
                "total_transactions_analyzed": len(expense_df)
            }
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        }, indent=2)

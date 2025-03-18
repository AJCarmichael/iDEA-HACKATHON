import pandas as pd
import json
import os
from collections import defaultdict

# Load CSV data
df = pd.read_csv('businessMan3.csv')
def process_transactions(df):
    customer_profiles = defaultdict(lambda: {'total_spent': 0, 'total_income': 0, 'categories': set(),
                                             'income_sources': set(), 'cash_usage': 0, 'recurring_payments': [],
                                             'transactions': []})
    for _, row in df.iterrows():
        cid = row['Customer ID']
        amount = row['Amount (INR)']
        is_cash = str(row['Cash Indicator']).strip().lower() == 'yes'
        description = str(row['Description']).strip()
        transaction_type = str(row['Transaction Type']).strip().lower()
        
        if transaction_type == 'debit':
            customer_profiles[cid]['total_spent'] += amount
            customer_profiles[cid]['categories'].add(description)
            if is_cash:
                customer_profiles[cid]['cash_usage'] += amount
        elif transaction_type == 'credit':
            customer_profiles[cid]['total_income'] += amount
            customer_profiles[cid]['income_sources'].add(description)
        
        customer_profiles[cid]['transactions'].append(amount)
        
        # Identify recurring payments (e.g., monthly rent, fees, etc.)
        if 'rent' in description.lower() or 'fees' in description.lower():
            customer_profiles[cid]['recurring_payments'].append(amount)
    
    return customer_profiles

customer_data = process_transactions(df)

# Ensure metadata directory exists
os.makedirs('metadata', exist_ok=True)

for cid, data in customer_data.items():
    avg_spent = data['total_spent'] // 6  # Assuming 3 months of data
    avg_income = data['total_income'] // 6
    cash_percent = (data['cash_usage'] / data['total_spent']) * 100 if data['total_spent'] else 0
    
    profile = {
        'Customer ID': cid,
        'Profile Type': 'Student' if avg_income < 10000 else 'Professional',
        'Average Monthly Spending': f'{avg_spent}',
        'Average Monthly Income': f'{avg_income}',
        'Spending Categories': ', '.join(data['categories']),
        'Income Sources': ', '.join(data['income_sources']),
        'Cash Usage': f'{cash_percent:.2f}%',
        'Recurring Payments': f'{sum(data["recurring_payments"])} monthly' if data['recurring_payments'] else 'None',
        'Risk Indicators': 'Low' if avg_spent < 20000 else 'Medium',
        'Behavioral Patterns': 'Frequent small transactions' if avg_spent < 15000 else 'Occasional large transactions'
    }
    
    # Save metadata as JSON
    with open(f'metadata/{cid}_metadata.json', 'w') as f:
        json.dump(profile, f, indent=4)
    
    print(f"Saved metadata for {cid} at metadata/{cid}_metadata.json")

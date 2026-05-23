"""
Generate Diabetic Retinopathy Dataset - Simple Format
10,000 patient records with essential columns only
"""

import pandas as pd
import numpy as np
import os

def create_simplified_dataset():
    """Create simplified diabetic retinopathy dataset with 10,000 records"""
    
    np.random.seed(42)
    n_records = 10000
    
    # Generate data
    data = {
        'Image': [f'IMG_{str(i+1).zfill(5)}.jpg' for i in range(n_records)],
        'Patient_ID': np.random.randint(10000, 99999, n_records),
        'Age': np.random.randint(25, 80, n_records),
        'Years_of_Diabetes': np.random.uniform(0.5, 30, n_records).round(1),
        'Gender': np.random.choice(['Male', 'Female'], n_records),
        'HbA1c_Level': np.random.uniform(4.5, 13, n_records).round(1),
        'Severity': None
    }
    
    # Assign severity based on HbA1c and years of diabetes
    severity_list = []
    for i in range(n_records):
        hba1c = data['HbA1c_Level'][i]
        years = data['Years_of_Diabetes'][i]
        
        # Logic: Higher HbA1c and longer duration = worse severity
        score = (hba1c - 4.5) / 8.5 * 50 + (years / 30) * 50
        
        if score < 15:
            severity = 'Normal'
        elif score < 30:
            severity = 'Mild'
        elif score < 50:
            severity = 'Moderate'
        elif score < 75:
            severity = 'Severe'
        else:
            severity = 'Proliferative'
        
        severity_list.append(severity)
    
    data['Severity'] = severity_list
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure unique Patient IDs
    df['Patient_ID'] = range(1, len(df) + 1)
    
    # Save to Excel
    output_path = os.path.join(os.path.dirname(__file__), 'Diabetic_Retinopathy_Dataset.xlsx')
    
    try:
        df.to_excel(output_path, sheet_name='Patient Data', index=False, engine='xlsxwriter')
    except Exception as e:
        # Fallback - save as CSV
        csv_path = output_path.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved as CSV: {csv_path}")
        return csv_path
    
    print(f"✓ Dataset created: {output_path}")
    print(f"✓ Total records: {len(df)}")
    print(f"✓ Columns: Image, Patient_ID, Age, Years_of_Diabetes, Gender, HbA1c_Level, Severity")
    print(f"\nSeverity Distribution:")
    print(df['Severity'].value_counts().sort_index())
    print(f"\nDataset preview:")
    print(df.head(10))
    
    return output_path

if __name__ == "__main__":
    create_simplified_dataset()

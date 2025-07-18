#!/usr/bin/env python3
"""
Convert Excel files to CSV format
"""

import pandas as pd
import os
import sys

def convert_excel_to_csv():
    """Convert Excel files to CSV format"""
    print("Converting Excel files to CSV...")
    print()
    
    # Convert salesdaily.xls to CSV
    print("Converting salesdaily.xls to CSV...")
    try:
        df = pd.read_excel('salesdaily.xls', engine='xlrd')
        df.to_csv('salesdaily.csv', index=False)
        print(f"✓ Successfully converted salesdaily.xls to salesdaily.csv")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df['datum'].min()} to {df['datum'].max()}")
    except Exception as e:
        print(f"✗ Error converting salesdaily.xls: {e}")
    
    print()
    
    # Convert Data_Model_IoTMLCQ_2024.xlsx to CSV
    print("Converting Data_Model_IoTMLCQ_2024.xlsx to CSV...")
    try:
        # Read Excel file (might have multiple sheets)
        excel_file = pd.ExcelFile('Data_Model_IoTMLCQ_2024.xlsx', engine='openpyxl')
        print(f"  Sheet names: {excel_file.sheet_names}")
        
        # If multiple sheets, convert each one
        if len(excel_file.sheet_names) > 1:
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel('Data_Model_IoTMLCQ_2024.xlsx', sheet_name=sheet_name, engine='openpyxl')
                # Clean sheet name for filename
                clean_sheet_name = sheet_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                csv_filename = f'Data_Model_IoTMLCQ_2024_{clean_sheet_name}.csv'
                df.to_csv(csv_filename, index=False)
                print(f"  ✓ Converted sheet '{sheet_name}' to {csv_filename}")
                print(f"    Shape: {df.shape}")
                if not df.empty:
                    print(f"    Columns: {list(df.columns)[:5]}...")
        else:
            # Single sheet
            df = pd.read_excel('Data_Model_IoTMLCQ_2024.xlsx', engine='openpyxl')
            df.to_csv('Data_Model_IoTMLCQ_2024.csv', index=False)
            print(f"  ✓ Successfully converted to Data_Model_IoTMLCQ_2024.csv")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)[:5]}...")
            
    except Exception as e:
        print(f"✗ Error converting Data_Model_IoTMLCQ_2024.xlsx: {e}")
    
    print()
    print("Conversion complete!")
    
    # List all CSV files created
    print("CSV files created:")
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            size = os.path.getsize(file)
            print(f"  - {file} ({size:,} bytes)")

if __name__ == "__main__":
    convert_excel_to_csv()

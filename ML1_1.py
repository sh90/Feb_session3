### The Scenario

"""You're building a data processing pipeline for a machine learning project. You have different data sources but they all follow a common pattern:

1. Load data
2. Validate data quality
3. Handle missing values
4. Scale features
5. Save processed data

Without inheritance, each data loader would be massive and repetitive.

### Understanding the Problem

Let's start with what you might write without inheritance:"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class CSVDataProcessor:
    """Process CSV data"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.scaler = StandardScaler()
        self.errors = []
    
    def load(self):
        """Load CSV file"""
        try:
            self.data = pd.read_csv(self.filepath)
            print(f"✓ Loaded {len(self.data)} rows")
        except Exception as e:
            self.errors.append(f"Load error: {str(e)}")
    
    def validate(self):
        """Validate data quality"""
        missing = self.data.isnull().sum().sum()
        duplicates = self.data.duplicated().sum()
        
        if missing > 0:
            self.errors.append(f"Found {missing} missing values")
        if duplicates > 0:
            self.errors.append(f"Found {duplicates} duplicate rows")
        
        if self.errors:
            raise ValueError(", ".join(self.errors))
        
        print("✓ Validation passed")
    
    def handle_missing(self):
        """Handle missing values"""
        for col in self.data.columns:
            if self.data[col].dtype in ['float64', 'int64']:
                self.data[col].fillna(self.data[col].mean(), inplace=True)
            else:
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        print("✓ Missing values handled")
    
    def scale_features(self, numeric_columns):
        """Scale numerical features"""
        self.data[numeric_columns] = self.scaler.fit_transform(
            self.data[numeric_columns]
        )
        print(f"✓ Scaled {len(numeric_columns)} features")
    
    def save(self, output_path):
        """Save processed data"""
        self.data.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")
    
    def process(self, output_path, numeric_cols):
        """Complete pipeline"""
        self.load()
        self.validate()
        self.handle_missing()
        self.scale_features(numeric_cols)
        self.save(output_path)


class JSONDataProcessor:
    """Process JSON data - 95% IDENTICAL to CSV version!"""
    
    def __init__(self, url):
        self.url = url
        self.data = None
        self.scaler = StandardScaler()
        self.errors = []
    
    def load(self):
        """Load JSON from API"""
        try:
            import requests
            response = requests.get(self.url)
            self.data = pd.DataFrame(response.json())
            print(f"✓ Loaded {len(self.data)} rows")
        except Exception as e:
            self.errors.append(f"Load error: {str(e)}")
    
    def validate(self):
        """Validate data quality - EXACT SAME CODE"""
        missing = self.data.isnull().sum().sum()
        duplicates = self.data.duplicated().sum()
        
        if missing > 0:
            self.errors.append(f"Found {missing} missing values")
        if duplicates > 0:
            self.errors.append(f"Found {duplicates} duplicate rows")
        
        if self.errors:
            raise ValueError(", ".join(self.errors))
        
        print("✓ Validation passed")
    
    def handle_missing(self):
        """Handle missing values - EXACT SAME CODE"""
        for col in self.data.columns:
            if self.data[col].dtype in ['float64', 'int64']:
                self.data[col].fillna(self.data[col].mean(), inplace=True)
            else:
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        print("✓ Missing values handled")
    
    def scale_features(self, numeric_columns):
        """Scale numerical features - EXACT SAME CODE"""
        self.data[numeric_columns] = self.scaler.fit_transform(
            self.data[numeric_columns]
        )
        print(f"✓ Scaled {len(numeric_columns)} features")
    
    def save(self, output_path):
        """Save processed data - EXACT SAME CODE"""
        self.data.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")
    
    def process(self, output_path, numeric_cols):
        """Complete pipeline - EXACT SAME CODE"""
        self.load()
        self.validate()
        self.handle_missing()
        self.scale_features(numeric_cols)
        self.save(output_path)


# Problem: 200+ lines of duplicated code!
"""

Notice how `validate()`, `handle_missing()`, `scale_features()`, `save()`, and `process()` are **identical** in both classes?

If you find a bug in validation logic, you have to fix it in two places (and later four, five, or ten places)."""

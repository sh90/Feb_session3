### The Inheritance Solution

# Extract the common logic into a parent class:


class BaseDataProcessor:
    """Parent class with shared logic for all data processors"""
    
    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
        self.errors = []
    
    def validate(self):
        """Validate data quality - ONCE"""
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
        """Handle missing values - ONCE"""
        for col in self.data.columns:
            if self.data[col].dtype in ['float64', 'int64']:
                self.data[col].fillna(self.data[col].mean(), inplace=True)
            else:
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        print("✓ Missing values handled")
    
    def scale_features(self, numeric_columns):
        """Scale numerical features - ONCE"""
        self.data[numeric_columns] = self.scaler.fit_transform(
            self.data[numeric_columns]
        )
        print(f"✓ Scaled {len(numeric_columns)} features")
    
    def save(self, output_path):
        """Save processed data - ONCE"""
        self.data.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")
    
    def load(self):
        """Abstract method - must override in child"""
        raise NotImplementedError("Subclass must implement load()")
    
    def process(self, output_path, numeric_cols):
        """Complete pipeline - ONCE"""
        self.load()
        self.validate()
        self.handle_missing()
        self.scale_features(numeric_cols)
        self.save(output_path)


class CSVDataProcessor(BaseDataProcessor):
    """Child class - only 10 lines of unique code"""
    
    def __init__(self, filepath):
        super().__init__()  # Initialize parent
        self.filepath = filepath
    
    def load(self):
        """Only the CSV-specific load logic"""
        try:
            self.data = pd.read_csv(self.filepath)
            print(f"✓ Loaded {len(self.data)} rows from CSV")
        except Exception as e:
            self.errors.append(f"CSV load error: {str(e)}")


class JSONDataProcessor(BaseDataProcessor):
    """Child class - only 10 lines of unique code"""
    
    def __init__(self, url):
        super().__init__()  # Initialize parent
        self.url = url
    
    def load(self):
        """Only the JSON-specific load logic"""
        try:
            import requests
            response = requests.get(self.url)
            self.data = pd.DataFrame(response.json())
            print(f"✓ Loaded {len(self.data)} rows from JSON API")
        except Exception as e:
            self.errors.append(f"JSON load error: {str(e)}")


class DatabaseDataProcessor(BaseDataProcessor):
    """Child class - only 10 lines of unique code"""
    
    def __init__(self, connection_string, query):
        super().__init__()  # Initialize parent
        self.connection_string = connection_string
        self.query = query
    
    def load(self):
        """Only the database-specific load logic"""
        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(self.connection_string)
            self.data = pd.read_sql(self.query, engine)
            print(f"✓ Loaded {len(self.data)} rows from database")
        except Exception as e:
            self.errors.append(f"Database load error: {str(e)}")




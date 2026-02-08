import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import json
import sqlite3
from pathlib import Path


# ============================================================================
# PART 1: DATA LOADING LAYER - INHERITANCE IN ACTION
# ============================================================================

class BaseDataLoader(ABC):
    """
    Abstract base class for all data loaders.
    
    Defines the interface all loaders must follow and implements
    common functionality.
    """
    
    def __init__(self):
        self.data = None
        self.load_errors = []
        self.row_count = 0
        self.column_count = 0
    
    @abstractmethod
    def load(self, source):
        """
        Load data from source.
        Each child class implements its own loading logic.
        """
        pass
    
    def validate_schema(self, required_columns):
        """
        Common validation logic - ONCE for all loaders.
        Each loader inherits this method.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        missing_cols = set(required_columns) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return True
    
    def get_data_info(self):
        """Common method to get data info"""
        if self.data is None:
            return None
        
        return {
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'memory_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'column_names': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.to_dict()
        }
    
    def get_missing_values_summary(self):
        """Common method - summarize missing values"""
        if self.data is None:
            return None
        
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        
        return {
            'columns_with_missing': missing[missing > 0].to_dict(),
            'percentage': missing_pct[missing_pct > 0].to_dict(),
            'total_missing': missing.sum()
        }


class CSVDataLoader(BaseDataLoader):
    """Load data from CSV file"""
    
    def load(self, filepath):
        """
        CSV-specific loading logic.
        Validation, error handling, and info collection are inherited.
        """
        try:
            self.data = pd.read_csv(filepath)
            print(f"✓ CSV loaded: {len(self.data)} rows, {len(self.data.columns)} columns")
            return self.data
        except FileNotFoundError:
            self.load_errors.append(f"File not found: {filepath}")
            raise
        except Exception as e:
            self.load_errors.append(f"Error loading CSV: {str(e)}")
            raise


class JSONDataLoader(BaseDataLoader):
    """Load data from JSON file"""
    
    def load(self, filepath):
        """
        JSON-specific loading logic.
        """
        try:
            with open(filepath, 'r') as f:
                json_data = json.load(f)
            
            # Convert list of dicts to DataFrame
            if isinstance(json_data, list):
                self.data = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                self.data = pd.DataFrame([json_data])
            else:
                raise ValueError("JSON must be list of objects or single object")
            
            print(f"✓ JSON loaded: {len(self.data)} rows, {len(self.data.columns)} columns")
            return self.data
        except FileNotFoundError:
            self.load_errors.append(f"File not found: {filepath}")
            raise
        except Exception as e:
            self.load_errors.append(f"Error loading JSON: {str(e)}")
            raise


class DatabaseDataLoader(BaseDataLoader):
    """Load data from SQLite database"""
    
    def load(self, db_config):
        """
        Database-specific loading logic.
        
        Args:
            db_config: dict with 'database', 'table' or 'query' keys
        """
        try:
            if 'database' not in db_config:
                raise ValueError("db_config must include 'database' key")
            
            conn = sqlite3.connect(db_config['database'])
            
            # Load from table or custom query
            if 'table' in db_config:
                self.data = pd.read_sql_table(db_config['table'], conn)
            elif 'query' in db_config:
                self.data = pd.read_sql_query(db_config['query'], conn)
            else:
                raise ValueError("db_config must include 'table' or 'query'")
            
            conn.close()
            print(f"✓ Database loaded: {len(self.data)} rows, {len(self.data.columns)} columns")
            return self.data
        except Exception as e:
            self.load_errors.append(f"Error loading from database: {str(e)}")
            raise


# ============================================================================
# PART 2: DATA VALIDATION LAYER - POLYMORPHISM IN ACTION
# ============================================================================

class BaseValidator(ABC):
    """
    Abstract base class for all validators.
    Defines common interface for validation.
    """
    
    def __init__(self, data):
        self.data = data
        self.validation_errors = []
        self.validation_warnings = []
    
    @abstractmethod
    def validate(self):
        """
        Validate the data.
        Each validator implements its own validation logic.
        """
        pass
    
    def is_valid(self):
        """Common method - check if validation passed"""
        return len(self.validation_errors) == 0
    
    def get_report(self):
        """Common method - generate validation report"""
        return {
            'is_valid': self.is_valid(),
            'errors': self.validation_errors,
            'warnings': self.validation_warnings,
            'total_issues': len(self.validation_errors) + len(self.validation_warnings)
        }


class SchemaValidator(BaseValidator):
    """Validate data schema and structure"""
    
    def __init__(self, data, required_columns, expected_dtypes=None):
        super().__init__(data)
        self.required_columns = required_columns
        self.expected_dtypes = expected_dtypes or {}
    
    def validate(self):
        """Validate schema"""
        # Check required columns
        missing_cols = set(self.required_columns) - set(self.data.columns)
        if missing_cols:
            self.validation_errors.append(f"Missing columns: {missing_cols}")
        
        # Check data types
        if self.expected_dtypes:
            for col, expected_type in self.expected_dtypes.items():
                if col in self.data.columns:
                    actual_type = str(self.data[col].dtype)
                    if actual_type != expected_type:
                        self.validation_warnings.append(
                            f"Column '{col}': expected {expected_type}, got {actual_type}"
                        )
        
        return self.is_valid()


class DataQualityValidator(BaseValidator):
    """Validate data quality"""
    
    def __init__(self, data, max_missing_pct=5.0, allow_duplicates=False):
        super().__init__(data)
        self.max_missing_pct = max_missing_pct
        self.allow_duplicates = allow_duplicates
    
    def validate(self):
        """Validate data quality"""
        # Check missing values
        missing_pct = (self.data.isnull().sum() / len(self.data)) * 100
        missing_cols = missing_pct[missing_pct > self.max_missing_pct]
        
        if not missing_cols.empty:
            for col, pct in missing_cols.items():
                self.validation_errors.append(
                    f"Column '{col}' has {pct:.2f}% missing values (max: {self.max_missing_pct}%)"
                )
        
        # Check duplicates
        duplicates = self.data.duplicated().sum()
        if duplicates > 0 and not self.allow_duplicates:
            self.validation_warnings.append(f"Found {duplicates} duplicate rows")
        
        return self.is_valid()


class BusinessRuleValidator(BaseValidator):
    """Validate business rules"""
    
    def __init__(self, data):
        super().__init__(data)
    
    def validate(self):
        """Validate business rules specific to churn prediction"""
        # For example: tenure should be non-negative
        if 'tenure' in self.data.columns:
            if (self.data['tenure'] < 0).any():
                self.validation_errors.append("Tenure values cannot be negative")
        
        # Monthly charges should be positive
        if 'MonthlyCharges' in self.data.columns:
            if (self.data['MonthlyCharges'] <= 0).any():
                self.validation_warnings.append("Some customers have zero/negative monthly charges")
        
        # Churn column should be binary
        if 'Churn' in self.data.columns:
            unique_values = self.data['Churn'].unique()
            if not all(val in ['Yes', 'No', 0, 1] for val in unique_values):
                self.validation_errors.append("Churn should be binary (Yes/No or 0/1)")
        
        return self.is_valid()


# ============================================================================
# PART 3: MODEL LAYER - INHERITANCE & POLYMORPHISM COMBINED
# ============================================================================

class BaseMLModel(ABC):
    """
    Abstract base class for all ML models.
    Defines common interface and shared functionality.
    """
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.metrics = {}
        self.feature_names = None
    
    @abstractmethod
    def build_model(self):
        """Each model must define how to build itself"""
        pass
    
    def train(self, X_train, y_train):
        """
        Common training logic.
        All models inherit this method.
        """
        if self.model is None:
            self.build_model()
        
        print(f"Training {self.__class__.__name__}...")
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"✓ {self.__class__.__name__} training complete")
    
    def evaluate(self, X_test, y_test):
        """
        Common evaluation logic.
        All models inherit this method.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"✓ {self.__class__.__name__} - Accuracy: {self.metrics['accuracy']:.3f}, F1: {self.metrics['f1']:.3f}")
        return self.metrics
    
    def predict(self, X):
        """
        Common prediction logic.
        All models inherit this method.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def get_metrics_summary(self):
        """Common method to get metrics"""
        return {
            'model': self.__class__.__name__,
            'metrics': self.metrics,
            'is_trained': self.is_trained
        }


class LogisticRegressionModel(BaseMLModel):
    """Logistic Regression - simple and fast"""
    
    def build_model(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)


class RandomForestModel(BaseMLModel):
    """Random Forest - good accuracy"""
    
    def build_model(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )


class XGBoostModel(BaseMLModel):
    """XGBoost - best accuracy but slower"""
    
    def build_model(self):
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        except ImportError:
            print("⚠ XGBoost not installed, using Random Forest instead")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)


# ============================================================================
# PART 4: COMPLETE PIPELINE - PUTTING IT ALL TOGETHER
# ============================================================================

class ChurnPredictionPipeline:
    """
    Complete pipeline that uses polymorphism.
    Works with ANY data loader, validator, or model.
    """
    
    def __init__(self):
        self.loader = None
        self.validators = []
        self.models = []
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def set_loader(self, loader: BaseDataLoader):
        """Set the data loader (polymorphic)"""
        self.loader = loader
    
    def add_validator(self, validator: BaseValidator):
        """Add a validator (polymorphic)"""
        self.validators.append(validator)
    
    def add_model(self, model: BaseMLModel):
        """Add a model (polymorphic)"""
        self.models.append(model)
    
    def load_data(self, source):
        """Load data using the set loader"""
        if self.loader is None:
            raise ValueError("No loader set. Call set_loader() first.")
        
        self.data = self.loader.load(source)
        return self
    
    def validate_data(self):
        """Validate data using all validators (polymorphism!)"""
        print("\nValidating data...")
        
        for validator in self.validators:
            validator.validate()
            report = validator.get_report()
            print(f"  {validator.__class__.__name__}: {'✓ PASS' if report['is_valid'] else '✗ FAIL'}")
            
            if not report['is_valid']:
                for error in report['errors']:
                    print(f"    - {error}")
        
        return self
    
    def prepare_data(self, feature_cols, target_col, test_size=0.2):
        """Prepare data for modeling"""
        print("\nPreparing data...")
        
        X = self.data[feature_cols]
        y = self.data[target_col]
        
        # Handle encoding if necessary
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.factorize(X[col])[0]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            scaler.fit_transform(self.X_train),
            columns=X.columns
        )
        self.X_test = pd.DataFrame(
            scaler.transform(self.X_test),
            columns=X.columns
        )
        
        print(f"✓ Training set: {len(self.X_train)} rows")
        print(f"✓ Test set: {len(self.X_test)} rows")
        
        return self
    
    def train_all_models(self):
        """Train all models using polymorphism"""
        print("\nTraining models...")
        
        for model in self.models:
            model.train(self.X_train, self.y_train)
        
        return self
    
    def evaluate_all_models(self):
        """Evaluate all models using polymorphism"""
        print("\nEvaluating models...")
        
        results = {}
        for model in self.models:
            metrics = model.evaluate(self.X_test, self.y_test)
            results[model.__class__.__name__] = metrics
        
        return results
    
    def get_best_model(self):
        """Get model with best accuracy"""
        best_model = max(
            self.models,
            key=lambda m: m.metrics.get('accuracy', 0)
        )
        return best_model
    
    def predict(self, X, model=None):
        """Make predictions with specified model (or best model)"""
        if model is None:
            model = self.get_best_model()
        
        return model.predict(X)
    
    def get_pipeline_summary(self):
        """Get complete pipeline summary"""
        summary = {
            'loader': self.loader.__class__.__name__,
            'validators': [v.__class__.__name__ for v in self.validators],
            'models': [m.__class__.__name__ for m in self.models],
            'best_model': self.get_best_model().__class__.__name__,
            'best_accuracy': self.get_best_model().metrics.get('accuracy', 0)
        }
        return summary


# ============================================================================
# PART 5: USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Create sample data for testing
    print("="*70)
    print("CHURN PREDICTION PIPELINE")
    print("="*70)
    
    # Create sample dataset
    np.random.seed(42)
    sample_data = {
        'customerID': range(1, 101),
        'tenure': np.random.randint(0, 73, 100),
        'MonthlyCharges': np.random.uniform(20, 120, 100),
        'TotalCharges': np.random.uniform(100, 8000, 100),
        'Churn': np.random.choice(['Yes', 'No'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('/tmp/churn_data.csv', index=False)
    
    # Initialize pipeline
    pipeline = ChurnPredictionPipeline()
    
    # Set loader (polymorphic - could be any loader type)
    pipeline.set_loader(CSVDataLoader())
    
    # Add validators (polymorphic - different validator types)
    pipeline.add_validator(SchemaValidator(
        df,
        required_columns=['customerID', 'tenure', 'MonthlyCharges', 'Churn']
    ))
    pipeline.add_validator(DataQualityValidator(df, max_missing_pct=5.0))
    pipeline.add_validator(BusinessRuleValidator(df))
    
    # Add models (polymorphic - different model types)
    pipeline.add_model(LogisticRegressionModel())
    pipeline.add_model(RandomForestModel())
    
    # Run pipeline
    (pipeline
     .load_data('/tmp/churn_data.csv')
     .validate_data()
     .prepare_data(
         feature_cols=['tenure', 'MonthlyCharges', 'TotalCharges'],
         target_col='Churn'
     )
     .train_all_models()
     )
    
    # Evaluate
    results = pipeline.evaluate_all_models()
    
    # Get summary
    summary = pipeline.get_pipeline_summary()
    print(f"\n{summary}")



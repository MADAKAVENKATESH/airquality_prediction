import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataPreprocessor:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None
        
    def load_data(self, data_path=None):
        """Load data from CSV or other sources"""
        if data_path:
            self.data_path = data_path
            
        if self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path)
        # Add support for other file formats if needed
        
        return self.data
    
    def clean_data(self):
        """Clean and preprocess the data"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Make a copy to avoid SettingWithCopyWarning
        df = self.data.copy()
        
        # Convert all columns to numeric where possible
        for col in df.columns:
            try:
                # First try direct conversion to numeric
                df[col] = pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                try:
                    # If that fails, try to convert to datetime first
                    datetime_series = pd.to_datetime(df[col], errors='coerce')
                    if not datetime_series.isna().all():
                        # If we have valid datetimes, keep them
                        df[col] = datetime_series
                    else:
                        # If not datetimes, try to clean and convert to numeric
                        # Remove any non-numeric characters except decimal point and negative sign
                        cleaned = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                        df[col] = pd.to_numeric(cleaned, errors='coerce')
                except Exception as e:
                    # If all else fails, drop the column
                    df = df.drop(columns=[col])
        
        # Drop any rows with missing values
        df = df.dropna()
        
        # Try to identify and convert datetime column
        date_columns = [col for col in df.columns if any(term in str(col).lower() for term in ['date', 'time', 'timestamp', 'dt']) 
                       and pd.api.types.is_datetime64_any_dtype(df[col])]
        
        # If no datetime columns found, use the index if it looks like a datetime
        if not date_columns:
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    # Drop rows where datetime conversion failed
                    df = df[df.index.notna()]
                except (ValueError, TypeError):
                    # If index can't be converted to datetime, use a simple numeric index
                    df = df.reset_index(drop=True)
                    st.warning("No valid datetime column found. Using numeric index.")
        else:
            # Try each potential datetime column
            for col in date_columns:
                try:
                    # Try to convert to datetime
                    datetime_series = pd.to_datetime(df[col], errors='coerce')
                    if datetime_series.isna().all():
                        continue  # Skip if all NaT after conversion
                        
                    # If we get here, conversion was successful
                    df['datetime'] = datetime_series
                    df = df.set_index('datetime')
                    df = df.drop(columns=[col], errors='ignore')
                    break
                except Exception as e:
                    continue
            else:
                # If no datetime column could be converted, use numeric index
                df = df.reset_index(drop=True)
                st.warning("Could not convert any datetime columns. Using numeric index.")
        
        # Ensure numeric data types for all columns
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception as e:
                st.warning(f"Could not convert column '{col}' to numeric: {str(e)}")
        
        self.data = df
        return self.data
    
    def add_time_features(self):
        """Add time-based features"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Index must be a DatetimeIndex. Call clean_data() first.")
            
        self.data['hour'] = self.data.index.hour
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['month'] = self.data.index.month
        self.data['is_weekend'] = self.data.index.dayofweek >= 5
        
        return self.data
    
    def resample_data(self, freq='H'):
        """Resample time series data to specified frequency"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        return self.data.resample(freq).mean()
    
    def get_feature_importance(self, target_column='AQI'):
        """Get feature importance using correlation with target"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        corr = self.data.corr()
        return corr[target_column].sort_values(ascending=False)
    
    def prepare_for_training(self, target_column='AQI', test_size=0.2, sequence_length=24):
        """Prepare data for LSTM training"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Ensure target column exists
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        # Split into train and test sets
        train_size = int(len(self.data) * (1 - test_size))
        train_data = self.data.iloc[:train_size]
        test_data = self.data.iloc[train_size:]
        
        return train_data, test_data

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import joblib
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app_debug.log')
    ]
)
logger = logging.getLogger(__name__)

from air_quality_model import AirQualityPredictor, get_preventive_measures
from data_preprocessing import DataPreprocessor

# Page config
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌫️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {font-size: 24px; font-weight: bold; color: #1E88E5;}
    .sub-header {font-size: 18px; font-weight: bold; color: #42A5F5;}
    .info-text {font-size: 14px; color: #616161;}
    .warning {color: #FFA000; font-weight: bold;}
    .danger {color: #E53935; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌫️ Air Quality Prediction System")
st.markdown("""
    This application uses LSTM neural networks to predict Air Quality Index (AQI) based on historical pollution and weather data.
    Upload your dataset or use the sample data to get started.
""")

# Sidebar
st.sidebar.header("Settings")
st.sidebar.subheader("Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Upload CSV", "Use Sample Data"]
)

# Initialize session state
if 'model' not in st.session_state:
    # Initialize model without specifying features - will be set during prepare_data
    st.session_state.model = AirQualityPredictor(sequence_length=24)
    st.session_state.data_loaded = False
    st.session_state.model_trained = False

def load_sample_data():
    """Load sample data for demonstration"""
    # Generate sample data
    np.random.seed(42)
    date_range = pd.date_range(start='2023-01-01', periods=1000, freq='H')
    data = {
        'PM2.5': np.random.normal(30, 10, 1000).clip(0, 150),
        'PM10': np.random.normal(50, 15, 1000).clip(0, 200),
        'NO2': np.random.normal(25, 8, 1000).clip(0, 100),
        'CO': np.random.normal(1.5, 0.5, 1000).clip(0, 5),
        'temperature': np.random.normal(20, 5, 1000).clip(0, 40),
        'humidity': np.random.normal(60, 20, 1000).clip(20, 100),
        'wind_speed': np.random.normal(10, 3, 1000).clip(0, 30),
    }
    df = pd.DataFrame(data, index=date_range)
    
    # Calculate a simple AQI (this is a simplified example)
    df['AQI'] = (0.3 * df['PM2.5'] + 0.2 * df['PM10'] + 0.2 * df['NO2'] + 
                 0.1 * df['CO'] + 0.1 * (100 - df['humidity']) + 0.1 * df['wind_speed'])
    
    return df

def log_execution():
    """Log function execution for debugging"""
    import inspect
    caller = inspect.currentframe().f_back.f_code.co_name
    logger.info(f"Executing: {caller}")

def main():
    logger.info("\n" + "="*80)
    logger.info(f"Starting app execution at {datetime.now()}")
    
    # Data Loading Section
    st.header("1. Data Loading")
    log_execution()
    
    if data_source == "Upload CSV":
        logger.info("CSV upload selected")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="csv_uploader")
        logger.info(f"File uploaded: {uploaded_file is not None}")
        if uploaded_file is not None:
            try:
                # Try different encodings and delimiters
                # Read the file with progress
                with st.spinner('Reading CSV file...'):
                    try:
                        # Read the file content once and work with it
                        file_content = uploaded_file.read().decode('utf-8', errors='ignore')
                        
                        # Remove comment lines starting with #
                        lines = [line for line in file_content.split('\n') if not line.strip().startswith('#')]
                        file_content = '\n'.join(lines)
                        
                        # Try to detect the delimiter
                        possible_delimiters = [',', ';', '\t', '|']
                        delimiter_counts = {delim: line.count(delim) for delim in possible_delimiters 
                                         for line in lines[:10] if line.strip()}
                        
                        # Get the most common delimiter
                        delimiter = max(delimiter_counts, key=delimiter_counts.get) if delimiter_counts else ','
                        
                        # Create a file-like object from the content
                        from io import StringIO
                        file_like = StringIO(file_content)
                        
                        # Try to read with the detected delimiter
                        df = pd.read_csv(
                            file_like, 
                            sep=delimiter,
                            nrows=1000,  # Only read first 1000 rows initially
                            encoding='latin1',  # More permissive encoding
                            on_bad_lines='warn',  # Skip bad lines instead of failing
                            engine='python',  # More flexible parser
                            comment='#',  # Skip comment lines
                            skip_blank_lines=True  # Skip empty lines
                        )
                        
                        # Show file info
                        st.info(f"File loaded with {len(df)} rows and {len(df.columns)} columns.")
                        
                        # Let user select date column if multiple options exist
                        date_columns = [col for col in df.columns if any(term in str(col).lower() 
                                                                      for term in ['date', 'time', 'timestamp', 'dt'])]
                        
                        if date_columns:
                            date_col = st.selectbox(
                                "Select datetime column (or 'None' if not applicable)",
                                options=['None'] + date_columns
                            )
                            
                            if date_col != 'None':
                                # Read the full file with the selected date column as index
                                file_like = StringIO(file_content)  # Recreate file-like object
                                df = pd.read_csv(
                                    file_like,
                                    sep=delimiter,
                                    parse_dates=[date_col],
                                    index_col=date_col,
                                    encoding='latin1',
                                    on_bad_lines='warn',
                                    engine='python'
                                )
                            else:
                                # Read without datetime index
                                file_like = StringIO(file_content)  # Recreate file-like object
                                df = pd.read_csv(
                                    file_like,
                                    sep=delimiter,
                                    encoding='latin1',
                                    on_bad_lines='warn',
                                    engine='python'
                                )
                        else:
                            # No obvious date columns found, read without datetime parsing
                            file_like = StringIO(file_content)  # Recreate file-like object
                            df = pd.read_csv(
                                file_like,
                                sep=delimiter,
                                encoding='latin1',
                                on_bad_lines='warn',
                                engine='python'
                            )
                        
                        # Process the data
                        preprocessor = DataPreprocessor()
                        preprocessor.data = df
                        
                        with st.spinner('Processing data...'):
                            # Clean the data
                            df_cleaned = preprocessor.clean_data()
                            
                            # Add time features if we have a datetime index
                            if isinstance(df_cleaned.index, pd.DatetimeIndex):
                                df_with_features = preprocessor.add_time_features()
                            else:
                                df_with_features = df_cleaned
                                st.warning("No valid datetime index found. Using data as-is without time features.")
                            
                            # Store the processed data
                            st.session_state.data = df_with_features
                            st.session_state.data_loaded = True
                            
                            # Display results
                            st.success(f"Data processed successfully! Shape: {df_with_features.shape}")
                            
                            # Show data summary
                            st.subheader("Data Summary")
                            st.write(f"Total rows: {len(df_with_features)}")
                            st.write(f"Total columns: {len(df_with_features.columns)}")
                            
                            # Show data types
                            st.subheader("Column Types")
                            st.write(df_with_features.dtypes)
                            
                            # Show sample data
                            st.subheader("Sample Data")
                            st.dataframe(df_with_features.head())
                            
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
                        st.error(f"Error type: {type(e).__name__}")
                        import traceback
                        st.text(traceback.format_exc())
                        return
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    else:
        logger.info("Sample data option selected")
        if st.button("Load Sample Data", key="load_sample_btn"):
            logger.info("Loading sample data...")
            with st.spinner('Loading sample data...'):
                try:
                    df = load_sample_data()
                    st.session_state.data = df
                    st.session_state.data_loaded = True
                    logger.info(f"Sample data loaded. Shape: {df.shape}")
                    st.success("Sample data loaded successfully!")
                except Exception as e:
                    logger.error(f"Error loading sample data: {str(e)}")
                    st.error(f"Error loading sample data: {str(e)}")
    
    # Display Data
    if st.session_state.get('data_loaded', False):
        st.subheader("Preview of the Data")
        st.dataframe(st.session_state.data.head())
        
        # Data Visualization
        st.subheader("Data Visualization")
        
        # Only show numeric columns for plotting
        numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns found for plotting.")
        else:
            plot_cols = st.multiselect(
                "Select columns to plot (only numeric columns shown):",
                options=numeric_cols,
                default=numeric_cols[0:min(3, len(numeric_cols))]
            )
            
            if plot_cols:
                try:
                    # Ensure we're only using numeric data
                    plot_data = st.session_state.data[plot_cols].apply(pd.to_numeric, errors='coerce')
                    
                    # Drop any rows with all NA values that might have been created
                    plot_data = plot_data.dropna(how='all')
                    
                    if not plot_data.empty:
                        # Reset index to include it in the plot if it's a datetime
                        if isinstance(plot_data.index, pd.DatetimeIndex):
                            plot_data = plot_data.reset_index()
                            x_col = plot_data.columns[0]  # First column is the datetime index
                            
                            # Create separate traces for each column to handle different scales
                            fig = px.line(plot_data, x=x_col, y=plot_cols, 
                                        title="Time Series Plot",
                                        labels={'value': 'Value', 'variable': 'Metric'})
                            
                            # Update layout for better readability
                            fig.update_layout(
                                xaxis_title='Date/Time',
                                yaxis_title='Value',
                                legend_title='Metrics',
                                hovermode='x unified'
                            )
                        else:
                            # If no datetime index, just plot the values
                            fig = px.line(plot_data, title="Time Series Plot")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No valid numeric data to plot after cleaning.")
                        
                except Exception as e:
                    st.error(f"Error creating plot: {str(e)}")
                    logger.error(f"Plotting error: {str(e)}")
                    logger.error(traceback.format_exc())
        
        # Model Training Section
        st.header("2. Model Training")
        
        # Check if AQI column exists or needs to be created
        data_columns = st.session_state.data.columns.tolist()
        aqi_column = None
        
        # Try to find AQI column (case insensitive)
        possible_aqi_columns = [col for col in data_columns if 'aqi' in col.lower()]
        
        if possible_aqi_columns:
            aqi_column = possible_aqi_columns[0]
            st.info(f"Using column '{aqi_column}' as the AQI target variable.")
        else:
            st.warning("No AQI column found. You can either:")
            st.markdown("1. Upload a dataset with an AQI column (or similarly named column)")
            st.markdown("2. Select which column to use as the target variable")
            
            # Let user select which column to use as target
            target_col = st.selectbox(
                "Select the target variable (what you want to predict):",
                options=data_columns,
                index=0 if data_columns else None
            )
            
            if st.button(f"Use '{target_col}' as target variable"):
                # Rename the selected column to 'AQI' for the model
                st.session_state.data = st.session_state.data.rename(columns={target_col: 'AQI'})
                aqi_column = 'AQI'
                st.experimental_rerun()
        
        if aqi_column or 'AQI' in st.session_state.data.columns:
            if st.button("Train Model"):
                with st.spinner('Training model... This may take a few minutes.'):
                    try:
                        # Initialize preprocessor with the loaded data
                        preprocessor = DataPreprocessor()
                        preprocessor.data = st.session_state.data
                        
                        # Prepare data
                        train_data, test_data = preprocessor.prepare_for_training(target_column=aqi_column or 'AQI')
                        
                        # Prepare data for LSTM
                        X_train, y_train = st.session_state.model.prepare_data(
                            train_data, 
                            target_column=aqi_column or 'AQI',
                            fit_scalers=True  # Fit scalers on training data
                        )
                        
                        # Prepare validation data using the same scalers (fit_scalers=False)
                        X_val, y_val = st.session_state.model.prepare_data(
                            test_data,
                            target_column=aqi_column or 'AQI',
                            fit_scalers=False
                        )
                        
                        # Train model with more epochs and patience
                        history = st.session_state.model.train(
                            X_train, 
                            y_train, 
                            X_val, 
                            y_val, 
                            epochs=100,  # Increased epochs
                            batch_size=64  # Increased batch size
                        )
                        
                        # Save the trained model
                        st.session_state.model.save_model()
                        
                        st.session_state.model_trained = True
                        st.success("Model trained successfully!")
                        
                        # Plot training history
                        hist_df = pd.DataFrame(history.history)
                        fig = px.line(hist_df, y=['loss', 'val_loss'], 
                                    title='Training History (Loss)')
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())
        else:
            st.info("Please select a target variable to proceed with model training.")
        
        # Prediction Section
        if st.session_state.get('model_trained', False):
            st.header("3. Make Predictions")
            
            # Get last available data point
            last_data = st.session_state.data.iloc[-24:].copy()  # Last 24 hours
            
            # Make prediction
            try:
                # Ensure we don't include the target column in features
                if aqi_column in last_data.columns:
                    last_data = last_data.drop(columns=[aqi_column])
                elif 'AQI' in last_data.columns:
                    last_data = last_data.drop(columns=['AQI'])
                
                # Make prediction using the model
                # The model's predict method will handle scaling and reshaping
                prediction = st.session_state.model.predict(last_data)[0]
                prediction = round(prediction, 1)  # Round to 1 decimal place
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                prediction = 0  # Default value in case of error
            
            # Display prediction
            st.subheader("Predicted AQI")
            st.metric("Next hour AQI prediction", f"{prediction:.1f}")
            
            # Get preventive measures
            measures = get_preventive_measures(prediction)
            st.subheader("Recommended Actions")
            st.info(measures)
            
            # Show AQI scale
            st.subheader("AQI Scale")
            
            # Define AQI scale data
            aqi_scale = [
                {"Range": "0-50", "Level": "Good", "Color": "#4CAF50"},
                {"Range": "51-100", "Level": "Moderate", "Color": "#FFEB3B"},
                {"Range": "101-150", "Level": "Unhealthy for Sensitive Groups", "Color": "#FF9800"},
                {"Range": "151-200", "Level": "Unhealthy", "Color": "#F44336"},
                {"Range": "201-300", "Level": "Very Unhealthy", "Color": "#9C27B0"},
                {"Range": "301-500", "Level": "Hazardous", "Color": "#880E4F"}
            ]
            
            # Create a 2x3 grid of colored boxes
            cols = st.columns(3)
            for i, level in enumerate(aqi_scale):
                with cols[i % 3]:
                    st.markdown(
                        f"""
                        <div style='
                            background-color: {level['Color']};
                            color: {'black' if level['Level'] in ['Moderate', 'Unhealthy for Sensitive Groups'] else 'white'};
                            padding: 10px;
                            border-radius: 5px;
                            margin: 5px 0;
                            text-align: center;
                        '>
                            <div style='font-weight: bold;'>{level['Range']}</div>
                            <div>{level['Level']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()

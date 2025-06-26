# Air Quality Prediction System

An AI-powered system that predicts Air Quality Index (AQI) using LSTM neural networks and provides preventive measures based on the predictions.

## Features

- **Data Preprocessing**: Handles missing values, feature engineering, and time series preparation
- **LSTM Model**: Deep learning model for accurate AQI prediction
- **Web Interface**: User-friendly Streamlit app for interaction
- **Real-time Predictions**: Get AQI predictions and health recommendations
- **Data Visualization**: Interactive plots for data exploration

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd air-quality-prediction
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. In the web interface:
   - Upload your dataset or use the sample data
   - Explore the data using interactive visualizations
   - Train the LSTM model
   - Get AQI predictions and health recommendations

## Data Format

The application expects a CSV file with the following columns (column names are flexible):
- Date/time information (will be automatically detected)
- Air quality parameters (PM2.5, PM10, NO2, CO, etc.)
- Weather data (temperature, humidity, wind speed, etc.)
- (Optional) AQI values if available for supervised learning

## Model Architecture

The LSTM model consists of:
- Two LSTM layers (100 and 50 units)
- Dropout layers for regularization
- Dense layers for prediction
- Mean Squared Error (MSE) loss function
- Adam optimizer

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Python, TensorFlow, and Streamlit
- Uses scikit-learn for data preprocessing
- Plotly for interactive visualizations

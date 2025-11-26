import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR
from datetime import datetime, timedelta

# ===============================
# LOAD & PREPARE DATA
# ===============================
def load_data(filename='bitcoin.csv', prediction_days=30):
    """
    Loads and prepares the data for time-series forecasting.
    
    Args:
        filename (str): The name of the CSV file.
        prediction_days (int): The number of days to predict into the future.
        
    Returns:
        tuple: Contains features (X), labels (y), the array for future prediction,
               and the original dataframe with dates.
    """
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df.set_index('Date', inplace=True)
    
    # Create the label column by shifting the 'Price' column
    df['Prediction'] = df[['Price']].shift(-prediction_days)

    # Create the feature dataset (X) and drop the last 'prediction_days' rows
    X = np.array(df[['Price']])[:-prediction_days]

    # Create the label dataset (y) and drop the last 'prediction_days' rows
    y = np.array(df['Prediction'])[:-prediction_days]

    # Get the data to be used for the actual prediction (last 30 days of available data)
    prediction_array = np.array(df[['Price']])[-prediction_days:]
    
    return X, y, prediction_array, df

# ===============================
# TRAIN MODEL WITH HYPERPARAMETER TUNING
# ===============================
def train_model(X, y):
    """
    Trains an SVR model using a chronological split and GridSearchCV for optimization.
    
    Args:
        X (np.array): Feature data.
        y (np.array): Label data.
        
    Returns:
        tuple: The best trained model, its R-squared accuracy score on the test set,
               and the best parameters found.
    """
    # --- Correct, Chronological Splitting for Time-Series Data ---
    # We will use the first 80% of the data for training and the last 20% for testing.
    test_size = 0.2
    split_index = int(len(X) * (1 - test_size))
    xtrain, xtest = X[:split_index], X[split_index:]
    ytrain, ytest = y[:split_index], y[split_index:]

    print("--- Model Training ---")
    print(f"Training data size: {len(xtrain)} samples")
    print(f"Testing data size: {len(xtest)} samples")

    # --- Hyperparameter Tuning using GridSearchCV ---
    # Define the parameter grid to search through
    param_grid = {
        'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
    }

    # Use TimeSeriesSplit for cross-validation that respects the time order
    tscv = TimeSeriesSplit(n_splits=5)

    # Create and fit the GridSearchCV object
    grid_search = GridSearchCV(
        SVR(kernel='rbf'), 
        param_grid, 
        cv=tscv, 
        verbose=1, 
        n_jobs=-1 # Use all available CPU cores
    )
    grid_search.fit(xtrain, ytrain)

    # The best model found by the search
    best_model = grid_search.best_estimator_
    
    # Evaluate the best model on the held-out test set
    accuracy = best_model.score(xtest, ytest)
    
    print("\n--- Model Evaluation ---")
    print(f"Best Parameters Found: {grid_search.best_params_}")
    print(f"R-squared Score on Test Set: {accuracy:.4f}")
    
    return best_model, accuracy

# ===============================
# PREDICT & VISUALIZE
# ===============================
def predict_and_visualize(model, prediction_array, original_df, prediction_days=30):
    """
    Makes future predictions and plots the results.
    
    Args:
        model (SVR): The trained SVR model.
        prediction_array (np.array): The last 'prediction_days' of data for forecasting.
        original_df (pd.DataFrame): The original dataframe for plotting historical data.
        prediction_days (int): The number of days being forecast.
    """
    # Get predictions from the model
    future_predictions = model.predict(prediction_array)
    
    print("\n--- 30-Day Bitcoin Price Forecast ---")
    print(future_predictions)

    # --- Plotting the results ---
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(16, 8))

    # Get the dates for the forecast
    last_date = original_df.index[-1]
    prediction_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, prediction_days + 1)])

    # Create a dataframe for the forecast
    forecast_df = pd.DataFrame(index=prediction_dates, data={'Forecast': future_predictions})

    plt.plot(original_df.index, original_df['Price'], label='Historical Price', color='royalblue')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='30-Day Forecast', color='orangered', linestyle='--')
    
    plt.title('Bitcoin Price: History and 30-Day Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == '__main__':
    # Define constants
    PREDICTION_DAYS = 30
    FILENAME = 'bitcoin.csv'
    
    # 1. Load and prepare the data
    X, y, future_prediction_array, df = load_data(filename=FILENAME, prediction_days=PREDICTION_DAYS)
    
    # 2. Train the model and find the best parameters
    # Note: If X is empty, it means the dataset is too small for the prediction window
    if len(X) > 0:
        best_svr_model, model_accuracy = train_model(X, y)
    
        # 3. Predict future prices and visualize the results
        predict_and_visualize(
            model=best_svr_model, 
            prediction_array=future_prediction_array, 
            original_df=df, 
            prediction_days=PREDICTION_DAYS
        )
    else:
        print(f"Error: The dataset is too small to create a training set with a {PREDICTION_DAYS}-day prediction window.")

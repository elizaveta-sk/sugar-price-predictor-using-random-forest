"""
Dataset:
    Yahoo Finance Sugar #11 futures, ticker SB=F
    https://finance.yahoo.com/quote/SB%3DF/

Algorithm used:
    Random Forest Regressor
"""
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def download_data() -> pd.DataFrame:
    """Import daily Sugar #11 futures prices from Yahoo Finance (from 2015-01-01 until today)."""

    data = yf.download("SB=F", start="2015-01-01", auto_adjust=True) #auto_adjust adjusts prices for dividends
    data = data[["Close"]] #keep only 'close' information in the dataframe
    data.to_csv("data/simple_sugar_prices.csv")

    print(f"Successfully downloaded data to 'data/simple_sugar_prices.csv'")
    return data

def clean_and_prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Clean the data and create features."""
    data = data.copy() #create a copy of data

    # Sort data by date
    data = data.sort_index()

    # Remove missing prices.
    data = data.dropna()

    # Create columns from the closing price.
    data["price_yesterday"] = data["Close"].shift(1) #shift one row back to capture yesterday's price
    data["price_5_days_ago"] = data["Close"].shift(5) #shift five row back to capture price 5 days ago
    data["average_price_5_days"] = data["Close"].rolling(5).mean() #take an average of the last 5 days
    data["average_price_20_days"] = data["Close"].rolling(20).mean() #take an average of the last 20 days

    feature_columns = [
        "price_yesterday",
        "price_5_days_ago",
        "average_price_5_days",
        "average_price_20_days",
    ]
    latest_features = data[feature_columns].dropna().tail(1) #today's data (features only) for model input

    # Model aim: predict the price in 5 days
    data["future_price"] = data["Close"].shift(-5)

    # Remove extra rows after processing
    data = data.dropna()

    features = data[feature_columns] #model input
    target = data["future_price"] #model output

    return features, target, latest_features


def train_model(features: pd.DataFrame, target: pd.Series) -> RandomForestRegressor:
    """Train one machine-learning algorithm: Random Forest."""
    split_point = int(len(features) * 0.8) #finding a point to split data at for training and testing 80/20%

    x_train = features.iloc[:split_point]
    x_test = features.iloc[split_point:]
    y_train = target.iloc[:split_point]
    y_test = target.iloc[split_point:]

    model = RandomForestRegressor(n_estimators=100, random_state=42) #100 decision trees together
    model.fit(x_train, y_train) #learn patterns

    predictions = model.predict(x_test) #testing with test data

    #Evaluation of the results
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions) #comparing to average guesses

    print("\nModel results")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return model

def predict_latest_price(model: RandomForestRegressor, latest_features: pd.DataFrame) -> None:
    """Predict the sugar price 5 trading days after the latest row."""
    prediction = model.predict(latest_features)[0]
    latest_date = latest_features.index[0].date()

    print("\nPrediction")
    print(f"Latest available date: {latest_date}")
    print(f"Predicted sugar price in 5 trading days: {prediction:.4f}")


def main() -> None:
    file_path = "data/simple_sugar_prices.csv"

    if os.path.exists(file_path):
        print("Using data file.")
        data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    else:
        data = download_data()

    features, target, latest_features = clean_and_prepare_data(data)
    model = train_model(features, target)
    predict_latest_price(model, latest_features)


if __name__ == "__main__":
    main()

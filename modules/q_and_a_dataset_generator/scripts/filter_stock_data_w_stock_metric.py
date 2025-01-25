import json
import re
import traceback as tb
from datetime import datetime

import yfinance as yf
from src.paths import DATA_DIR


def get_sp500_baseline(date_recommended: str, last_date: str, verbose: bool = False) -> float:
    """
    Fetches the S&P 500 (^GSPC) market return for the given period and calculates its annualized return.

    Parameters:
        date_recommended (str): The start date for the market comparison (YYYY-MM-DD).
        last_date (str): The end date for the market comparison (YYYY-MM-DD).
        verbose (bool): If True, logs details.

    Returns:
        float: Annualized return of S&P 500 over the given period.
    """
    try:
        sp500_data = yf.Ticker("^GSPC")
        historical_data = sp500_data.history(start=date_recommended, end=last_date)

        if historical_data.empty:
            raise ValueError("No S&P 500 data available for the given dates.")

        # Get closing prices for the start and end dates
        start_price = historical_data['Close'].iloc[0]
        end_price = historical_data['Close'].iloc[-1]

        # Calculate the number of days in the period
        days_difference = (datetime.strptime(last_date, '%Y-%m-%d') - datetime.strptime(date_recommended, '%Y-%m-%d')).days

        # Calculate annualized return
        sp500_annualized_return = ((end_price / start_price) ** (365 / days_difference) - 1)

        if verbose:
            print(f"S&P 500 Start Date: {date_recommended}, Start Price: {start_price}")
            print(f"S&P 500 End Date: {last_date}, End Price: {end_price}")
            print(f"S&P 500 Annualized Return: {sp500_annualized_return * 100:.2f}%")

        return sp500_annualized_return

    except Exception:
        print("Error fetching S&P 500 baseline return.")
        tb.print_exc()
        return 0.1  # Default to 10% if data fetch fails


def check_stock_performance(stock_name: str, date_recommended: str, last_date: str, verbose: bool = False) -> bool:
    """
    Check if a stock's value has increased more than the S&P 500 market return (annualized).

    Parameters:
        stock_name (str): The stock ticker symbol (e.g., 'AAPL', 'GOOG').
        date_recommended (str): The start date (YYYY-MM-DD) when the stock was recommended.
        last_date (str): The end date (YYYY-MM-DD) to compare the stock price.
        verbose (bool): If True, prints detailed logs.

    Returns:
        bool: True if the stock outperformed the S&P 500 over the period, False otherwise.
    """
    try:
        # Convert date strings to datetime objects for validation
        start_date = datetime.strptime(date_recommended, '%Y-%m-%d')
        end_date = datetime.strptime(last_date, '%Y-%m-%d')

        if start_date >= end_date:
            raise ValueError("The start date must be earlier than the end date.")

        # Get S&P 500 performance as the baseline interest
        baseline_interest = get_sp500_baseline(date_recommended, last_date, verbose)

        # Fetch stock data
        stock_data = yf.Ticker(stock_name)
        historical_data = stock_data.history(start=date_recommended, end=last_date)

        if historical_data.empty:
            raise ValueError("No stock data available for the given dates.")

        # Get closing prices on the start and end dates
        start_price = historical_data['Close'].iloc[0]
        end_price = historical_data['Close'].iloc[-1]

        # Calculate percentage increase
        percentage_increase = ((end_price - start_price) / start_price) 

        # Calculate annualized percentage increase
        days_difference = (end_date - start_date).days
        annualized_increase = ((end_price / start_price) ** (365 / days_difference) - 1)

        if verbose:
            print(f"\nStock: {stock_name}")
            print(f"Start Date: {date_recommended}, Start Price: {start_price}")
            print(f"End Date: {last_date}, End Price: {end_price}")
            print(f"Days Difference: {days_difference}")
            print(f"Percentage Increase (Non-Annualized): {percentage_increase * 100:.2f}%")
            print(f"Annualized Percentage Increase: {annualized_increase * 100:.2f}%")
            print(f"S&P 500 Baseline Annualized Return: {baseline_interest * 100:.2f}%")

        # Check if the stock outperformed the S&P 500
        outperforms_baseline = annualized_increase > baseline_interest

        print(f"Did {stock_name} outperform the S&P 500? {outperforms_baseline}\n")
        return outperforms_baseline

    except Exception:
        print("Error while running check_stock_performance()")
        tb.print_exc()
        return False
    
    
def extract_stock_recommendation(response: str) -> str:
    """
    Extracts the stock recommendation from the response using regex.
    """
    match = re.search(r"\[Stock Recommendation\]:\s*(\S+)", response)
    return match.group(1) if match else None

def is_valid_stock(stock_ticker: str) -> bool:
    """
    Checks if the stock ticker exists on Yahoo Finance.
    """
    try:
        stock_data = yf.Ticker(stock_ticker)
        hist = stock_data.history(period="1d")
        return not hist.empty  # If historical data exists, the stock ticker is valid
    except:
        return False

def print_questions_and_recommendations():
    """
    Reads the dataset, filters valid stock recommendations, and stores the cleaned dataset.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    count_invalid = 0
    filtered_data = []

    for example in data:
        question = example["about_me"].split("\n")[-1]  # Extracts the question
        stock_recommendation = extract_stock_recommendation(example.get("response", ""))
        
        if not stock_recommendation or len(stock_recommendation) > 5 or not is_valid_stock(stock_recommendation):
            print(f"Invalid stock recommendation: {stock_recommendation}")
            count_invalid += 1
            continue  # Skip invalid stock recommendations
        
        # Assume dates for now
        date_recommended = "2023-08-21"
        last_date = "2024-08-21"

        # Check if the stock outperformed the S&P 500
        if not check_stock_performance(stock_recommendation, date_recommended, last_date, verbose=False):
            print(f"Stock {stock_recommendation} underperformed S&P 500. Skipping.")
            continue  # Skip underperforming stocks

        # Store valid recommendations
        filtered_data.append(example)
        print(f"Question: {question}")
        print(f"Stock Name Recommendation: {stock_recommendation}")
        print("-" * 50)

    # Save filtered dataset
    with open(output_file_path, "w") as f:
        json.dump(filtered_data, f, indent=4)

    # print how much stocks outperformed out of the valid recommendations
    print(f"Total valid stock recommendations: {len(filtered_data)} that outperformed the S&P 500.")
    print(f"Total valid stock recommendations: {len(data) - count_invalid}, out of {len(data)} examples.")
    print(f"Total invalid stock recommendations: {count_invalid}, out of {len(data)} examples.")
    print(f"Filtered dataset saved to {output_file_path}")


if __name__ == "__main__":
    
    # Load the dataset
    file_path = DATA_DIR / "training_data_w_stocks.json"
    output_file_path = DATA_DIR / "filtered_training_data_based_on_stock_metric.json"

    print_questions_and_recommendations()

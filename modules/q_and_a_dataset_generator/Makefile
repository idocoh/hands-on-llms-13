.PHONY: init download embed training-data

# install Poetry and Python dependencies
init:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install

# download 150k news articles from Alpaca and save them in JSON file
download:
	@echo "Downloading data from Alpaca..."
	poetry run python scripts/download_news_from_alpaca.py
	
# run the feature-pipeline and send the feature to the feature store
embed:
	@echo "Embedding text chunks into Qdrant..."
	poetry run python scripts/embed_news_into_qdrant.py

# backfills the feature group using historical data
training-data:
	poetry run python scripts/generate_training_data.py

data-w-stocks:
	poetry run python scripts/generate_stock_training_data.py

data-w-stocks-optimized:
	poetry run python scripts/filter_stock_data_w_stock_metric.py \
	 --date-recommended 2023-08-21 --last-date 2024-08-21
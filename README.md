# Disease Prediction Using Machine Learning

A predictive system for disease classification based on symptoms using ML algorithms and ensemble learning.

## Project Structure

- **data/**: Dataset CSV file
- **models/**: Saved trained model and encoders
- **src/**: Source code (`train.py`, `predict.py`, `utils.py`)
- **tests/**: Unit and integration tests
- **requirements.txt**: Python dependencies
- **Dockerfile**: Containerization instructions
- **.github/workflows/ci.yml**: CI pipeline
- **README.md**: Documentation

## Running Locally

1. Install dependencies:  
   `pip install -r requirements.txt`
2. Train models:  
   `python src/train.py`
3. Run prediction:  
   `python src/predict.py`

## Docker

    docker build -t disease-prediction-ml .
    docker run --rm disease-prediction-ml

## CI/CD

See `.github/workflows/ci.yml` for CI steps and integration.

## License

MIT

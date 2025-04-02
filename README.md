# Radar Data Processing and Model Training

This project involves processing mmWave radar data and training a deep learning model to classify fall and non-fall events using radar signals.

## Project Structure

### Key Files

- **`training.ipynb`**: Contains the code for data preprocessing, model training, evaluation, and saving the trained model.
- **`try.ipynb`**: Used for testing and visualizing the results.
- **`best_model.h5` / `best_model.keras`**: Saved models from training.
- **`content/data/`**: Contains example radar data files and logs for fall and non-fall events.

## How to Run

1. **Install Dependencies**:
   Install the required Python libraries:
   ```bash
   pip install tensorflow scikit-learn matplotlib scipy
   ```

## Model Details

The model uses a custom architecture with:

- **Conv2Plus1D layers** for spatiotemporal feature extraction.
- **ResidualBlock layers** for efficient learning.
- **Adam optimizer** with a learning rate of `1e-4`.
- **Sparse categorical crossentropy loss** for classification.

## Data Processing

Radar data is processed into range-Doppler heatmaps using the `process_radar_data` function. The data is split into training, validation, and test sets using `train_test_split`.

## Results

- **Accuracy**: ~78%
- **Precision**: 70%
- **Recall**: 100%
- **F1 Score**: 82.5%

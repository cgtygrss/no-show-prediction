# No-Show Prediction Pipeline

This project implements a machine learning pipeline to predict whether a patient will show up for their medical appointment. It uses various classification algorithms to analyze the dataset and identify patterns associated with "no-show" appointments.

## Project Overview

The pipeline is designed to be modular and robust, handling data loading, cleaning, feature engineering, visualization, preprocessing, model training, and evaluation.

### Key Features

*   **Data Cleaning**: Handles missing values, renames columns for consistency, and filters out invalid data (e.g., negative ages).
*   **Feature Engineering**: Extracts meaningful features from date columns (e.g., day of the week, month) and calculates the waiting time between scheduling and the appointment.
*   **Visualization**: Generates exploratory data analysis (EDA) plots to visualize relationships between features and the target variable (saved as PNG files).
*   **Preprocessing**: Performs binary encoding for gender and target variables, one-hot encoding for categorical features, and standard scaling for numerical features.
*   **Model Training**: Trains multiple machine learning models to compare performance:
    *   Decision Tree
    *   Logistic Regression
    *   Random Forest
    *   K-Nearest Neighbors (KNN)
    *   Naive Bayes
*   **Evaluation**: detailed classification reports and confusion matrices for each model.

## Prerequisites

*   Python 3.x
*   pip

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd no-show-prediction
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project includes a command-line interface (CLI) for easy execution.

### Basic Run
To run the pipeline with the default dataset (`KaggleV2-May-2016.csv`):

```bash
python main.py
```

### Specify Dataset File
To specify a different dataset file:

```bash
python main.py --file path/to/your/dataset.csv
```

### Enable Visualization
To generate and save visualization plots during the run:

```bash
python main.py --visualize
```

### Combined Example
```bash
python main.py --file data/appointments.csv --visualize
```

## Project Structure

*   `main.py`: The main script containing the `NoShowPredictionPipeline` class and execution logic.
*   `requirements.txt`: List of Python dependencies.
*   `README.md`: Project documentation.

## Dataset

The project expects a CSV dataset with columns similar to the Kaggle "Medical Appointment No Shows" dataset, including:
*   `PatientId`
*   `AppointmentID`
*   `Gender`
*   `ScheduledDay`
*   `AppointmentDay`
*   `Age`
*   `Neighbourhood`
*   `Scholarship`
*   `Hipertension`
*   `Diabetes`
*   `Alcoholism`
*   `Handcap`
*   `SMS_received`
*   `No-show`

## License

[Choose a license, e.g., MIT]

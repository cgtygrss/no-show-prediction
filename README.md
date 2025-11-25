# No-Show Prediction Pipeline

This project implements a machine learning pipeline to predict whether a patient will show up for their medical appointment. It uses various classification algorithms to analyze the dataset and identify patterns associated with "no-show" appointments.

## Project Overview

The pipeline is designed to be modular and robust, handling data loading, cleaning, feature engineering, visualization, preprocessing, model training, and evaluation.

### Key Features

*   **Automated Data Loading**: Automatically downloads the dataset from Kaggle if not found locally.
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
*   **Evaluation**: Detailed classification reports and confusion matrices for each model.

## Prerequisites

*   Python 3.x
*   pip
*   Kaggle API credentials

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

3.  **Set up Kaggle API Credentials**:
    To automatically download the dataset, you need a `kaggle.json` file.
    *   Go to your [Kaggle Account Settings](https://www.kaggle.com/settings).
    *   Scroll to the "API" section and click "Create New Token".
    *   Move the downloaded `kaggle.json` file to `~/.kaggle/`:
        ```bash
        mkdir -p ~/.kaggle
        mv ~/Downloads/kaggle.json ~/.kaggle/
        chmod 600 ~/.kaggle/kaggle.json
        ```

## Usage

The project includes a command-line interface (CLI) for easy execution.

### Basic Run
To run the pipeline. If the dataset is missing, it will attempt to download it from Kaggle:

```bash
python main.py
```

### Enable Visualization
To generate and save visualization plots during the run:

```bash
python main.py --visualize
```

### Specify Dataset File or URL
To specify a different local file or a different Kaggle dataset URL:

```bash
python main.py --file my_data.csv --url username/dataset-slug
```

## Project Structure

The project follows a modular architecture:

*   `main.py`: The entry point script that orchestrates the pipeline.
*   `src/`: Source code directory.
    *   `data_loader.py`: Handles loading and downloading data from Kaggle.
    *   `data_cleaner.py`: Performs data cleaning and column renaming.
    *   `feature_engineer.py`: Creates new features from existing data.
    *   `visualizer.py`: Generates and saves EDA plots.
    *   `preprocessor.py`: Handles encoding, scaling, and train-test splitting.
    *   `model_trainer.py`: Trains and evaluates machine learning models.
*   `requirements.txt`: List of Python dependencies.
*   `README.md`: Project documentation.

## Dataset

The project uses the **Medical Appointment No Shows** dataset from Kaggle.

**Default URL:** [https://www.kaggle.com/datasets/joniarroba/noshowappointments](https://www.kaggle.com/datasets/joniarroba/noshowappointments)

The dataset contains the following columns:
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


# Predictive Maintenance Analysis

Overview: This project leverages data science techniques to predict equipment failures in industrial settings. It includes comprehensive EDA, data processing, and modeling experiments using various machine learning methods (e.g., XGBoost, Balanced Random Forest, etc.) with advanced sampling technique

## Project Structure
  main/v2: Contains the current production-ready version of the project.
  EDA & Processing_v2.ipynb: Notebook for exploratory data analysis and data cleaning
  modeling_v2.ipynb: Notebook for modeling experiments with various sampling techniques and hyperparameter tuning.

## Data Processing & Feature Engineering
The data is first cleaned by dropping irrelevant columns (UDI, Product ID) and standardizing column names. Categorical variables such as type and failure_type are encoded using OrdinalEncoder and numerical features are scaled using RobustScaler (for outlier-prone features) and MinMaxScaler for the rest. The dataset is then split into training and test sets using stratified sampling.

## Modeling and Evaluation
Multiple models were experimented with, including XGBoost, Balanced Random Forest, and Logistic Regression. Various sampling techniques (SMOTE, SMOTETomek, TomekLinks, ADASYN) were used to address class imbalance. The best performing model was XGBoost with TomekLinks sampling and hyperparameter tuning, achieving a macro F1-score of 0.918 on the test set. Additional threshold tuning and calibration steps were also explored.

## Installation & Usage
  1. git clone https://github.com/kealankuar/predictive-maintenance-ML.git
  2. pip install -r requirements.txt
  3. Open the notebooks in Jupyter Notebook or JupyterLab or any other suitable IDE
  4. Start with 'EDA & Processing_v2.ipynb' for data analysis and data processing
  5. Execute modeling_v2.ipynb to run the modeling experiments

## Results & Discussion
The project demonstrates that handling class imbalance is critical for predictive maintenance. The tuned XGBoost model with TomekLinks sampling provided the best balance of recall and precision for identifying failures. Future work may include further hyperparameter tuning, additional feature engineering, exploring other ensemble methods and also balancing between recall and precision. I determined that recall might be more important than precision for a predicitive maintenance scenario as having false positives is better than missing a true positive.

## Future Work
Future work may focus on further threshold tuning, experimenting with additional models (e.g., LightGBM, CatBoost), and deploying the model in a real-time monitoring system.

## Dataset Credits
This project uses the **Machine Predictive Maintenance Classification** dataset from Kaggle.  
Dataset URL: [https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

Dataset provided by **shivamb**. Please refer to the dataset page for the license and additional usage details.

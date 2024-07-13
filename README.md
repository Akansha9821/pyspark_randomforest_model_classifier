#### Project Name: CC_Integration

#### Project Information
- **Created By:** Akansha Rana, Aman Chandel
- **Networks:** NPW, Frozen, Ambient, Chilled

#### Purpose
The purpose of this project is to build a classification model for the DRP Salvage project. The project involves the following steps:
- Data preparation
- Feature engineering
- Model training & evaluation
- Fail-safe implementation
- Model Testing

#### Acceptance Criteria
- The model should be able to identify the accuracy of classified Executed Recommendation.
- Building a random forest model and predict the probability of classification. If the classification recommended is '1' and the probability is >80%, then the value will be taken as 1. Otherwise, it will be taken as 0.
- For this model, we need columns like Time duration before move (days), Value of move. After that, train the model for daily DRP output and merge predictions with DRP output recommendations from 2023 to 2024 data and then save the record into Azure Data Lake Storage (ADLS).

#### Training Recommendation Input Paths
- **From 2023-2024:** `solutions/ift/ift/outbound/CC_Integration/Outbound/training_data`
- **Daily Training Recommendation Input:** `solutions/ift/ift/outboundCC_Integration/Inbound`

#### Output Folder
- `solutions/ift/ift/outbound/Tredence_CC_Integration/Inbound/DRP_All_Model_Predection/`

#### Code Functionality
Certainly! Let's break down the functionality described in the code:

1. **Importing Necessary Modules and Packages**: The code begins by importing required modules and packages such as `os`, `pyspark.sql.functions`, `pyspark.sql.types`, etc. These modules are necessary for various data processing, model building, and evaluation tasks.

2. **Importing Azure Connections Script**: It imports a script named `AzureConnections` from a specific path. This script likely contains connection details or functions related to interacting with Azure services, which are required for the project.

3. **`process_columns` Function**: This function processes columns in a DataFrame. It performs several transformations on the DataFrame columns, such as converting certain columns to specific data types (`date`, `int`), calculating a new column based on existing columns (`Time_duration_before_move`), and selecting a subset of columns (`DaystoSalvage`, `Distance`, `Value_of_move`, `IsExecuted`). Finally, it assembles features using `VectorAssembler`.

4. **`oversample_minority` Function**: This function oversamples the minority class in the training data to balance the dataset. It identifies the minority and majority classes based on the `IsExecuted` column, calculates the oversampling ratio, and then samples the minority class with replacement to achieve the desired ratio.

5. **`train_model` Function**: This function trains a Random Forest classifier model on the input training data. It initializes a Random Forest classifier with specified parameters (e.g., `numTrees`, `maxDepth`), defines a binary classification evaluator to evaluate the model's performance, fits the model to the training data, and returns the trained model along with the evaluator.

These functions collectively handle data preprocessing, model training, and evaluation tasks necessary for building and assessing the classification model for the DRP Salvage project.

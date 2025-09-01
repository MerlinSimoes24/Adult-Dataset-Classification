# Description:
This project involved data analysis and classification using the Adult Dataset from the UCI Machine Learning Repository. The objective is to predict whether an individual's income exceeds $50,000 based on various demographic and work-related attributes.  

The dataset contains approximately 48,000 observations and 14 features for a sample of the US population of adults. The demographic attributes in this dataset include age, educational attainment, and occupation, while financial variables pertain to capital gains/losses and hours worked per week. 

# Requirements:
In order to run the files in this project, the following packages are required:  
ucimlrepo, pandas, numpy, seaborn, matplotlib, scikit-learn, scipy, xgboost, umap-learn, tensorflow, keras, keras_tuner

# Loading the data:
The dataset can be loaded using the following:

from ucimlrepo import fetch_ucirepo  
adult = fetch_ucirepo(id=2)  
X = adult.data.features  
y = adult.data.targets  

The dataset is also available in this repository in case the above method does not work.

# Requirements:
Python 3 is required to run the code.

# Executing the Files:
1. Install Jupyter Notebook
If you don’t already have Jupyter installed, install it using pip:

pip install notebook

Or, if you're using Anaconda, Jupyter is already included in the distribution. You can skip this step if it's installed.

2. Launch Jupyter Notebook
Open a terminal (or command prompt) and navigate to the directory containing your .ipynb file. Then, run:

jupyter notebook
This will start the Jupyter Notebook server and open it in your default web browser.

3. Open and Run the Notebook
Locate your notebook file (.ipynb) in the Jupyter Notebook interface.
Click on the file to open it.

Execute cells:
Click a cell and press Shift + Enter to execute it.
Alternatively, use the "Run" button in the toolbar.

4. Execute Notebook in VS Code
If you prefer using Visual Studio Code:

Install the Python extension for VS Code.
Open the .ipynb file in VS Code.
Click on the play button (▶) beside each cell to execute it.

# Description for Each File
1. RandomForest.ipynb: Data Analysis, Correlation between Features, Feature Impact on Target, Independence Assumptions, PCA, t-SNE, UMAP, Random Forest Classifier Implementation with K-Fold Cross Validation and Hyperparameter Tuning

2. Gradient_Boosting.ipynb: Train XGBoost model with hyperparameter tuning using GridSearch and find importance of features

3. Hierarchical_clustering.ipynb: Unsupervised learning using hierarchical clustering to find clusters

4. Kmeans_clustering.ipynb: Implements KMeans clustering using the elbow and silouette method to find the optimal number of clusters

5. Neural_Network.ipynb: Implements Neural Network model with dropout and early stopping to avoid overfitting. 

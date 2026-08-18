{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1f0a8d71",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# Import visualization packages\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Import relevant ML packages\n",
    "import xgboost as xgb\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import OrdinalEncoder\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.metrics import roc_auc_score\n",
    "import shap"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3ce31a53",
   "metadata": {},
   "source": [
    "## Refining the Initial Model\n",
    "\n",
    "The initial model identified Practice/Facility Size as having the highest feature importance. Subsequent investigation shows that missing facility-size information was strongly associated with default decisions. This raised the question of whether the feature was capturing a characteristic of default disputes rather than a general predictor of arbitration outcomes. This notebook therefore excludes default decisions and examines whether the predictive structure changes among non-default disputes. In other words, we are analyzing disputes where both parties are present in the negotiations."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "651b9899",
   "metadata": {},
   "source": [
    "### 1. Loading and Initial Cleaning of the Data\n",
    "\n",
    "Load the IDR data and prepare the fields for modeling by cleaning values, standardizing data types, and removing unsuitable predictors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "efbc48d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# This is our target variable.\n",
    "target_column = \"Offer Selected from Provider or Issuer\"\n",
    "\n",
    "# Drop columns that will not be used as predictors, including identifiers,\n",
    "# descriptive fields, and fields excluded from the analytical dataset.\n",
    "columns_to_drop = [ \"QPA as Percent of Median QPA\",\n",
    "                    \"Provider/Facility Offer as Percent of Median Provider/Facility Offer Amount\",\n",
    "                    \"Health Plan/Issuer Offer as Percent of Median Health Plan/Issuer Offer Amount\",\n",
    "                    \"Prevailing Offer as Percent of Median Prevailing Offer Amount\",\n",
    "                    \"Item or Service Description\",\n",
    "                    \"Dispute Number\",\n",
    "                    \"DLI Number\",\n",
    "                    \"Provider/Facility Group Name\",\n",
    "                    \"Provider/Facility NPI Number\" ]\n",
    "\n",
    "# These are columns linked to the target outcome, and will also be dropped.\n",
    "outcome_derived_columns = [ \"Payment Determination Outcome\",\n",
    "                            \"Default Decision\",\n",
    "                            \"Provider/Facility Offer as % of QPA\",\n",
    "                            \"Health Plan/Issuer Offer as % of QPA\",\n",
    "                            \"Prevailing Party Offer as % of QPA\" ]\n",
    "\n",
    "float_columns = [ \"Length of Time to Make Determination\",\n",
    "                  \"IDRE Compensation\",\n",
    "                  \"Prevailing Party Offer as % of QPA\",\n",
    "                  \"Provider/Facility Offer as % of QPA\",\n",
    "                  \"Health Plan/Issuer Offer as % of QPA\",\n",
    "                  \"Quarter\" ]\n",
    "\n",
    "# Load in Independent Dispute Resolution data\n",
    "dataset_path = r\"C:\\Users\\choia\\Downloads\\concatenated_IDR_PUFs_2024.csv\"\n",
    "idr_data = pd.read_csv(dataset_path, low_memory=False)\n",
    "\n",
    "print(f\"Rows: {len(idr_data):,}\")\n",
    "print(f\"Columns: {len(idr_data.columns)}\")\n",
    "\n",
    "\n",
    "# Initial data cleaning\n",
    "idr_data[\"Quarter\"] = idr_data[\"Quarter\"].astype(str)\n",
    "\n",
    "idr_data[\"Length of Time to Make Determination\"] = idr_data[\"Length of Time to Make Determination\"].astype(str)\n",
    "\n",
    "\n",
    "# Replace source placeholders with missing values.\n",
    "idr_data.replace( [\"+\", \"*\", \"N/R\"], np.nan, inplace=True )\n",
    "\n",
    "# Remove formatting characters from percentage fields.\n",
    "percentage_columns = [ \"Prevailing Party Offer as % of QPA\",\n",
    "                       \"Provider/Facility Offer as % of QPA\",\n",
    "                       \"Health Plan/Issuer Offer as % of QPA\" ]\n",
    "\n",
    "for column in percentage_columns:\n",
    "    idr_data[column] = (idr_data[column]\n",
    "                        .astype(str)\n",
    "                        .str.replace(\",\", \"\", regex=False)\n",
    "                        .str.replace(\"%\", \"\", regex=False) )\n",
    "\n",
    "# Remove currency formatting from IDRE compensation.\n",
    "idr_data[\"IDRE Compensation\"] = (idr_data[\"IDRE Compensation\"]\n",
    "                                    .astype(str)\n",
    "                                    .str.replace(\"$\", \"\", regex=False)\n",
    "                                    .str.replace(\",\", \"\", regex=False) )\n",
    "\n",
    "# Convert the dataframe to categorical dtype.\n",
    "idr_data = idr_data.astype(\"category\")\n",
    "\n",
    "# Convert designated numeric fields to float.\n",
    "idr_data[float_columns] = idr_data[float_columns].astype(float)\n",
    "\n",
    "# Retain records with non-negative prevailing-party offers.\n",
    "idr_data = idr_data[idr_data[\"Prevailing Party Offer as % of QPA\"] >= 0]\n",
    "\n",
    "# Remove fields excluded from the analytical dataset.\n",
    "idr_data = idr_data.drop( columns=columns_to_drop )\n",
    "\n",
    "print(f\"Rows after cleaning: {len(idr_data):,}\")\n",
    "print(f\"Columns after cleaning: {len(idr_data.columns)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b64382a0",
   "metadata": {},
   "source": [
    "### 2. Removing Default Decisions to Refine Dataset\n",
    "\n",
    "Define and verify the binary arbitration outcome used as the model target and identify variables that could directly encode the outcome."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "18e14a3e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original number of records: 1806014\n",
      "Refined number of records: 1397948\n",
      "Records removed: 408066\n",
      "\n",
      "Default decision distribution in refined data:\n",
      "Default Decision\n",
      "No     1397948\n",
      "Yes          0\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "refined_data = (idr_data[idr_data[\"Default Decision\"] == \"No\"].copy())\n",
    "\n",
    "print(\"Original number of records:\", len(idr_data))\n",
    "print(\"Refined number of records:\", len(refined_data))\n",
    "print(\"Records removed:\", len(idr_data) - len(refined_data))\n",
    "print(\"\\nDefault decision distribution in refined data:\")\n",
    "print(refined_data[\"Default Decision\"].value_counts(dropna=False))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f9c402fc",
   "metadata": {},
   "source": [
    "### 3. Target Definition\n",
    "Define and verify the binary arbitration outcome for the non-default dispute population."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "4015b219",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Target proportions:\n",
      "Offer Selected from Provider or Issuer\n",
      "In Favor of Provider/Facility/AA Provider    85.55\n",
      "In Favor of Plan/Issuer                      14.45\n",
      "Name: proportion, dtype: float64\n",
      "Target value counts:\n",
      "Offer Selected from Provider or Issuer\n",
      "In Favor of Provider/Facility/AA Provider    1196011\n",
      "In Favor of Plan/Issuer                       201937\n",
      "Name: count, dtype: int64\n",
      "\n",
      "Original and encoded target values:\n",
      "                                 original  encoded\n",
      "                  In Favor of Plan/Issuer      0.0\n",
      "In Favor of Provider/Facility/AA Provider      1.0\n"
     ]
    }
   ],
   "source": [
    "target_column = \"Offer Selected from Provider or Issuer\"\n",
    "\n",
    "print(\"\\nTarget proportions:\")\n",
    "print(refined_data[target_column].value_counts(normalize=True).mul(100).round(2))\n",
    "\n",
    "# Inspect target variable before encoding\n",
    "\n",
    "print(\"Target value counts:\")\n",
    "print(refined_data[target_column].value_counts(dropna=False))\n",
    "\n",
    "target_data = refined_data[[target_column]].copy()\n",
    "\n",
    "# Encode target and verify mapping\n",
    "target_encoder = OrdinalEncoder()\n",
    "\n",
    "target_encoded = target_encoder.fit_transform(target_data).ravel()\n",
    "\n",
    "# Verify the original-to-encoded mapping.\n",
    "target_check = pd.DataFrame({ \"original\": refined_data[target_column].astype(str),\n",
    "                              \"encoded\": target_encoded })\n",
    "\n",
    "print(\"\\nOriginal and encoded target values:\")\n",
    "print( target_check.drop_duplicates().sort_values(\"encoded\").to_string(index=False) )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1b4b3782",
   "metadata": {},
   "source": [
    "Among non-default disputes, 85.55% of selected outcomes were in favor of the provider/facility and 14.45% were in favor of the plan/issuer. The target encoding was verified so that 0 represents a plan/issuer win outcome and 1 represents a provider/facility win outcome."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b3dcc2a4",
   "metadata": {},
   "source": [
    "#### 4. Feature Definition\n",
    "This step is identical to the initial analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "08d4d327",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Number of features:\n",
      "17\n",
      "\n",
      "Categorical features:\n",
      "['Type of Dispute', 'Provider/Facility Name', 'Provider Email Domain', 'Practice/Facility Size', 'Health Plan/Issuer Name', 'Health Plan/Issuer Email Domain', 'Health Plan Type', 'Dispute Line Item Type', 'Type of Service Code', 'Service Code', 'Place of Service Code', 'Location of Service', 'Practice/Facility Specialty or Type', 'Initiating Party']\n"
     ]
    }
   ],
   "source": [
    "# Remove target variables from feature dataset\n",
    "feature_data = refined_data.drop( columns=[target_column, *outcome_derived_columns])\n",
    "\n",
    "# Identify categorical features.\n",
    "categorical_columns = ( feature_data\n",
    "                        .select_dtypes(exclude=np.number)\n",
    "                        .columns\n",
    "                        .tolist() )\n",
    "\n",
    "# Ensure categorical features use pandas categorical dtype.\n",
    "for column in categorical_columns:\n",
    "    feature_data[column] = ( feature_data[column].astype(\"category\") )\n",
    "\n",
    "print(\"\\nNumber of features:\")\n",
    "print(len(feature_data.columns))\n",
    "\n",
    "print(\"\\nCategorical features:\")\n",
    "print(categorical_columns)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "082218a0",
   "metadata": {},
   "source": [
    "### 5. Train/test split\n",
    "\n",
    "Divide the refined dataset into training and test sets."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "8ff8336d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Training rows: 936,625\n",
      "Testing rows: 461,323\n"
     ]
    }
   ],
   "source": [
    "random_state = 42\n",
    "test_size = 0.33\n",
    "\n",
    "feature_train, feature_test, target_train, target_test = ( train_test_split( feature_data,\n",
    "                                                                             target_encoded,\n",
    "                                                                             test_size=test_size,\n",
    "                                                                             random_state=random_state,\n",
    "                                                                             shuffle=True ))\n",
    "\n",
    "print(\"\\nTraining rows:\", f\"{len(feature_train):,}\")\n",
    "print(\"Testing rows:\", f\"{len(feature_test):,}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ed6623eb",
   "metadata": {},
   "source": [
    "### 6. 4-Fold Cross-Validation\n",
    "Use stratified four-fold cross-validation to assess whether model performance is consistent across different subsets of the training data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ed374736",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cross-validation AUC scores:\n",
      "[0.8730657191159485, 0.8731452614274513, 0.8756555557359015, 0.8765632812833253]\n",
      "\n",
      "Mean cross-validation AUC:\n",
      "0.8746074543906566\n",
      "\n",
      "Standard deviation of cross-validation AUC:\n",
      "0.0015361259049078918\n"
     ]
    }
   ],
   "source": [
    "params = {  \"objective\": \"binary:logistic\",\n",
    "            \"eval_metric\": \"auc\",\n",
    "            \"max_depth\": 6,\n",
    "            \"learning_rate\": 0.3,\n",
    "            \"verbosity\": 0 }\n",
    "# Define cross-validation\n",
    "\n",
    "cv = StratifiedKFold(n_splits=4,\n",
    "                     shuffle=True,\n",
    "                     random_state=42)\n",
    "\n",
    "cv_auc_scores = []\n",
    "\n",
    "for fold, (train_index, validation_index) in enumerate(cv.split(feature_train, target_train), 1):\n",
    "\n",
    "    cv_feature_train = feature_train.iloc[train_index]\n",
    "    cv_feature_validation = feature_train.iloc[validation_index]\n",
    "\n",
    "    cv_target_train = target_train[train_index]\n",
    "    cv_target_validation = target_train[validation_index]\n",
    "\n",
    "    cv_dtrain = xgb.DMatrix(data=cv_feature_train,\n",
    "                            label=cv_target_train,\n",
    "                            enable_categorical=True)\n",
    "\n",
    "    cv_dvalidation = xgb.DMatrix(data=cv_feature_validation,\n",
    "                                 label=cv_target_validation,\n",
    "                                 enable_categorical=True)\n",
    "\n",
    "    cv_model = xgb.train(params=params,\n",
    "                         dtrain=cv_dtrain,\n",
    "                         num_boost_round=175,\n",
    "                         evals=[(cv_dvalidation, \"validation\")],\n",
    "                         verbose_eval=False)\n",
    "\n",
    "    cv_predictions = cv_model.predict(cv_dvalidation)\n",
    "\n",
    "    cv_auc = roc_auc_score(cv_target_validation,\n",
    "                           cv_predictions)\n",
    "\n",
    "    cv_auc_scores.append(cv_auc)\n",
    "\n",
    "\n",
    "print(\"Cross-validation AUC scores:\")\n",
    "print(cv_auc_scores)\n",
    "\n",
    "print(\"\\nMean cross-validation AUC:\")\n",
    "print(np.mean(cv_auc_scores))\n",
    "\n",
    "print(\"\\nStandard deviation of cross-validation AUC:\")\n",
    "print(np.std(cv_auc_scores))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "328b88b7",
   "metadata": {},
   "source": [
    "The refined model achieved a mean 4-fold cross-validation AUC of 0.8746 with a standard deviation of 0.0015. Performance was consistent across folds, with AUC ranging from 0.8731 to 0.8766. Although predictive perforamnce was lower than in the initial model, the results indicate that the model retains substantial predictive ability after excluding default decisions."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9d4a9ebd",
   "metadata": {},
   "source": [
    "### 7. Refined XGBoost model\n",
    "\n",
    "Train the refined XGBoost classifier on non-default disputes using the established model specification."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c7f4e2b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create XGBoost data matrices\n",
    "dtrain = xgb.DMatrix(data=feature_train,\n",
    "                     label=target_train,\n",
    "                     enable_categorical=True)\n",
    "\n",
    "dtest = xgb.DMatrix(data=feature_test,\n",
    "                    label=target_test,\n",
    "                    enable_categorical=True)\n",
    "\n",
    "# Trained refined model\n",
    "params = {  \"objective\": \"binary:logistic\",\n",
    "            \"eval_metric\": \"auc\",\n",
    "            \"max_depth\": 6,\n",
    "            \"learning_rate\": 0.3,\n",
    "            \"verbosity\": 0 }\n",
    "\n",
    "refined_model = xgb.train(params=params,\n",
    "                          dtrain=dtrain,\n",
    "                          num_boost_round=175,\n",
    "                          verbose_eval=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1278df34",
   "metadata": {},
   "source": [
    "### 8. Model Performance with AUC\n",
    "\n",
    "Evaluate the refined model on the test set and compare its performance with the cross-validation results."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3f563e02",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Refined model test AUC:\n",
      "0.8817789547941066\n"
     ]
    }
   ],
   "source": [
    "refined_predictions = refined_model.predict(dtest)\n",
    "\n",
    "refined_test_auc = roc_auc_score(\n",
    "    target_test,\n",
    "    refined_predictions\n",
    ")\n",
    "\n",
    "\n",
    "print(\"Refined model test AUC:\")\n",
    "print(refined_test_auc)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "faee5761",
   "metadata": {},
   "source": [
    "The refined model achieved a held-out test AUC of 0.8818, while its mean 4-fold cross-validation AUC was 0.8746. Performance was lower than in the initial model, consistent with removing a highly predictive subset of default disputes, but the refined model retained much of its predictive performance."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d03716d",
   "metadata": {},
   "source": [
    "### 9. Feature Importance\n",
    "\n",
    "Examine the refined model's feature importance to identify which predictors emerge after default decisions are excluded."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "0ba5a9ac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA5EAAAHHCAYAAAAmmtx8AAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjgsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvwVt1zgAAAAlwSFlzAAAPYQAAD2EBqD+naQABAABJREFUeJzsnQm4TdX7x5d5Hss8RqLIlMySuVKJ5iSalUqhZJ4jJWnQQIQkKSQRkqE5s6SQNCEiU8m8/89n/Z51/vvse86953INx/1+nue4zt7rrL3W2nuv/X7X+66103ie5xkhhBBCCCGEECIG0saSSAghhBBCCCGEkIgUQgghhBBCCJEs5IkUQgghhBBCCBEzEpFCCCGEEEIIIWJGIlIIIYQQQgghRMxIRAohhBBCCCGEiBmJSCGEEEIIIYQQMSMRKYQQQgghhBAiZiQihRBCCCGEEELEjESkEEIIIUQy+eWXX0yaNGnMs88+G7dt9+abb9o6UBchhEgOEpFCCCFEAAzrWD4LFy486W33yiuvmBtvvNEUL17cHrNdu3aJCoJInz///DPJ41x++eVRf//jjz+ehJoZM3LkSFtukTSbNm0yDz30kLngggtM1qxZ7eeiiy4yHTp0MKtXr1YTCiFOKelP7eGEEEKIM58JEyaEfR8/fryZN29egu0XXnjhSS/L008/bfbt22eqV69utm7dmmT6/v37m/POOy9sW+7cuWM6VtGiRc3gwYMTbC9cuLA5WSLy3HPPjSqMxf+YOXOmufnmm0369OlN69atTaVKlUzatGmtuJ86daodaEBklihRIllN1qZNG3PLLbeYTJkyqamFEMlCIlIIIYQIcPvtt4d9//rrr62IDG4/FSxatCjkhcyePXuS6a+88kpTrVq14zpWrly5TksdUxLP88yBAwdMlixZzNnAxo0brdBDIM6fP98UKlQowSADYhxRmVzSpUtnP0IIkVwUziqEEEIcB//++6/p3LmzKVasmPXklC1b1s6PQ8T4QfwRhjhx4kSbJnPmzOaSSy4xixcvjuk4iAfySA54Lo8ePWpSmoMHD5o+ffqY888/39aZuj/xxBN2u5+xY8eahg0bmvz589t0hF3iLfNTsmRJ8/3331uR7MJmCamFvn37RqxzpDl85HP11VebOXPmWPGMeHzttdfsvt27d5tHH300dI4oN6Lr2LFjYfm+88479pzkyJHD5MyZ01x88cVmxIgRMbfL8OHD7Xni2PXr1zdr1qwJawvKvGLFigS/e+qpp6yI27x5c9S8hw4daq818gkKSMA7+cgjj9g6OghvxbtbqlQpe70VLFjQ3HXXXWbnzp0xt+fnn39uvd/8nnzwxgshhEOeSCGEECKZIBSvvfZas2DBAnP33XebypUrWxHz+OOPW0GAqPCDUJo8ebI19hEzeI6uuOIK8+2335oKFSqkaPs3aNDA/PPPPyZjxoymWbNmZtiwYaZMmTIx/RbhuWPHjrBtiAg8oAgv6oy4uO+++2wo73fffWfrun79ejN9+vTQbxCM5cuXt+kROR9++KF58MEHbR7M4YPnn3/ePPzwwzbvHj162G0FChQ4rjqvW7fO3Hrrreb+++839957rxXr+/fvt4KO88F2vLlffvml6datmw0L5viAh5nfNmrUyApM+OGHH8wXX3xhOnbsmOSxEVeIduqFBxTxiYCmbajPDTfcYPcxiFClSpWw37IN4VykSJFEQ1kRvzVq1Ii5PajTzz//bO68804rIBHrr7/+uv2LVz2pQYmffvrJlptru23btmbMmDFWlCK0Oa9CCMGDUAghhBCJ0KFDB9yLoe/Tp0+33wcOHBiW7oYbbvDSpEnj/fTTT6FtpOOzdOnS0LZff/3Vy5w5s9eyZctktXu2bNm8tm3bRtw3efJkr127dt64ceO8adOmeT179vSyZs3qnXvuud5vv/2WZN7169cPldX/ccebMGGClzZtWu+zzz4L+92rr75q033xxRehbfv370+Qf7NmzbxSpUqFbStfvrw9bpA+ffqEtbdj7NixdvumTZtC20qUKGG3ffzxx2FpBwwYYNtr/fr1YduffPJJL126dKE26dixo5czZ07vyJEjXnKgDBw3S5Ys3h9//BHa/s0339jtjz32WGjbrbfe6hUuXNg7evRoaNvy5cttOuoUjT179tg01113XYJ9u3bt8v7666/Qx9/mkdp/0qRJNq/FixfH1J7+dNu3b/cyZcrkde7cOcbWEUKc7SicVQghhEgms2bNsmGIeBb9EN6Kbpw9e3bY9lq1alkvjgOvWIsWLaz3MqXCTm+66SYb8njHHXeY6667zgwYMMDmTwjjoEGDYsqDUEa8WP4P4aowZcoU630sV66c9Va6D143wCvr8M9H3LNnj02HVxDvGN9T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UgghhBBCCCFEzEhECiGEEEIIIYSIGYlIIYQQQgghhBAxIxEphBBCCCGEECJmJCKFEEIIIYQQQsSMRKQQQgghhBBCiJiRiBRCCCGEEEIIETMSkUIIIYQQQgghYkYiUgghhBBCCCFEzEhECiGEEEIIIYSIGYlIIYQQQgghhBAxIxEphBBCCCHEWUTJkiVNmjRpEnw6dOhg9x84cMD+/5xzzjHZs2c3119/vdm2bVvo96tWrTK33nqrKVasmMmSJYu58MILzYgRI5I87vLly02TJk1M7ty5bd733Xef+eeff6Lme+mllybI4/PPPzd16tSxvydNuXLlzPDhwxOk27x5s7n99ttD6S6++GKzdOnSiOVq3769rf/zzz+fZDsNGTIkLM3q1atNvXr1TObMmW25hw4dGrb/+++/t+3n8goeA/bt22ceffRRU6JECVvW2rVrmyVLloSladeuXYKyXHHFFWFprr32WlO8eHFblkKFCpk2bdqYLVu2hPZzXsmHtkifPr257rrrEpRl6tSp9hzly5fP5MyZ09SqVcvMmTPHJBeJSBEXvPnmm7ZDSoy+ffuaypUrm7OJX375xXYiK1eutN8XLlxov+/evdvEM3RwkTo2IYQQQpw4CJStW7eGPvPmzbPbb7zxRvv3scceMx9++KGZMmWKWbRokRUirVq1Cv1+2bJlJn/+/Oatt96yIqlHjx6mW7du5qWXXop6TPJo3LixOf/8880333xjPv74Y/tbnvnR8u3SpYvd/vrrr4fSZMuWzTz00ENm8eLF5ocffjA9e/a0H3+aXbt2WaGZIUMGM3v2bLN27VozbNgwkydPngTlmjZtmvn6669N4cKFI5a7f//+YW318MMPh/bt3bvXNG3a1Io/yv7MM89Ye/N1X1n2799vSpUqZcVnwYIFIx7jnnvusedgwoQJ5rvvvrN50lYIYT+IRn9ZJk2aFLa/QYMG5t133zXr1q0z77//vtm4caO54YYbQvuPHj1qReojjzxi848E7YqInDVrlq0TeV5zzTVmxYoVJll4QiSDtm3belw2fDJkyOCVLl3a69evn3f48OGT2o779+/3tm3blmiaPn36eJUqVTopxy9RokSo3v7P4MGDvZPJkSNHvK1bt4bad8GCBfa4u3btiqmsmTNntt9vvPFGb/78+d6Zwu7duxOtQ3LZs2ePre+OHTu81MqhQ4e86dOn27+pEdVf51/Xv+5/9X/R+/+OHTtam+3YsWP2GYwNN2XKlND+H374wT5Hv/rqq6h5PPjgg16DBg2i7n/ttde8/Pnze0ePHg1tW716tc13w4YNiT6/69Wrl2gf37JlS+/2228Pfe/atatXt25dLyn++OMPr0iRIt6aNWusPTR8+PCw/ZG2+Rk5cqSXJ08e7+DBg2HHLlu2bMT0kfLDhk2XLp03c+bMsO1Vq1b1evToEWZjt2jRwksOH3zwgZcmTZqI135y8rvooousPZ8c5IkUycaNkmzYsMF07tzZjsgwMhOJQ4cOpUgLM6rCyNXJ5vDhw1H3BUeqgqNVJ4N06dLZUS1CEpKDKysjVePHj7deXEakBg0aZM4EcuXKlaRnWQghhBAnDrYYnr+77rrLRjPhfcLe8XuqCBklTPKrr76Kms+ePXtM3rx5o+4/ePCgyZgxo0mbNm2Y/eZCVBMjkgfRgYfsyy+/NPXr1w9tmzFjhqlWrZr1rGIfVqlSxYwaNSrsd8eOHbPhno8//rgpX7581PzxIBISSx7Ys0eOHAntoz0uu+wyWy9Hs2bNrH2FNzQWyA8PISGofmibYLsQcUZ9ypYtax544AGzc+fOqPn+/fffZuLEiTY0Fo/s8UI7EW6b2LmNRPIsUyGMMZkyZQq567nACRPgZibMgZAFQi2JcX/55Zdt2k2bNlnXfceOHe3NmDVrVhs7/txzz9k4/Llz59oY7z///DNMWJCe33366ac2nJVYcn8YJzc9MfKEEdx00002tjvI6NGjbXgDZSBWHff+gw8+GAoVPe+888w777xjRo4caUMvXn311bCwCz85cuSIGqbATU84AKEbTz75pPnxxx9tjDl501l36tTJhixcffXVtky0AZB+4MCBZs2aNVYw8hvmHJQuXTqsjHSgyQnV9ZeVhwIdILHzvXv3tmEPdE5ACAudK3MU6Dzatm1ry+NE6+WXX27j6inbuHHjbCfK/ttuu82Gmrz33numQIEC5sUXXzRXXnml/Q0dJXMgOG+cU45Pm3M+He46mT59eug4FStWtB0s7cNxmL/AAEVyqDF4vjmSPptJjWRK55mh1Y2p0HeOOXg0jUltqP46/7r+df+r/5tj1g26OkH/yLOWZ66zb3g285wNDubyPGdfJBBxkydPNh999FHUfrhhw4bW3kGI8cz/999/rU0EDGxHAtsLItleRYsWNX/99ZcVYdgDhIQ6fv75Z/PKK6/Y43Xv3t2G72LjUS9sGXj66aetPcP2aLCvatWq1gaijtiylBUb1bUVdliwndy+xMSv3ybDvhswYICdW8rvCVPFJib01++kIaSY4xGmSr2wrUiHHebo2rWrDSvG/q1Zs6aZOXOmORGeffZZO28VWzo5SESKE4aRFP9Iyfz58+1EXRd/TyfCqA03EDf59u3bbUeACEEcNmrUyHZkxHbffffdISFCZxXNc0Y8OB0KQrVu3bo2xvyFF16wMekORmcQTdxojC4hxO69914ba+86GKCDQ2iSJjhKlFwoE8dDJHIz8kFIv/322/YGbdmypRVcdACubegAEVDsp7ykYQ6kfyQvJaBDpwP74IMPzBNPPGFF7VVXXWU7bryVCF/ahzbwizfEI+m//fZbe07cwAHlpINDyDPS99tvv9l6M6JFx888C0b26JQRlYjYxDoojkNb8EChw6RczHcgbj/SaCcf/5wFyJTWM+nSERmT+qDu/r+pDdVf599/HaQ2dP3r+nfXQaSIKgZnscMYbGe/87QF03qeZ+2v4HYGulu0aGHnJTJgHi1q64ILLjBvvPGGtRkQYwgfbD1EE3lHypdFdgBbMMhnn31mbSPmM2KrIbhcemwNPJFPPfWU/Y4NR344A7DxGMBnUJ6FfvC+RgO7w4Ethgi9//77zeDBg639llJMmDDBeoKLFCli2wXhSl0op+OWW24J/Z8BfMqDUwFHhb99GPzHXv71119Nv379zB133GGFZGL1jAb2KXlgGyY34k8iUhw3dAgIRlZ08od1ItKcNwkIL2C1KIQK+wChxSReRonoXLhxuJCdiCRfRs3wWEaCla9I69LjHfvkk0/scRx9+vSx4tBNFGdkh4nXr732WpiIxMPpn0weDYQfHagfJnOzYpeDciB8gLLRiTKa5MQtXsAFCxaERGSwfmPGjLGdPOWsUKGCSUkYZaODwLsJeF9ZZYxzQcdDKAuT4ikbYtaJ2EqVKoXqTX3wAJ977rlWcAJpGQ1k9TJGxAipoENy0O6IQoR/YiKSzpJzBmXKlLHl4jqIJCLp3P3HcPSscsxkzXrUpGYGVDtmUjOqv85/akbXv65/Fkvxw8A9z1Ke7W4f4oMQV57LRIQ52E6Ipj+P33//3doAPIuJiArmH2m6CnYWNhwizK1WyvdI+RKJhICJhPMAIqhYOZYBbiciGZi+6KKLwtLj5cMh4QQodScayoFAZhoW5XG2UJAaNWpYkc1+oraI6vKvWgvue8Eo0WmRQAwS/YXzgIFvyn/zzTeHOT+CsA9766effgoTkWzjg2inzthyCG2cNcmBaDmcOgz6R1uEJzEkIkWyYbSDTocRJUaCCG30e6642f2x46yshRBxAhIQWvyWmHJEZOvWra0AQcSwehZexObNm0edN0eehDv64eZBoAE3KeINIefEDtAx0MH5YSQrFhj5CYZbMKIUFEIO6oVnzt9BsA2PnoN5pYgwvG87duywbQJ49VJaRDrh70aqaEPazD9yxXlh1O+PP/4Idbz+OjF6hneRc+yvE9BZO/AQI4ipx3///WcfVkmF4/qPA3Sw/jz9IGb9o4d0yHSiA1ekNUcy/H/IR2qCEWgMyF5L05qDx1JhOKvqr/Ov61/3v/o/s6z3FQnWSGAAuVevXqGpKjzriUziOxFJgD1G6Oidd95phRSweiqRRNhSwddexAoRZ0Q4YUM5m86fLxFN0USkH+wjfwQSdaDMftavX29XUQUipILCCG8s26ljNFwkmPPKYSexMi02r5t3SKRd2bJlYwplDYItzAexjhMm+LoQP9hiRPphD0XD2Y3+tokFwmnxjCIksbePB4lIkWwIZcDzhFBE8AUXffGLxVhhDiWjNFzMLlySjud4ce8kwgvqOkOHP648OeVl1Mcfux4J/8RmxFlwojPb3A0PeGPp8Cgnbck+xGNKLUjkh46IB0Qwtj8pItUhWE9w9eIcsmQ3XmA6X+YCMD/CzXtIznH8beWH0c1IYSaIpyOpcD5gsA1S45xIh+qv86/rX/d/au7//M9SnqFEgRF95Ra4cfYMAo6wU8QSU5CIKOOZzRQhIDSU11AgvBCAbtoSNpRbg4JBcUIp8XS6QXWiiFjoBWcDYovfIkDdb4L5sjgMMJBOOdxANAPZREi5V1Iwb88/t5FXlHAcwlmJcqIsvHbDvXqDAW8+fmgbvIduXQiipLBNsGuxVfhOvrx70glEHCVEPtFeeHMpP2Gyw33vrcRmI4LM/Z/pQohR2sDZjQhGBvI5Np5F6k79nKDFbuU4RKhRRhwhnB9+T1sBZWVaGOeI8pGGwQHsZ78XkrJQDtqWBXPca+LcYD6Rf1wT1AMb2c2D5RoJOloSJVlruYpUT1LLBUfa//rrr9vlkf/555/Qto8++shLmzat9+eff4a29e3b1y53PHnyZC9XrlzegQMHQvvGjh1rtzlq1apll5r2U7NmzbBXfBQuXNjr379/1LJu2rTJLiu9YsWKJM9rUktAR3r1RrDMwdeQ8DoKfrN48eLQ/s8++8xumzZtWsQyxvqKj0hl7dWrl11i2i2z3b17d7tENct9O15++WUvR44coeW569evb5cFTyp/f5kfeughr2HDhmH7GzVqFHZugtdJpOOwn3SxoFd86BUXesWHXvGhV3zoFR96xUf4ax7mzJljn8/r1q1L8Nz877//rB2FfZY1a1b7Cg1eKea3VyK92gwbwOFsEmwVR5s2bby8efN6GTNm9CpWrOiNHz8+7LjR8i1evHgozQsvvOCVL1/elitnzpxelSpV7Ks2/K8OgQ8//NCrUKGClylTJq9cuXLW3kyMoP2ybNkyr0aNGtZW45VoF154offUU0+F2Z+watUq+zoRjsPrQoYMGRK239lqwQ+2jQPbtlSpUrZdChYs6HXo0MG+asX/GpCmTZt6+fLls69foaz33ntvmJ3M61J4xQrtS1lKlizptW/f3r7GJFjPSOVxUK5I+2O1uRzyRIqTDqGqzHVj1IOwV7xhjHgRUuBCIV069rOYDnMHE5vQzCIxhJYSikpIA+GvhEf4Q0cZ0WHUilEVVrzC1b906VIbQuAPhYwVRnOCq5YRrupGzpILo0iMkjFqRqgCoZ9uFbMTxZWV8AtWpmVpb+apMpfQjYqxYirzAjgXTHwnLITzRNucyKI+zGdk5JNRN7yeTCZn5Cy5HlAhhBBCHD94/P43zpsQQkzx+PGJBPZYUiukM58xmD/P/8QI5st0FOw0VuN3YJfE8go1VrznEyvBeZAsbsNcwqRgug1zLKNRsmTJqO3scIstRgMvIHZTYjCViJXvkyLafE8HC/WkBHpPpDjpILS4MXCrE7aKQGSCMCEPfhA31atXtwu0ICgTg8nIuPBx9V9yySV2MjhhsH6YLIxwGjt2rL3xeL8QIbLHK2aYu4jY8384/vGCUHOvACGElRCKaO/bPN6y0qaIdd7t5CbWOwg9YZI7ISDMWWWOKeEawcWDkgurmrFQEeeIMAlCYNxrVYQQQgghRPyTBnfk6S6EEEKcCG4kkzkVwTkQqQW8zgwKsEjCibx0OF5R/XX+df3r/lf/F3/9v3t+M9h9vJFd4vQgT6QQQgghhBBCiJiRiBRCCCGEEEIIETMSkUIIIYQQQgghYkYiUgghhBBCCCFEzEhECiGEEEIIIYSIGYlIIYQQQgghhBAxIxEphBBCCCGEECJmJCKFEEIIIYQQQsSMRKQQQgghhBBCiJiRiBRCCCGEEEIIETMSkUIIIYQQQgghYkYiUgghhBBJ8sorr5iKFSuanDlz2k+tWrXM7NmzQ/vvv/9+U7p0aZMlSxaTL18+06JFC/Pjjz9GzGvnzp2maNGiJk2aNGb37t2JHrdkyZI2nf8zZMiQ0P6FCxeaVq1amTvvvNPkzp3bVK5c2UycODEsj1GjRpl69eqZPHny2E/jxo3Nt99+G5amXbt2CY5zxRVXhPb/8ssv5u677zbnnXeerSN17dOnjzl06FAozYEDB2w+F198sUmfPr257rrrItbp5ZdfNhdeeKHNp2zZsmb8+PFh+y+//PIEZeHTvHlzu//w4cOma9eu9jjZsmUzJUqUMM8//7zZsmVLgmN99NFHpkaNGvZY1D1YpkjHeeedd8LSHDx40PTo0cMeJ1OmTPacjBkzJkXbVwgRX0hEnkHwIIz1gcrDIiV588037cNXnHlwTUyfPj1kxPB95cqVJp7p27evNfSEEPEDog/xtmzZMrN06VLTsGFDKxS///57u/+SSy4xY8eONT/88IOZM2eO8TzPNG3a1Bw9ejRBXogxBGms9O/f32zdujX0efjhh0P7vvzySyumEFWUDTF5xx13mJkzZ4Y9X2+99VazYMEC89VXX5lixYrZsm3evDnsOIga/3EmTZoU2ocgPnbsmHnttddsnYcPH25effVV071791Aa6opYe+SRR6yQiibGu3XrZvtB8unXr5/p0KGD+fDDD0Nppk6dGlaONWvWmHTp0pkbb7zR7t+/f79Zvny56dWrl/377rvv2rogpv28//77pk2bNrZNVq1aZb744gtz2223JSgT581/vKDQvOmmm8z8+fPNG2+8YdatW2fbBfGbku0rhIgzPJEobdu29Vq0aJFg+4IFCzyab9euXSnWgsE8x44d6+XKlStBuhIlSnjDhw9Pdj3Im0+GDBm80qVLe/369fMOHz6c6LFSgv3793tZs2b1NmzYcFKPczKpX79+qP38n/vvv/+kH3vr1q3egQMH7P83bdpkj7tixYqYypoxY0avcOHC3tVXX+29//773pnCvn37vB07dqRYfnv27LH1Tck8441Dhw5506dPt39TI6r/6Tn/efLk8UaPHh1x36pVq+x9+dNPP4VtHzlypO2n5s+fH9NzNJZnXvD8X3XVVd6dd94ZNf2RI0e8HDlyeOPGjUvyeZ8YQ4cO9c4777yI+6LlV6tWLa9Lly5h2zp16uTVqVMn6nGoP+X9559/Iu6n3s8884xtz19//dVu4/lepEiRqOfHwW+mTZsWdf/s2bPtc3vnzp1erKRU+8aK7v/47f/d85u/Ir5If7pFrDh1MALIaCNhKbNmzbIjnxkyZLAjoieTefPm2RCY888/33z++efmTIYQIdokEvfee68dDfeTNWvWk16mggULJvs3rqxHjhwxf/zxh5k2bZq55ZZbbDjR66+/bk432bNnt5+Upsbg+eZI+mwmNZIpnWeGVjemQt855uDRNCa1ofqfvPP/y5D/hVD6weM2ZcoU8++//9qw1iBs53lD6CdeKcfatWtt3/TNN9+Yn3/+OeYy4AEdMGCAKV68uPWkPfbYYzZcNBp79uyx4aLRwJNHf583b96w7XjU8ufPb0My8bQOHDjQnHPOOYkeJ5hHUvAMzpw5c9g2vJeEf0Z7BuEBpA8ndDWxOhGp4qKK8FDiCUybNq2pUqWK+fPPP20EyDPPPGMqVKgQ9lvsgXvuuceUKlXKtG/f3nouyQtmzJhhqlWrZoYOHWomTJhgy3Dttdfa80G5T2b7CiHOXBTOmoIgkJgTQKfKQ5NwFh6kDjpfOuIcOXJYYcCDcPv27RHzoqOlE+cB5eYOEPri76DvuusumxcP1ViEAfMYOC6C7oEHHrChNjwcIrFx40YbplSgQAFr7F966aXmk08+SRBW+9RTTyVZjg8++MA+cCJBeE2DBg3s75ljQzgUYVLw66+/mmuuucY+bHholS9f3orfaOG3hHy6h57/2FWrVrUPbB6OhA0hrBykJ7SI8nGMQYMGRW0/BCPt5/9QZn+YKSFF7hqgzdavX2+WLFlizzvteOWVV5q//vorlCf7mjRpYs4991yTK1cuU79+ffvgjxbOGiuurISf1axZ0zz99NM2BIt5K/7z+N1339kHOeXlQX7fffeZf/75J7Qf0UlYE+eZa4E2d+L08ccftwYCx8BY9ENY2QUXXGDLQbsTcoVBES2c1R3n2WefNYUKFbJlwajx/yZohO3duzfsA5nSelZMpMpPWi91t4Hqf9LOP/eh+9A/0ZfxPEFsICTLlCkT2v/iiy+GBonor/nQh7GPvgUhNHjwYHufu77Yn3+kD33BW2+9ZebOnWuFDv1Rly5dEqRzeb399tu2b7399tuj5kn/VbhwYdvnum08E5nn9/HHH9tnAc9hBl+Z5xgpD8J2qS9lirSf0Fc+we0cZ/To0VZIM5/y66+/tt/ZR4hnMD3huoSz0k9Gq8++ffvMuHHjbLgr/TnbeP64/vbJJ5+0zxGeM8y33LZtW+i3zOukzThX9MMPPvignTLj9mMPYN/wvOB800+/99579vyfzPZN7ieWa+ls/sRz/UV8Ik9kCkEnS2fIqBqdJELhoYcesh9nYHOjMHLHPALEY6dOnexDwQkjP7Vr17adeO/eve38A/B7boYNG2bzYi4GnTmikM7aP0chKXjQsLhBJHjYX3XVVbajx1hg0j+CjrIgFmMtBw9Q5qVEE0GtW7e2I6QIOeZ7MNfPjcJiOPCAXbx4sRV4jGAnx3v12Wef2XkxL7zwghV2nCNEEvDQdPCAZZSb9k5sZDsWyJd8aCPENQMFCOQRI0ZYQcW8Es4p9QUe/G3btrWGCFFFtCftvmHDBvu7lITjdO7c2c614WHOAEezZs2sFwGDi2sSY4hrFpHu+PTTT61Q5Dwwn4a5TBg1l112mTWCJk+ebBfUQAyTDig7eWBEYHjgGWXbE088EbV8zKXBsOTvTz/9ZG6++WYrNPltEIxQBgSC9KxyzGTNmnD+VWpiQLVjJjWj+qf8+fc/o3iOISLoP5j7xnw7nhPO28gAEJ6uXbt22X6fhWDoXzNmzGifjYgYBgbJk74BEIeJ9e0MSHE8PvQxiMOXXnrJ1K1bN4HX7rnnnrPPYZ5FDETyCcI8QaIzSEf/5nB97u+//27zffTRR61QwgNXqVKlsDx4drLQTPXq1W2/Fek5ThQIZQ7uY7CUuZt16tSx/T6Dczw3KRP9X3CAdOTIkXbwF7si0nEQ4wwUAiLQpXEDkpwDBlLxRN5www12MST6T/p/4BnMWgx8GKxlAJlzygAgIDgpJ89rNwjKYADtQt7YCCndvicS+ZSaicf64xQR8YlEZAwggoIPuOBCARi1dLB0isDILOKFBwOCgQ4cUeGgc2Y/3ioEWzB/Hrg8bBnBjRTOiNBgtNB5fZjgz8MnFhHJw4AJ8ix84F+cwA8dur9TRyjyUMBziciItRyMsAIrw0Xit99+syOW5cqVC7Wbf9/1119vF0xwbZYceEgy+op4cr+nHggZv4hE6OH1TQoe5IwW+8G7x3l3MDruHswdO3a0Cw3Q1hgLgADzCzS8gH7w5GJALFq0yFx99dUmJSGkCWMMrykw8swIMAMELkQKw4zBAgwSPI+At5Frld9zXnng0+m7xSQIh8ZIZKQawwJ69uwZ5rGmXVjtLzERiWHJ8RlM4HrAOKHtIolIjskgjANPJEbswBVpzZEM6UxqBA8UAqrX0rTm4LFUGM6q+p+087+m7//6tCBE2zB4SkQJA0lB6AMJXaSfQdwwgIZHjX4d/jcd738DXPTV/n45MRBU9BX0E+5Zg7jl+UNfxF8GxCKByOQ5hrGNmEsK+jIiRXjWOVgBlYG4Ro0a2TBT+sZIIKYQZv7fOlq2bGnLjEBDhPJsIUqEPtSfHyKUwVDaJlI+5MFz5r///rMDoghAJ6wZuKQtGLx0zyCgDyeKJlJ+DqJqqB8CkQFi7BS3qA8QpsxzgrBY/3M7Jdr3eKAdOCaDmdGmpJzNxHP9XSSRiD8kImOAcEvnOXLggWE01MFDdPXq1WHLivOAxBO3adMmOzeDkUc6edIySss+J5YuuuiiZJ04/6p2TmhGC40NimE6G46NePKHyPrhgcE+lgYnvIaRTh5SlDU55SCcFDEU7SGLEOBhT6gvD2UeUiyb7gwURpMZpWYfhkdyVvNzK9H5Q1QR/xg0iCA3n5FQ01hALDLy7McJLYe/fG6fE8Fum799MCB4iBLWw3bKR9mC7ZxScE26kF9CsRgo8M+xwdDg2sDj7MrPyLT//LHdP58G0Yf3wV8vvJMIT7y/XEtcPy70Nxoch7wcGFbOUxEEwyY4+g0Yz0dS4XzAYBukxjmRDtU/5c9/YkYpfQrPlEhp6EvYT7/GfqIgeI44iIBgcJWoEfr9WI1fVjSlTypSpEjoNwy84flC2PDciATiiVBYBlAJ808KPInuVSTuOMwxxFDnuUH4qL/PCkIZ+USrF9sRY0CYKM/KYL+GN5fwfYR2MB/anecS/SzPSdqTNC4dg7fkx35CWN1v8M4yqBqtXAh9BvXc4DaRPAhiyuG2YddQN8rv8kmJ9j1R/PVPjcRj/eOtvOL/kYiMAYxsFoUJdn5+MJQZiUX4BCG00YUO8kFo8g4thALf/e+YOt6bDmHgRGlSYhgvJ2GGiYVu4jliVIuwJepO6CthMMGyJlUORiT97/MKglBFzCJWCbFhtBWPFaO0iEvah308IPH2Eu6J95SHlxvFdgTj6jkneCODS56Df1GDxBYq8INnOHgdBPG3hxNrwW3+9sEw4CFKuKt7/xbhpcdzTSQFhhxhsni/k0Okc5zYeSfEDcPGhUvRbpxTzl1KX9NBvunWKNUu0sD1TxgbXqPU+FBW/U/++ScCgHndPNMIxSeagQEwRAOL5DB4xGsdeL7xjKTv59nhvExugNCxY8cO+5dBVhfCyeIyeN6IQkAk0p8waOvmzvOdRXUYxEXoANEveOCIXuDZQdgm8KxzC7sgLvGEUmaiI1waN3/TPS8YrGQwFOFF5AR9vosuQUAixuireTb657f7I4aYekEf/vfff9t2cq9kcvPAmatIPRF5DCjjvUO4IUqD4OnEixvs17jeeSYTssoAMf07eVEvBvqoOwN3hIvyXCVSg3ITagzOq8hrRRjMRPTxXOS57+acOnhGE8VDxA5txHkjgogBALewTkq0rxAivpCITCFYvIUHRzSRgUcFscBD1c0dcQvIRIOHQKT3a6WkGI4GHjzma/JABh4ALgwyVhAsjHgyapsYhFjywTAgLIc5pO64tBUPQT4YMCwMg4jESOHhjDh3IjD47kTOCR61WOt8OqCdCZN1RhZzRZxhldJgoGBkuFAyDDdCa/1tSHlc2OrxwnxJjBW/1zbSvCQhRHxBtAECj+gUBoeIvEBA0scT4olHkTnh9DMIGeZN0x8Q0horRGLQb7tBQQbWGIRiwBFPGJ4vnhX+cHb6Nn6Ht4yPg+kkiFxgABVhh/Dyg8AibzyKRBORFyGoDLQiiBFPzjuIwGK+Nh83/9vhH9SkP/f3ecw59Kfhuc6gGvVE8COQaSfElx/2M02AQdQgCFq3MF7wnbuIaud5RDQyYMzcVbzACFfmKToBzvFffvll26aUj+clotY/jQARSN159uKBRdASIovn15ES7SuEiC8kIlMI5gMyksd8QTxobiEYOl7mbjByiyhkARUEEaOOdJ6JwQMF8caILGGHhF+eildKAHMcCD1ifhweIVbXTK5XiFBWwlCjlZkHGqOZPHQwDBi5JhzHiRzmlzLqjcDEKOHB6JZs50FIvszLw/vLSLV/riEwKkp4EG3PMRBHhLjS9v6HX6xgpLjRVQcPP/cwPt52dqv2Mi+A9oi2ZPrxlNX/ig/mxhDmhcECeAt5wOMN5SHPqDpGAsZGMEw3uXXCy47hh9cTTzLHF0LEN3jFooEoiLToS2IgdIIRJcFtDAa6ufXRoO9ngJHjI+AieWKTGgSl30UQJwYDq3ySIqlj8RxbsWJFkvkwmBdsH7994N/nPPHB+vN/vKZ8IsGcVj5JwfzTxBZtSYn2FULEF3rFRwrBiCxzMghTYf4AI4+IGB6sgOeMBx3zHpj/iEcyWqfuX6EVwckqlfye+QanCkYiEUeUASFJuAkP8+SQ2Ks9gJFJvLOMbCMUGdlENLpVNxmtZYVWHrg85EiD1w4IUWLJdx6azDmcNGlSgvmdlJkwH0ZxETOIfIQUXrLjASOFeXr+D57TEzXKEMi0LeINQZycUfukykr4GOG8DGgQaubaDxDhPNQJuaJ9ENospMCgx4nAOWdUmwEVRsgZYWcQQgghhBBCnB2k8aINcwlxAhCSiYjBC3YiXi0hYgEvLuF1XHepfU5kNE/M2Y7qr/Ov61/3v/q/+Ov/3fOb96IntQCfOLOQJ1KcFPBu4c2UgBRCCCGEEOLsQnMixUnBLZYjhBBCCCGEOLuQJ1IIIYQQQgghRMxIRAohhBBCCCGEiBmJSCGEEEIIIYQQMSMRKYQQQgghhBAiZiQihRBCCCGEEELEjESkEEIIIYQQQoiYkYgUQgghhBBCCBEzEpFCCCGEEEIIIWJGIlKcNNKkSWOmT59+1rXwwoULbd127959uosihBDHxSuvvGIqVqxocubMaT+1atUys2fPDu1//fXXzeWXX273JdXfHTx40FSuXNmmW7lyZaLH3bhxo2nZsqXJly+fzfumm24y27ZtC0tz7bXXmuLFi5vMmTObQoUKmTZt2pgtW7aEpVm9erWpV6+eTVOsWDHz7LPPhu2n7JQn+GnevHkoTbt27RLsv+KKK8LyKVmyZII0Q4YMCUvjeZ49/gUXXGAyZcpkihQpYgYNGhTav3XrVnPbbbfZ/WnTpjWPPvpognZ58803ExyHujkOHz5sunbtai6++GKTLVs2U7hwYXPHHXckaBfaNqm2c/z0008mR44cJnfu3CYa77zzji3LddddFzWNECJ1IhEpkoX/oZshQwZToEAB06RJEzNmzBhz7NixsLQ8OK+88spT1sI8hBN7GCYHDIfnn38+4r7atWvbuuXKlcuc7Lb2P7gxiiIZHyebX375JaIx5v/Q9kKI+KFo0aJWDC1btswsXbrUNGzY0LRo0cJ8//33dv/+/futoOrevXuSeT3xxBNW1CTFv//+a5o2bWr7jE8//dR88cUX5tChQ+aaa64Je340aNDAvPvuu2bdunXm/ffft8LzhhtuCO3fu3evzadEiRK2/M8884wZMGCAmTNnTijN1KlTbT/tPmvWrDHp0qUzN954Y1iZqKM/3aRJkxKUu3///mFpHn744bD9HTt2NKNHj7ZC8scffzQzZsww1atXDxPZiOaePXuaSpUqRW0fRLX/OL/++mtoH+dj+fLlplevXvYv9aN9ENx+eE4k1nZ+UXrrrbdaIZ5Y39+lS5dE0wghUi/pT3cBRPzBQ3fs2LHm6NGjdgT5448/tg/R9957zz4806f/32VVsGBBczaSMWPGs7ZukWCUH4PGgaHEOf/kk09C2062oBZCpCwINz94zvBOfv3116Z8+fKhASsiLxID7+XcuXOtYPF7MiOBaESYrFixwgomGDdunMmTJ48VlY0bN7bbHnvssdBvEIpPPvmkHVBD+DB4OXHiRCs+GbykP6a8iEkE4IgRI+zv8ubNm8CjljVr1gQiEs9hUv053rpoaX744QfbbojUsmXL2m3nnXdegkFJVy7KHA3EdbTj0MfOmzcvbNtLL71kxepvv/1mvY7As5g2itZ2DgRtuXLlTKNGjcyXX36Z4Hg831u3bm369etnPvvsM0XeCCESIBEpko3/oUvYTtWqVU3NmjXtwwiP1D333BN6IE6bNs0+wHjgd+rUyRoau3btsh7M9u3bm27duoXSjhw50opQjBYeiEOHDg2NoLKN0Wl+67yNhE1VqVLFbNq0yRomd955Zygv6NOnj+nbt68dBe7Ro4c1MAjJqlChgnn66aftiO3xECwLdcbgmjx5sv37+++/m7p161qh7R7swEj1sGHDbHkxKh555BHz4IMPxuyVXLRokf04Y8Tlg/Hy+OOP2wc9YU6M0A8fPtyce+65Nh31JASKUXgMNoyugQMH2vCqhx56yIp/zseLL74Y0XPM7/yGTfbs2e1AAdto91KlSplvv/3WVKtWLZQGLy5loIyLFy+27TVz5kx7vtevX29D32gPzoXj888/t/vxilB2wrIG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aV9++WWvSZMmXqZMmbySJUt6kydPDsvv888/9y6++GIvc+bMXr169bwpU6bY323atCmUpn379t4555xjt/fp08duO3TokNe7d2+bZ4YMGbxChQp5LVu29FavXh21niVKlAjV0/+ZMGGCt2DBAvv/Xbt22bRjx471cuXKFfZ76h28vSZOnGjbIGPGjF6ePHm8yy67zJs6dWqibd2iRYvQ93Xr1nk1a9b0smTJElbv9evX2/rkzp3b7itXrpz36KOPeseOHbP769ev73Xs2DFB/YYPHx71fCUG7VqpUqUE29944w2bx7fffhu23bXXhx9+6JUvX97Wv3r16t6qVavC0vE7zn/27NntdVKxYkVv0KBBXizs2bPHHmPHjh1eaoXrfPr06fZvakT11/nX9a/7X/1f/PX/7vnNXxFfpOGflJOkQhwf/ndKiviEEXy8ycEVeiO94/NkTMzHO8FiHak9nNU/hyo1ofrr/Ov61/2v/i/++n/3/Ga1YaY9ifjh/1/iJoQQxwEhxG5OEiFdQgghhBDi7EYiUghxQjCPkfk7l19+uZ0PKYQQQgghzm60sI44I1BUdfzCeyH5RANxqfMrhBBCCHH2IE+kEEIIIYQQQoiYkYgUQgghhBBCCBEzEpFCCCGEEEIIIWJGIlIIIYQQQgghRMxIRAohhBBCCCGEiBmJSCGEEEIIIYQQMSMRKYQQQgghhBAiZiQihRBCCCGEEELEjESkECJF+eWXX0yaNGnMypUr1bIirhk8eLC59NJLTY4cOUz+/PnNddddZ9atWxfa//fff5uHH37YlC1b1mTJksUUL17cPPLII2bPnj1h+cyfP9/Url3b5lOwYEHTtWtXc+TIkZjK4HmeufLKK+09NX369Ihpdu7cac477zxbvt27d4fte/nll82FF15oy0c5x48fH7Z/6tSpplq1aiZ37twmW7ZspnLlymbChAlhaf755x/z0EMPmaJFi9p8LrroIvPqq68mKMdXX31lGjZsaPPJmTOnueyyy8x///0X2l+yZElbD/9nyJAhYXm8++67tgxZs2Y1JUqUMM8880yC4xw8eND06NHD7s+UKZPN98033wxLQzt06NDBFCpUyKa54IILzKxZsxItCx9+49i4caNp2bKlyZcvn63PTTfdZLZt2xbav3Dhwoh58FmyZEnEcyWEEGcLEpHitNOuXTtr/Pi/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9dd73333nffpp596pUqVsmn9bNiwgYldXsWKFb277747bN/YsWO9XLlyhb6/8847XubMmb0xY8Z469at83r37u3lyJHDq1SpUijNqFGjvCxZsngjRoywaVavXm3TDxs2zO7ftGmTPd6KFSvCjtW8efNQmqTqvXLlSpvH888/723cuNEbP368V6RIEbtt165dNg3Hzpgxo/fAAw94q1at8n744Qdv5MiR3l9//WX3169f3+vYsWMozxIlSnjDhw8PfSevadOm2f8fPnzY1mngwIHen3/+6e3evTuU7ujRo16xYsXssYYMGRL1HEU7zqFDh7zSpUvbdqVMjipVqnhNmjTx1q5d63399ddevXr1bBmilTFa2y5YsCCsXT7++GMvffr0Xt++fb01a9bY/CdNmuT16NHDi4U9e/bY/Hbs2OGlVjhn06dPt39TCneegh/uAXdeI334nWPJkiVe06ZNvbx589r7smbNmt6sWbPCjsNvuK8dW7du9dq1a+cVLlzYXoNly5a19+GxY8dCadavX++1atXKy58/v5c1a1bv4osvtveOv/5t2rSxx+U+oC/hngzC9R2pDtTPsXnzZnus7NmzewUKFLBl27lzZ4K2qly5sj0WfZq/Pkm1Jezfv9+2U758+bwMGTLYe/Lee++197Yf6te5c2dbb9qzcePG9p5x579nz54Rj0N+jqeffjp0f9M+l19+ue2L45mTcf3HE6q/zn+8Xv/u+c1fEV9IRIpTKiIBAdegQYOQAYOhtG/fvgTpqlevbjuWoHETFJEwaNAg79xzz7VGHsd94oknwkQkTJw4MWTk5cmTx7vsssu8qVOnRhU6//zzjy0jgjbWej/33HNWJCOsmjVrZo1Wv1iChQsXerVr1/YyZcrk5c6d26Zz+5MjIp04RiymTZs2TOxBr169vHTp0nlbtmzxkiJ4HHj77bft8fz5Ll++3KtWrZptlzJlynhTpkxJsoyxiEgnJGkX2i5nzpz2/L/++uteLEhEyoiUES0jOl6N6JRA17+u/3i9/iUi45c0/OP3TAoh/gcrQ7IwBau4xiPMhyJEj0VMznaYU8EiKoTWpvZwVkKNU2N4kOqv86/rX/e/+r/46//d85u53e71UCI+0JxIIaLAPCpeOxJv0BGz8AivSUkNAlIIIYQQQpxaJCKFiELTpk3jsm14L9y3335rF+Bg8Q8hhBBCCCFSEolIIc4ykvM6DyGEEEIIIZLL/95DIIQQQgghhBBCxIBEpBBCCCGEEEKImJGIFEIIIYQQQggRMxKRQgghhBBCCCFiRiJSCCGEEEIIIUTMSEQKIYQQQgghhIgZiUghhBBCCCGEEDEjESmEEEIIIYQQImYkIoUQ4hTQt29fkyZNmrBPuXLlQvs3btxoWrZsafLly2dy5sxpbrrpJrNt27ZE8xw8eLC59NJLTY4cOUyRIkXMU089ZdatWxeW5s8//zRt2rQxBQsWNNmyZTNVq1Y177//flia9evXmxYtWphzzz3XHrtu3bpmwYIFCY735ptvmooVK5rMmTOb/Pnzmw4dOoT2HThwwLRr185cfPHFJn369Oa6666LWOaXX37ZXHjhhSZLliymbNmyZvz48WH7Dx8+bPr3729Kly5tj1OpUiXz8ccfh6XZt2+fefTRR02JEiVsPrVr1zZLly4NS0NZgu19xRVXhKUZNGiQ/W3WrFlN7ty5I5b3kUceMZdcconJlCmTqVy5ckznlQ9t7fj+++/N9ddfb0qWLGn3Pf/88wnyiVSnJUuWRCyTEEIIcbqRiBSnjIULF1oDavfu3Ymmw9CKZGSdCBi/0YxEIU4V5cuXN1u3bg19Pv/8c7v933//NU2bNrX3x6effmq++OILc+jQIXPNNdeYY8eORc1v0aJFVsh9/fXXZtasWebo0aOmefPmNj/HHXfcYYXljBkzzHfffWdatWplBeqKFStCaa6++mpz5MgRe+xly5ZZ4cY2BKjjueeeMz169DBPPvmkFUWffPKJadasWWg/x0b8ILoaN24csbyvvPKK6datmxVe5NGvXz9b/g8//DCUpmfPnua1114zL774olm7dq1p3769Fdf+8t5zzz1m3rx5ZsKECbZOtB0CcefOnWHHY5u/vSdNmhS2nza+8cYbzQMPPJDoebvrrrvMzTffHHFfly5dwo7B56KLLrL5Ovbv329KlSplhgwZYsV8JCLViXbcvHlzomUTQgghTgteCrFr166UykqcQtq2beu1aNEiwfYFCxZ4XB4peV6DeY4dO9bLlStXgnQlSpTwhg8fnux6kDefDBkyeKVLl/b69evnHT58ONFjpQT79+/3smbN6m3YsOGkHudkUr9+fdt2kyZNCtvOeeB8nOns2bPHln/Hjh3emUqfPn28SpUqRdw3Z84cL23atLYejt27d3tp0qTx5s2bF1P+hw4d8saNG2fbYdGiRaHt2bJl88aPHx+WNm/evN6oUaPs///66y/7m8WLF4f27927125zx/7777+9LFmyeJ988skJ9Su1atXyunTpEratU6dOXp06dULfCxUq5L300kthaVq1auW1bt06dL+lS5fOmzlzZliaKlWqeDfeeKNth8TKEIlY7tvEzp+flStXJmjPpPq3aHWqWrWq16NHj5jqQL2nT58eqn9qQ/XX+df1H5/3v3t++59/Ij44Lk/k008/bSZPnhz6zqj2OeecY8OpVq1alZIaV4iYcV6HDRs2mM6dO1tvxzPPPHPSWxDvASFo559/vjnTIVQwGoQO4gVKLI04Mbg2CxcubL1SrVu3Nr/99pvdfvDgQeuFJGTSfz7Spk0b8lbGAh4vyJs3b2gbYZH013///bf1ar7zzjs29PTyyy+3++m7XVgpHkw8kngCCVcljNNd4/wWrxihqEWLFrX9/u+//56s+lNP6uUH7+W3334buu6ipXHtQPnwekZKg+cyGP1APagf3sagp/JkMHr0aHPBBReYevXqxfybxOqUnPMvhBBCnCqOS0S++uqrplixYiHjgs/s2bPNlVdeaR5//PGULqM4Q8CYwTDCsOH8E7bmD5sjDKtatWp2fhYhW7fddpvZvn17xLww7u68806zZ8+e0BwiRJ/fGCaEjLyKFy9uXn/99STLhwHOcRF0GIyEghHCFwnmnzEHrECBAiZ79ux2XhnhecGwWuaYJVWODz74wFx77bURj8OgSoMGDezvmWuGUe7mbv366682XDFPnjx2/hShjoQkRgu/nT59um2n4LGZ44bxiTAhPBCD1EF6QggpH8dgDlg0br31VhtqPGrUqKhpYm23gQMH2jBK0nA+OA9//fWX/S3bmFcXnMOW1PUVCzUGzzcln/zojPqEylajhj2vzO/jnGzatMnWl7lwNWvWtOena9eu9tqn3oRJIiwYGIkFRN4bb7xhRWOFChVC2999910r0BCL3CP333+/mTZtWmjQg2uEc0i4KNcp1xKhq5STaxN+/vlnmz/3A6Hm7733nhWlTZo0sSGhsUL4KyKLkFnP8+w1wHfKt2PHjlAajo/g5pg8X6ZOnRpqB8pYq1YtM2DAALNlyxbbRm+99ZYN6d21a1fYoBLCeP78+Xbgk9BfnlGkP1kgzidOnGjuvvvuZP0uWp2++uqrmM+/EEIIcSpJfzw/Yp6ME5EzZ860I9LM38B4xFASZx+IB4wyxMGYMWOsIHjooYfsZ+zYsTYNhiBGEKP+iMdOnTrZxS2cMPKDoYsx2rt379BCIIgLx7Bhw2xe3bt3twYrorB+/fo271hBjETzPPzzzz/mqquusqIKwxpjE0FHWRCLsZYDI5d7AIEXCbxNVapUsaIhXbp0ZuXKlSZDhgx2H3PBMMAXL15sBQReFH8bJMVnn31mhdoLL7xgxQjn6L777rP7+vTpE0qHOGcuFu3NgifRQOQy541FTdq2bRu2MEhy22348OFWcPTq1cv+n4VdOOcIcrzDiCXKzrw4REws15cfvFV8HHv37rV/M6X1TLp0RMacOTgPm3+eIN48xD9Cjnl6DKjw9+GHH7bnEw8kc/C4dvx5JAZtxcAE8yn96TmniCtEIUISQU+fzfxHFsFBzHFds6APi+lw33AOOK9ffvmlKVSokM2PD+KuYcOGNl/OPc8BRB79vx/uCz7BcjOfEpGEaOa4DEbcfvvt9j5DOJH+2WeftfMgWXSIa4PBEa5HBLjLj/JxrRP9wn1FOzEH0V93FrJxkBdtzl8Es6uDwwnLxNqZNJQ5sTRTpkyxgwIMoCWVV3B/pDpxDSxfvjym8+/SpNZIAtVf599/HaQ24vn6j8cyixMQkYxOE8aEAYFhguEHPGBP5iivODkggoLiJXgeWQUSQcTqgVCmTBlr7CKoEEh4LxAIDgw/9uOpQngE88+YMaPJlSuXNRIjLTSBUHnwwQft/xEcCBEM3FhEJNch3oc5c+ZYozwSLBzCx4FQxDuDgY0xHms58H5AtMETwhXxzrtVOGk3/z4MXQx512bJAa8jRjkGtvs99XjiiSfCRCQGLSIlFqjriBEjrFhAAJ5Iu+HxAgYKuEa4FtxiI7QlnhdWH+X8x3J9+SE99Q/Ss8oxkzXrmdUHRRpEcRBqOXfuXCumgHZHECMiuWcYhMFrm1gegIf8m2++scKdwQgX1okXa+TIkbYt8ZIRjoo3HO8wAyOIR7zl5I/nC080Hzx2nFPCm7lGEfUuP39Z8KDx3e/9hj/++MN6UyOVm0VyEKgch2cJ9Ue4shIp9QY8eQw8IMgIzUWwInL9+RGyzkAMnlvSMDhBOyJqExsowXtPW/ihDTBkEmtnPKOcm8TSUAbaF09rNCgv5ydSPpHqxHWQ1Pn3k1j9UwOqv85/aiYer383DUOkEhHJ6n4Yphh6eHowOIBwqHiYFybCIdwSQ90PBikeAr+RtXr1ahuq5RdreBsIy2OUH8MJrxdp8Xy4VSURS6xWmBwwnB1OaEYLjQ2KYYxBjs016g+R9YOwZd9HH31kDWOM4P/++y80Ry3WcmCQsoqlM36D4I1l1UVCffFEIaJ4dQEQrokRjxHNPox1//GSgnbG8+IPUUX8YyDTKfPaAiDEOFbwLuKJRHxHWrHyeNrNCSQnlv3baEvaNJbryw8rfNK2Dox7BrUGrkhrjmRIZ84k1vT9/xVMg21J/1mnTh0ruoMwWEG4N2Gt0QZPaCOENx5uPIu0FSGmztvNKp+AGPe3Ia/ZYF4jx3X3KZ5g/2AP/6ePJw39Oqul8hvnxSOcFZHHarAc0w+vEEEkRqpXEDzkhFtzH0WC+5k2INw6Wn70N2vWrLH3vL/+QWFLebnXgvkQSstvEisvobc//PBD1DS0PWUg9DaxfLgv6Q+TahtXJwZMYmlH2gkDMlr9z3ZUf51/Xf/xef+7SCKRSkQk3hhCV/FGDh06NGR4YFQ6r42IHwhbDIp/DK6gwYtnCeEThDBGvA7MZeKDEMBrgLDge3LmTDmCnSACLrFXHfjFMF5OFi9JLHQTo5QHDqFz1B1PyA033JCgrEmVA28NoaLRQHBh2CK6mDeMh5CFTfDGIC5pH/YhJDEWCetDwCFKEQiJhXxwTvDGMagTxO+5ixSWmhgMHtAuRBhwn59ou7l5nJG2ubZM6vqKJHb9i9A4FndtbEM2z0RoOzxweAEJ6eRaIGyR9qZtCNtF6HHvMBeuY8eO5rHHHgub39ioUSN77TivL/3t22+/bQcz8OoR+ogw5X2PnBuEO+eJ9Jwz2obQa0I6GXThuIRC81uuR7zG/I55sb/88osVd6Rhvi7zWfGU4fXEo4eQx8PuN1rwsHEtICARbIQrg3u/Iu+jZBEdPPeIJDyvpMHT6PJgAAuPKb/hL/cQ1wnHc2mIMuD+QFz/9NNP1tvP/2kf0hDqzL3BwAyDFIRL46GnLRC9Lh/6KMQwx2EAxpWXdO65Rv5cn3hjGaBxaRCC9DUOBooI/eUcc1790CbOO8z/mRJCPhzD9b2R6kT7cl6SYxSSNt6MyJRE9df51/UfX/d/aj5fqVJEcsIxiIJg8IizE+ZvYQRF8zTj8cB4RVC5+bLBhVOCYIClZPhzJDEcDTx4hApikANGIkZzciC8jTloQS9MEFZq5MP9gTcFseCOS1sx/4sPRjLGOyISIYERjjh3IhBvU/CcMBcxpb3/CFgELeI06I1MiXY7nuvrbICBGc4/9wnnt27dujYcmv8D55JrAFGDeGcuY7BPRQy5BWjARRC4lVYdXGOcJ/pqQiEJe0bccL5o43HjxoW8WwhOpiVwPLyMDFYgGhGm/tBlhB7lQYRxjeDd5Hd+A4A8uSccbk6nGxDhfmeghLryOwZ+mHfpH6xAqBFGy2I+iCzyRKD5F5rCQ0tb0aaEfSIWEZtcn4CIw7NNPRG0DCoxb5Pwa//gA6KZNMHy4gV2bYqIY1GeYBo8j67ciFzmbNLmQQEJDBq43wGCng9tyCJj0epElIEMLCGEEGeNiAQe6iwDz4OeUXNG1wlLOu+88+yItTi7YA4bi2Hg0cCocgvB4JV66aWXrLcIUUjIG4KIMCwMtsTAAMOoZf4ixiphXi4E82RDmB5hZxjWeMWY/5eUpzMIRjahcdHKTJgn3gQ8ddwXGIfM+3ILfhCGSCg4AhOvDIarCznEU0O+zFvDO4d3BiPVDwYwIYC0PcfAsCcslLZ385SPF4QCZeAed6GnKdVux3N9nQ3ggU4MBmAS82pDULD7vdVuTh+iyy88OGeElyYGIc94whID7yOrv/KJtXxBuL6Z9pAYCKvgqzqCsDAQn2ieerypSdUHuKeC91UQJ/ISg3svsded0NcFIwtiqZMQQghxVr3ig9Fv5iNhADPK67xJjBQjJMXZB3PcGI0nHI3wN0bVETGM8APeFIwxVickzAtjmJH2xGC1TgQnKxDye0KjTxWE0RHCRxkQRISV4g1LDom92gPwSOB1YhVShCIGIveMWxCG+4ZFNDCsmY9GGhZBATwRLHSCKCAkkZU7g/M7KTMhiYTCsmgNIoxQcwZ0UgJeixBcgCQl2u14ri8hhBBCCHHmkMZLang0AogEVgG87rrr7Op8eD9YGRIPCCFA/nArIc5GuMaZ/4R30e+pE6dvYj6r/XJeztQ5kSebaJ7I1ILqr/Ov61/3v/q/+Ov/3fObkH4iXsRZ7olkLoh/foeDuSbJfTm4EPEI89bwyklACiGEEEKI1MZxzYlkfheLfATD5lhkIbgUvxBnI26xHCGEEEIIIVIbxyUimQ/JXC7mSxENy5LtzNliRcfRo0enfCmFEEIIIYQQQsSviGT1RFa/Yxl2XmrOe/BYAGPEiBHmlltuSflSCiGEEEIIIYSITxF55MgR+3JrVmVs3bq1FZG8piF//vwnp4RCCCGEEEIIIeJ3YZ306dPb1zK4pf95l50EpBBCCCGEEEKkDo5rddbq1asn+cJoIYQQQgghhBBnH8c1J/LBBx80nTt3tu/Iu+SSS0y2bNkSvDhcCCGEEEIIIcTZx3GJSLd4ziOPPBLaliZNGrtSK3+PHj2aciUUQgghhBBCCBHfInLTpk0pXxIhhBBCCCGEEGfnnMgSJUok+hFCiNRM3759bVSG/1OuXLnQfhYm412755xzjsmePbu5/vrrzbZt2xLNk1WwH3roIVO0aFH7iqWLLrrIvPrqq2Fptm7dam644QaTL18+kzNnTnPTTTeF5fvLL7+Yu+++25x33nk2j9KlS5s+ffqYQ4cOJVp2Pv5pC99//70tc8mSJe2+559/PkF59+3bZx599FH7TOBYtWvXNkuWLAntP3z4sOnatau5+OKLbd68JuqOO+4wW7ZsSZDXRx99ZGrUqGHzyZMnj7nuuuvC9pMvK4bzuikWeuP/q1atSladKE///v1tm2TOnNlUqlTJfPzxx2HHIcqmV69eYe03YMAAG4WTnPN0//3329+yn3PVokUL8+OPP0Y870IIIcRZ44kcP358ovsxBIQQIjVTvnx588knn4StbO147LHHrDCaMmWKyZUrlxUdrVq1Ml988UXU/Dp16mQ+/fRT89Zbb1nxNnfuXDs/HfF17bXXmn///deKpZo1a9p0gOC55pprzNdff23Spk1rhcqxY8fMa6+9Zs4//3yzZs0ac++999rfPvvss/Y3Xbp0sStw+2nUqJG59NJLQ995tVOpUqXMjTfeaOsS7X3C5D9hwgRbRsrduHFjs3btWlOkSBGbx/Lly20ZEWy7du0yHTt2tHVZunRpKJ/333/flvGpp54yDRs2tK+ZIl+/aLviiivM1VdfbQU0YnXgwIFWSP7+++8mQ4YMMdWJ9x5TxlGjRlnBP2fOHNOyZUvz5ZdfmipVqtg0Tz/9tHnllVfMuHHj7PmlnHfeeac9h256R1LnCVhLgFdkFS9e3Pz999/2vDVt2tRG+aRLly7qNSCEEEKcMXjHQe7cucM+2bJl89KkSeNlypTJy5Mnz/FkKYRIYX777Tfvzjvv9AoVKuRlyJDBK168uPfII494O3bsOCVtXb9+fa9jx46n5Fh79uzBFXTK6pYUffr08SpVqhRx3+7du+35mDJlSmjbDz/8YMv/1VdfRc2zfPnyXv/+/cO2Va1a1evRo4f9/0cffeSlTZs2rA04Fn3zvHnzouY7dOhQ77zzzou6f+XKlbZsixcvjri/RIkS3vDhw8O27d+/30uXLp03c+bMqOWNxLfffmuP9euvv9rvhw8f9ooUKeKNHj066m+WLFlif7Nx40Zv+vTp3qFDh7zVq1fbbRs2bIi5TtwnL730Uli6Vq1aea1btw59b968uXfXXXclmiap8xSJVatW2fL89NNP3vFCvV39UyOqv86/rv/4vP/d85u/Ir44rnBWRoz9H0aC161bZ+rWrWsmTZqU8kpXCJEsfv75Z1OtWjWzYcMGe0/+9NNPNqRu/vz5platWtb7cbLwh0aeifmdKmh7vE947PA6/fbbb3b7smXLbOgkXjkHni+8Ul999VXU/PCwzZgxw2zevNmGTy5YsMCsX7/eerDg4MGD9m+mTJlCvyEsEw/k559/HjXfPXv2mLx580bdP3r0aHPBBReYevXqxVx3vIWEfnJ8P4RvJlUWwkxz585tv+OppL7UAW9goUKFzJVXXhnmiSxbtqwNCx47dqxt1//++8+88cYb5sILL7SewFjrRPslVV7OAfcQ7Q6EzLKfMsV6noLgBabshMgWK1YsatsIIYQQcR/OGokyZcqYIUOGmNtvv11zO4Q4zTDfLmPGjDaUDkMYECkY4szF6tGjhw3Lw2CfNm1a2BwzDHjmuLVr185+Z94aaXilT8GCBa0g6t27tw0TBELxpk+fbkMyBw0aZH799Vcb0r5o0SL7GTFihE1HqB5GPQLg8ccfN5999pmdk4ZxPXz4cHPuuefadJdffrmpUKGCDf8kJJA5cxjisVBj8HxzJH34K4dOJb8Maf6/ctSoYd58800rcJin2K9fPytYqPuff/5pz40TSo4CBQrYfdF48cUXzX333Wfn2tE2CCtCLy+77LLQMRFB3bt3t30xAubJJ5+0Yo4yRILBBfJ1oaxBmLs5ceJEm09yyJEjhx2sYL4gYo66MZiBSCaMNtqxuNZuvfVWO5/TDYa4a+y5556z18+wYcPsNYIwQ/xyrIULF9prmJBX9zwiHNUfQpxUnQh/5Ri0J/cIYnHq1Klhq43zm71791rRT9gp+7jmuSdiPU+OkSNHmieeeMKKSK6TefPm2etCCCGESFUi0maWPn3ERRGEEKcOvIwY0Bi3TkA6nAicPHmyNWJjASMdQYRX7bvvvrPz09iGAewXI8xdw+jGuGYxFYx8xCCLlQALiOzevdvOa2O+HMIRrxHCgQVg3Dw+YM7ZAw88EHWOIF4j53kDDHvIlNYz6dL9/yInpxo8YeD3MiKiqlatasUTQsp5u1xaB6IPURLc7kDYI8JoYwYE8IAxWMBCMszvQ5Qizpmz/tJLL1nxcvPNN4fm8wXzxVPGXEIWyGHAINJxmbPJAjksWBOtXBCp3GPGjLFiivmPXBOUg/LgXQym5TvXAPM1X3jhhdB+54VGvLn5hK+//rr12r3zzjv2WuQauuuuu6yIZt4jHnjyuOqqq2x7Be+BaHVCSPN7BCKDK3iQ27Zta699l477BgFKG7NgDp5I5ltyDtxaAEmdJwf1RQwzcIB4ZX4pgy5Bb2isuDImdp7OZlR/nX//dZDaiOfrPx7LLE5ARBKqEzR+GOnGcKlTp87xZCmESMEwSu5JxEsk2E4Y+l9//RVTfiw44sAThNGMAe8XkRj7GNYIRQdelaxZs1rh6qCPQEw4j5ETG4TxIToJMXSepKFDh0Yt0+DBg613L0FZqxwzWbOevvfUzpo1K+o+RASeYRaRob3effdduzKrAw8u5yVSHghmzgNiCnGIV5hzwSI6eB5ZYRVoWz6IatKRPwKxYsWKYfky0EB+tDcL70Qr9zPPPGMXgSEENxoskMNiOZHy6Ny5sxVQpMFrSH6UyZ+W0Fe2s4osAw7+8FEXAszgg/83rNCKdxqBigePa6dbt262zoTEIhCJiiG/YBhuYnVi5do2bdpYkUl53TXtjs1qs4huBlFYtIc0CHHaH096rOcpCOeI8uJxDXoskwvtkZpR/XX+UzPxeP3zfBCpSEQGl1dn1JYHLR4GQo2EEKcf/2sHIhFr6BzeFzw7GzdutPOfMfpduKEDz6NfQEYDzw3Gv188OcjfiUiM/MRAMLAKpgPRhBAduCKtOZLh9K1uuaZvs4jbabedO3faQTZECmGeRG7gLQPmlCPqWekTj1oQ6ke7V69e3YoWx8yZM+1f8mE0FwOiSZMmoVBj2hpRhfAnZNJ5IEnDHHY8vtFWAyX8mPBbPGqunJFgoACvXGJpAIFMfgwAuLSUmfBVRBte5+A1RBlZaZU5j/7fUCeeN2yjnHgbCYtmNVzqxjOJ9kU8+8sVa53ccWg3yufSck8RXu3/Ld75b7/91m6L5TxFAvGJ6IylHRMrb/D8pyZUf51/Xf/xef+7SCKRSkQkIUdCiDMTwiYxon/44Qf7ioIgbMdYJ/yRdEGx6Q8tISyP8Fe8fswZ41UGeCGDg0X+9+0lBmIKzxevSgjCoimx5sfiMf4FZByLuza2guN0g/ignohrQvzxQCHW8DbhscLjhScX7ySC/OGHH7ZzCBFNDsIqEVycQ+pUv359K57xgpEvoY/MGSUU0hkNzOMjLW3JueOVGbyCg7Biv4Dk9/wOD5/D7zEGXs1BPtQjKDTxpOJ9dP8nJJN3RzI44OY8ElLNtYV4JdyZUFvqRCgz5XUCkvBWRBYiCqENePgY5KAuhJjiUcSjR7nxJMItt9xi80Gs4fnD60k98UpyfSIigwZVYnX65ptvbPtUrlzZ/sUryLOONnd58DvmmxJOyys+VqxYYef8Ek5LmljOE/M8GZhB9HIf4q0kT4Qw+Z+oAcjv482ITElUf51/Xf/xdf+n5vMV9xzPkq79+vXz/v333wTbWdadfUKI00vTpk3tqxG4J/1s3brVy5o1q/f444/b7/nz5/defvnl0P7169fbpbbHjh1rvz/77LNeqVKlwvK4++67vVy5ciX5OosmTZp4Dz30UNi27t27e2XLlrWvbkjJV4Ocaa/4uPnmm+0rIzJmzGjPA9/9r2/477//vAcffNC+Eonz0bJlS3tu/PjPA7C/Xbt2XuHChb3MmTPbdhw2bJh37Ngxu5+l3XndRIECBewrRMqUKRO2H8iPfCN9/Bw9etQrWrSoPV+R2LRpU8Q8OHeOyZMn22uHNihYsKDXoUMH+8qRpPLgs2DBglA66tW5c2d7rebIkcNr3Lixt2bNmrDyzJ0716tdu7ZtS9q0YcOGCV6XklSdFi5c6F144YX2VVXnnHOO16ZNG2/z5s1hafbu3WuvTV6Xwzmgfry64+DBgzGfJ/K88sorbX04T5Tptttu83788UfvRNArLvSKC73iIj5fcZESxPP9r1d8xC/HJSJ5F9m2bdsSbMeAY58Q4vSCGDz33HO9evXqeYsWLbLvjJw9e7ZXoUIFr3Llyt6+fftsultuucUazsuXL7fv28P4xrB14uWDDz7w0qdP702aNMmKoBEjRnh58+aNSUTee++93qWXXmrFwl9//WWNeAzofPnyeTfccIN9JyB5fvzxx9boPnLkyFkjIk8H8WxEpASqv86/rn/d/+r/4q//l4hMZe+JRHwSBhdpvlNi7xsTQpwaWJhmyZIldoVJVoEkrI532THnkLlnbk4iYX/MJWTxERYjIQyTOW4OVsQkHJLXdxDm9+WXX5pevXrFVAbyImSQeV6E7bFICiu8cnxW8yScj/llLFZCaC3hjEIIIYQQ4iybE8mKeIhHPhijfiGJUch8J+avCCFOP8wh4/UEDublMS9r9erVdrVIQNQxd82Pf54csEpqcKVUhJ+DuWN8gtBHMC8vksBlYZNo8M4/IYQQQghxlohI3n+FF5JFBFhog0U2HCyCgNHK4hBCiDMP7lnu0a+//tquHinPnxBCCCGEOOkikhcvAyvT1a5dWysqCRFn8AoJIYQQQgghTvkrPljC3HHgwAG7xLuf4DvkhBBCCCGEEEKcHRzXShb79++3C23wjjPe58ZcSf9HCCGEEEIIIcTZyXGJSF4a/emnn5pXXnnFvvB79OjRdr4Vi3SMHz8+5UsphBBCCCGEECJ+w1k//PBDKxYvv/xyO8eK1wOcf/759jUCEydONK1bt075kgohhBBCCCGEiE9P5N9//23fP+fmP/Id6tataxYvXpyyJRRCCCGEEEIIEd8iEgG5adMm+/9y5cqZd999N+Sh5KXhQgghhBBCCCHOTo5LRBLCumrVKvv/J5980rz88ssmc+bM5rHHHrPzJYUQQgghhBBCnJ0c15xIxKKjcePG5scffzTLli2z8yIrVqyYkuUTQgghhBBCCBHvnkg/vCeSBXVatWolASmESDWwOjWDZswL51OrVi0ze/bs0P6NGzeali1bmnz58tn9N910k9m2bVuieR49etT06tXLnHfeeSZLliymdOnSZsCAAcbzvFCaf/75x75iqWjRojbNRRddZF599dWwfF5//XW78BnHTZMmjdm9e3eCYw0aNMjUrl3bZM2aNclpCDt37rTHi5QXkSgXXnihLUvZsmUTrNA9atQou/iaewUUA4/ffvttWBrapV27dnaFb8pzxRVXmA0bNiS7TrT5lVdeaetzzjnnmPvuu8+2l5/58+fbeufIkcMULFjQ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CCHiEYnIMxSMrEgGWbzQt29fa1Q6MHz9RtTll19uHn30UXO2EouoSKy9xOlj0aJFVox8/fXXZt68eebw4cOmadOm5t9//02Q9vnnn7fnOlYQjVu3bg19Jk2aFLYfAfn999/b486cOdMsXrzYCh2/mEJEPfLII1YkRWLTpk2mefPmpkGDBrb/4D675557rJhz7Nq1y9SpU8dkyJDBzJ4926xdu9YMGzbMiibH0KFDzQsvvGBeffVVK8ayZctmBRVCNsgTTzxhChcuHLEsLVq0MA0bNrRloQw7duwwrVq1SpAWoe5vGwS84+mnnzavvPKKeemll8wPP/xgv1O+F198MZSmTJkytq0Qip9//rkpWbKkPW9//fWX3b9lyxbbZohq6vPxxx/btqZvciAoOUfFixe3acgHkUi9uQ788P3WW2819erVM4n14126dEk0jRBCCBGXeCJE27ZtGda2nwwZMnilS5f2+vXr5x0+fPikH7dFixZh244cOeJt3bo1xY+9f/9+L2vWrN6GDRu8sWPHhurr/4waNeqEj7Nv3z5vx44dUeu4c+dOb+/evaHvJUqU8IYPH37Cx92+fbvXvn17r1ixYl7GjBm9AgUKeE2bNvU+//xz71TCuTtw4EDM6fv06eNVqlQp0WvidLFp06aI14n/w7V0OtmzZ48th/+aSym4psh70aJFYdtXrFjhFSlSxJ5r9k+bNi3RfJI6p2vXrrX5LFmyJLRt9uzZXpo0abzNmzcnmd+hQ4e86dOne507d/bKly8flvbmm2/2mjVrFvretWtXr27dulHLcuzYMa9gwYLeM888E9q2e/duL1OmTN6kSZPC0s6aNcsrV66c9/3339vy0y6OKVOmeOnTp/eOHj0a2jZjxgxbJ8oLCxYssL/btWtX1PI0b97cu+uuu8K2tWrVymvdunWC+rt83TXxySef2O+vvfaalz9//rCyrF692qahPwTanu+//fZb1DSOJ554wrv99tvttZ8rV64EZaYPr127tjd69OhTcj8H65/aUP11/nX9x+f97/pq/or4Qp7IKJ4CwrY6d+5sPUTPPPNMRAF+6NChkybu06VLZwoWLGhD1lISPBwlSpSwo/FA2Jx/9J8P3pATJXv27DZcLBp58+a1I/wpDaFyK1asMOPGjTPr1683M2bMsF7PnTt3mlMJ545Qwngk6HEpVqxY2PXB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EDEjESmEEEIIIYQQImYkIoVIAdKlS2emTJmS7DkdOnQw11xzzT++V0pdR/wf/fv3NzVr1jS5cuUyZ555pi3fNWvWhBXRX3/9ZTp37mwKFChgcubMaa699lrzyy+/JFuMffr0MRUrVjQ5cuQw+fLlM5dddplZunRp2Dlr1641LVq0MAULFjS5c+c2devWNfPmzQsdHz16tG1fkV47d+605yxatMjUq1fPtGvXzl6De77wwgtJ0rNt2zbTtm1bm4ds2bKZc88913zxxReh49Hu8+yzz9rjmzdvNrfddpspXbq0/X7ZsmVN7969zeHDhyPmf/369bZM8+bNm+TY4MGDTYUKFex1ihcvbh588EFbxv6yC6aDfPm56667bBq4RtGiRU2/fv3M6tWrw86JlJ/x48eHjm/fvt20adPGlC9f3qRPn9488MADSdIaqQ6yZs2a5LxVq1aZq6++2uTJk8fWOW1qy5YtEctGCCGEiGckIoVIAZFGR7RZs2ahjjadzK+++irsnCFDhtjOaKyk1HX+STm4DnPmzJnN2WefbZ588klz9OjRNCeCFyxYYAXi559/bmbNmmWOHDliLr/8cvPHH3+EzkHkTJs2zbz77rv2/J9//tm0atUq2esiTF566SXz7bffWqFXqlQpe91du3aFzrnqqqtsmc6dO9esWLHCVKtWzX62Y8cOe7x169a2fflfTZo0saIRwQsIlrvvvtv07dvXfPPNN6Znz5729eqrr4bus2fPHnPxxRebTJkymY8//tj88MMPZtCgQVbcOoL3eeONN2z9I5gBgXb8+HEzYsQI8/3331uh+sorr5j//e9/SfJOGd50003mkksuSXJs3Lhx5rHHHrMCFOH1+uuvmwkTJiS5zjnnnBOWHsrQz/nnn29GjRplr/Hhhx8az/PMlVdeaY4dOxZ2Huf4r+Nvf4cOHTJnnHGGLS/KPhqIc/81fvzxx7DjGzZssAMACN358+fbenj88ccjik0hhBAi7vGEEGG0b9/ea9GixSmXyqZNmzwerZUrV/6jkk2p6/yTcmjatKm3fft2b/Pmzd6wYcO8dOnSef369Tul6x09etQ7duzYPy7fSOzbt8+W1e7du1Pkejt37rTXW7BggX2/d+9eL1OmTN67774bOmfVqlX2nCVLlpx0OmfPnm3f79q1y75fuHBh6Jz9+/fbz2bNmhU1baRl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D9dff31C1L+IA9jiQwgh4pVatWp5nTt3Dr0/duyYV7RoUa9///5eWoKf68mTJ4feHz9+3CtcuLD37LPPhj7bu3evlyVLFu+dd96x73/44Qf7veXLl4fO+fjjj7106dJ527Zt8+KNnTt32vwsWLAglN9MmTJ57777buicVatW2XOWLFli33/00Ude+vTpvR07doTOGT58uJc7d27v0KFDXryRL18+77XXXkuovB84cMArV66cN2vWLK9evXpely5d7OdpvQx69+7tVatWLeKxtJ536Natm1e3bt2oxxPtN5B2X7ZsWZvvRKh/kfqREymEiFsYYcWlIYTJkT59evt+yZIlJi2zadMms2PHjrC858mTx4bzurzzl/CtCy64IHQO51NGjNrHG/v27bN/8+fPb/9S97iT/jIg3K9EiRJhZUAIXKFChULn4FTs378/5OjFA7gK48ePty4sYa2JlHfcaBwVf14hEcqA0EzC2cuUKWMdNcITEyXvH3zwgf3twnkjFLN69epm5MiRCfkbyL91b7/9tunYsaMNaU2E+hepH4lIIUTcsnv3btu59v8jCbync5GWcflLLu/8pfPlJ2PGjFaExVv5HD9+3M6FI7ytSpUq9jPykDlzZttJTK4MIpWRO5ba+fbbb+18pyxZsphOnTqZyZMnm8qVKydE3gHhTJg682ODpPUyQAwRfjhjxgw7PxbRdMkll5gDBw6k+bzDxo0bbb7LlStnPvnkE3P33Xeb+++/34ZlJtpvIPPh9+7dazp06GDfJ0L9i9RPxtOdACGEECIWN+q7774LmxOWCFSoUMF89dVX1oWdNGmSad++vZ3/lAhs3brVdOnSxc4FY9GsRKNZs2ah/2cuKKKyZMmSZuLEiXYRmbQOA0c4iP369bPvcSL5DWD+I89BIvH666/b9oArLURqQU6kECJuKViwoMmQIUOSFel4X7hwYZOWcflLLu/8DS4wxMp8rFYYT+Vz77332gUx5s2bZxebcZAHwrwYoU+uDCKVkTuW2sFtOPvss835559v3TgWWhoyZEhC5J2QPdpvjRo1rHvECwHNQir8P65KWi8DP7hO5cuXN+vXr0+I+mfFVVx3P5UqVQqF9CbKb+CPP/5oZs+ebW6//fbQZ4lQ/yL1IxEphIhb6GDTuZ4zZ07Y6DXvmTeWlildurTtCPjzzlwX5vm4vPOXTgad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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "xgb.plot_importance(refined_model, importance_type=\"gain\", max_num_features=15)\n",
    "plt.title(\"Top 15 Features by Gain\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "269dc58d",
   "metadata": {},
   "source": [
    "After excluding default decisions, the feature importance results have changed substantially. Provider Email Domain became the highest gain feature, followed by Health Plan/Issuer Email Domain and Dispute Line Item Type. Practice/Facility Size fell from the highest-gain feature in the initial model to 14th in the refined model, with its gain decreasing from approximately 1,918 to 9.55. This substantial reduction suggests that much of the predictive signal associated with Practice/Facility Size in the initial model was related to the default-decision population."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d81c5779",
   "metadata": {},
   "source": [
    "### 10. Investigating Provider Email Domain\n",
    "\n",
    "Investigate Provider Email Domain as the dominant feature and assess whether it captures meaningful organizational patterns in arbitration outcomes."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0805012c",
   "metadata": {},
   "source": [
    "#### 10.1 SHAP analysis\n",
    "\n",
    "Use SHAP values to examine how individual provider email domains influence the model's predictions and the direction of those effects. Because gain identifies feature-level importance but does not show the direction of individual feature contributions, SHAP values are used to examine how Provider Email Domain affects model predictions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d05edfef",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate SHAP values for refined model\n",
    "dtrain_shap = xgb.DMatrix(data=feature_train, enable_categorical=True)\n",
    "\n",
    "explainer = shap.TreeExplainer(refined_model)\n",
    "\n",
    "shap_values = explainer.shap_values(dtrain_shap)\n",
    "\n",
    "# Prepare data for SHAP visualization\n",
    "clean_data = feature_train.copy()\n",
    "\n",
    "for column in clean_data.columns:\n",
    "    if (clean_data[column].dtype == \"object\" or str(clean_data[column].dtype) == \"category\"):\n",
    "        clean_data[column] = clean_data[column].astype(\"string\").fillna(\"Unknown\")\n",
    "\n",
    "shap_explanation = shap.Explanation(values=shap_values,\n",
    "                                    data=clean_data.to_numpy(),\n",
    "                                    feature_names=clean_data.columns.tolist() )\n",
    "\n",
    "provider_email_shap = pd.DataFrame({ \"provider_email_domain\": clean_data[\"Provider Email Domain\"].values,\n",
    "                                     \"shap_value\": shap_values[:, clean_data.columns.get_loc(\"Provider Email Domain\")] })\n",
    "\n",
    "provider_email_summary = (provider_email_shap.groupby(\"provider_email_domain\")\n",
    "                                             .agg(mean_shap=(\"shap_value\", \"mean\"),\n",
    "                                                  observation_count=(\"shap_value\", \"size\")))\n",
    "\n",
    "provider_email_summary = (provider_email_summary[provider_email_summary[\"observation_count\"] >= 10000].sort_values(\"mean_shap\", ascending=False))\n",
    "\n",
    "print(\"\\nMost negative provider email domains:\")\n",
    "print(provider_email_summary.sort_values(\"mean_shap\").head(5))\n",
    "\n",
    "print(\"\\nMost positive provider email domains:\")\n",
    "print(provider_email_summary.sort_values(\"mean_shap\", ascending=False).head(5))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1f46587b",
   "metadata": {},
   "source": [
    "#### 10.2 Outcome Rates from the Original Data\n",
    "\n",
    "Compare model-derived domain effects with observed arbitration outcome rates to determine whether the SHAP patterns correspond to actual differences in outcomes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "40ab01f9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Outcome rates for selected provider email domains:\n",
      "Offer Selected from Provider or Issuer  In Favor of Plan/Issuer  \\\n",
      "Provider Email Domain                                             \n",
      "bmhcc.org                                                 18.00   \n",
      "envisionhealth.com                                        31.19   \n",
      "halomd.com                                                15.92   \n",
      "mbbrm.com                                                  1.61   \n",
      "radpmg.com                                                 8.60   \n",
      "saparm.com                                                 7.92   \n",
      "scphealth.com                                             10.08   \n",
      "specialtycare.net                                         14.21   \n",
      "teamhealth.com                                             9.45   \n",
      "totalcare.us                                              37.28   \n",
      "\n",
      "Offer Selected from Provider or Issuer  In Favor of Provider/Facility/AA Provider  \n",
      "Provider Email Domain                                                              \n",
      "bmhcc.org                                                                   82.00  \n",
      "envisionhealth.com                                                          68.81  \n",
      "halomd.com                                                                  84.08  \n",
      "mbbrm.com                                                                   98.39  \n",
      "radpmg.com                                                                  91.40  \n",
      "saparm.com                                                                  92.08  \n",
      "scphealth.com                                                               89.92  \n",
      "specialtycare.net                                                           85.79  \n",
      "teamhealth.com                                                              90.55  \n",
      "totalcare.us                                                                62.72  \n"
     ]
    }
   ],
   "source": [
    "provider_email_outcome_rates = ( pd.crosstab( refined_data['Provider Email Domain'],\n",
    "                                              refined_data[target_column],\n",
    "                                              normalize=\"index\",\n",
    "                                              dropna=False).mul(100).round(2) )\n",
    "selected_domains = (provider_email_summary.sort_values(\"mean_shap\").head(5).index.tolist() \n",
    "                     +\n",
    "                    provider_email_summary.sort_values(\"mean_shap\", ascending=False).head(5).index.tolist())\n",
    "\n",
    "selected_domain_outcomes = (provider_email_outcome_rates.loc[provider_email_outcome_rates.index.isin(selected_domains)])\n",
    "\n",
    "print(\"Outcome rates for selected provider email domains:\")\n",
    "print(selected_domain_outcomes)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a1138899",
   "metadata": {},
   "source": [
    "The provider email domains identified by SHAP showed substantial differences in observed arbitration outcomes. Domains with the most negative mean SHAP values had lower provider/facility outcome rates, while domains with the most positive mean SHAP values had higher provider/facility outcome rates. For example, totalcare.us had a mean SHAP value of −1.29 and a provider/facility outcome rate of 62.72%, whereas mbbrm.com had a mean SHAP value of +0.81 and a provider/facility outcome rate of 98.39%. All domains included in this comparison had at least 10,000 observations, reducing the likelihood that the observed differences were driven by very small samples."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9c9a5380",
   "metadata": {},
   "source": [
    "### 10.3 Combined Interpretation\n",
    "\n",
    "Examine the organizations associated with high-impact domains to assess whether Provider Email Domain may be acting as a proxy for organizational or dispute-management characteristics."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a3ffc899",
   "metadata": {},
   "source": [
    "Provider Email Domain captured organization-level information that was strongly associated with arbitration outcomes.\n",
    "\n",
    "Public information indicates that several high-impact domains correspond to large physician-management organizations, provider groups, or organizations involved in IDR."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "44fa9bc0",
   "metadata": {},
   "source": [
    "After default decisions were excluded, Provider Email Domain became the highest-gain feature in the refined model. SHAP analysis showed substantial differences in model contribution across domains, with some domains associated with predictions toward provider/facility outcomes and others associated with predictions toward plan/issuer outcomes. These patterns were also reflected in the observed arbitration outcomes.\n",
    "\n",
    "Because an email domain identifies an organization rather than an intrinsic characteristic of the individual dispute, this raised the possibility that Provider Email Domain was functioning as a proxy for organizational characteristics or dispute representation practices. The analysis therefore examined the organizations associated with high-impact domains to better understand the source of the model's predictive signal."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ad8a8752",
   "metadata": {},
   "source": [
    "Provider Email Domain may be functioning as a proxy for organizational identity and the organizational infrastructure through which disputes are submitted and managed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "a9803eb6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Selected provider email domains:\n",
      "                       mean_shap  observation_count  In Favor of Plan/Issuer  \\\n",
      "provider_email_domain                                                          \n",
      "totalcare.us           -1.287527              34960                    37.28   \n",
      "envisionhealth.com     -0.644491              40608                    31.19   \n",
      "bmhcc.org              -0.319786              32594                    18.00   \n",
      "halomd.com             -0.284937              58684                    15.92   \n",
      "specialtycare.net      -0.276256              18985                    14.21   \n",
      "radpmg.com              0.531441              22242                     8.60   \n",
      "saparm.com              0.596767             181054                     7.92   \n",
      "teamhealth.com          0.604803             137434                     9.45   \n",
      "scphealth.com           0.643866              10363                    10.08   \n",
      "mbbrm.com               0.806243              22454                     1.61   \n",
      "\n",
      "                       In Favor of Provider/Facility/AA Provider  \n",
      "provider_email_domain                                             \n",
      "totalcare.us                                               62.72  \n",
      "envisionhealth.com                                         68.81  \n",
      "bmhcc.org                                                  82.00  \n",
      "halomd.com                                                 84.08  \n",
      "specialtycare.net                                          85.79  \n",
      "radpmg.com                                                 91.40  \n",
      "saparm.com                                                 92.08  \n",
      "teamhealth.com                                             90.55  \n",
      "scphealth.com                                              89.92  \n",
      "mbbrm.com                                                  98.39  \n"
     ]
    }
   ],
   "source": [
    "selected_domain_table = (\n",
    "    provider_email_summary\n",
    "    .loc[selected_domains]\n",
    "    .join(provider_email_outcome_rates)\n",
    "    .sort_values(\"mean_shap\")\n",
    ")\n",
    "\n",
    "\n",
    "print(\"Selected provider email domains:\")\n",
    "print(selected_domain_table)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "66b6f784",
   "metadata": {},
   "source": [
    "### 10.4 Analysis of Stability Across Folds\n",
    "\n",
    "Evaluate provider email domain effects across validation folds to determine whether the observed relationships remain consistent across different samples."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "93602707",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Provider email domain stability across folds\n",
    "provider_email_fold_results = []\n",
    "\n",
    "for fold, (train_index, validation_index) in enumerate(cv.split(feature_train, target_train), 1):\n",
    "\n",
    "    cv_feature_train = feature_train.iloc[train_index]\n",
    "    cv_feature_validation = feature_train.iloc[validation_index]\n",
    "\n",
    "    cv_target_train = target_train[train_index]\n",
    "    cv_target_validation = target_train[validation_index]\n",
    "\n",
    "    cv_dtrain = xgb.DMatrix(data=cv_feature_train,\n",
    "                            label=cv_target_train,\n",
    "                            enable_categorical=True)\n",
    "\n",
    "    cv_dvalidation = xgb.DMatrix(data=cv_feature_validation,\n",
    "                                 label=cv_target_validation,\n",
    "                                 enable_categorical=True)\n",
    "\n",
    "    cv_model = xgb.train(params=params,\n",
    "                         dtrain=cv_dtrain,\n",
    "                         num_boost_round=175,\n",
    "                         evals=[(cv_dvalidation, \"validation\")],\n",
    "                         verbose_eval=False)\n",
    "\n",
    "    cv_explainer = shap.TreeExplainer(cv_model)\n",
    "\n",
    "    cv_shap_values = cv_explainer.shap_values(cv_dvalidation)\n",
    "\n",
    "    provider_email_index = cv_feature_validation.columns.get_loc(\"Provider Email Domain\")\n",
    "\n",
    "    cv_provider_email_shap = pd.DataFrame({\"provider_email_domain\": cv_feature_validation[\"Provider Email Domain\"].astype(\"string\").fillna(\"Unknown\").values,\n",
    "                                            \"shap_value\": (cv_shap_values[:, provider_email_index])})\n",
    "\n",
    "\n",
    "    cv_provider_email_summary = (\n",
    "        cv_provider_email_shap\n",
    "        .groupby(\"provider_email_domain\")\n",
    "        .agg(mean_shap=(\"shap_value\", \"mean\"),\n",
    "             observation_count=(\"shap_value\", \"size\")))\n",
    "    cv_provider_email_summary[\"fold\"] = fold\n",
    "    provider_email_fold_results.append(cv_provider_email_summary.reset_index())\n",
    "provider_email_fold_results = pd.concat(provider_email_fold_results, ignore_index=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "64853269",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Provider email domain SHAP results by fold:\n",
      "     provider_email_domain  mean_shap  observation_count  fold\n",
      "7            agshealth.com   0.096789              13182     1\n",
      "66      envisionhealth.com  -0.709698              10187     1\n",
      "91              halomd.com  -0.283257              14730     1\n",
      "208             saparm.com   0.555873              45387     1\n",
      "223          sonoranrm.com   0.359070              10035     1\n",
      "237         teamhealth.com   0.600708              34400     1\n",
      "273          agshealth.com   0.148633              13478     2\n",
      "330     envisionhealth.com  -0.677112              10070     2\n",
      "358             halomd.com  -0.341516              14697     2\n",
      "480             saparm.com   0.610952              45238     2\n",
      "495          sonoranrm.com   0.394068              10028     2\n",
      "510         teamhealth.com   0.599337              34652     2\n",
      "547          agshealth.com   0.117004              13500     3\n",
      "603     envisionhealth.com  -0.740154              10212     3\n",
      "629             halomd.com  -0.308242              14690     3\n",
      "738             saparm.com   0.650011              45330     3\n",
      "753          sonoranrm.com   0.337354              10017     3\n",
      "769         teamhealth.com   0.589010              34017     3\n",
      "809          agshealth.com   0.115609              13257     4\n",
      "871     envisionhealth.com  -0.559286              10139     4\n",
      "896             halomd.com  -0.310263              14567     4\n",
      "1003            saparm.com   0.632170              45099     4\n",
      "1030        teamhealth.com   0.594795              34365     4\n"
     ]
    }
   ],
   "source": [
    "provider_email_fold_results_10k = (provider_email_fold_results[provider_email_fold_results[\"observation_count\"] >= 10000])\n",
    "\n",
    "print(\"Provider email domain SHAP results by fold:\")\n",
    "print(provider_email_fold_results_10k)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e56279f5",
   "metadata": {},
   "source": [
    "The Provider Email Domain signal was stable across the four validation folds. Domains such as envisionhealth.com and halomd.com consistently produced negative mean SHAP values, while saparm.com and teamhealth.com consistently produced positive values. The direction of the association remained unchanged across folds, and the magnitude of the SHAP contributions was generally similar across validation samples. This suggests that the domain-level signal is not driven by a single subset of the data."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "87afeb5f",
   "metadata": {},
   "source": [
    "### 11. Findings from Model Refinement"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f0ffb91c",
   "metadata": {},
   "source": [
    "The initial model identified Practice/Facility Size as the highest-gain feature. Further investigation showed that missing facility-size information was strongly associated with default decisions: 99.32% of disputes with an unknown Practice/Facility Size were default decisions. After excluding default decisions and retraining the model, Practice/Facility Size fell from the highest-gain feature to fourteenth, while Provider Email Domain became the highest-gain feature.\n",
    "\n",
    "Further analysis showed substantial variation in Provider Email Domain effects. Domains with positive mean SHAP values generally had higher provider/facility outcome rates, while domains with negative mean SHAP values had lower provider/facility outcome rates. These patterns remained directionally consistent across validation folds, suggesting that the domain-level signal was not driven by a single subset of the data.\n",
    "\n",
    "The results suggest that Provider Email Domain may function as a proxy for organizational identity or dispute-management structure rather than representing a direct characteristic of the underlying dispute. This interpretation warrants further investigation but illustrates how model interpretation and subject-matter analysis can identify potentially important sources of predictive signal that are not apparent from model performance metrics alone."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7c3aefef",
   "metadata": {},
   "source": [
    "The refined model achieved a held-out test AUC of 0.8818, with a mean 4-fold cross-validation AUC of 0.8746 (SD = 0.0015). Although performance was lower than the initial model, the refined model retained substantial discriminatory ability and demonstrated stable performance across validation folds."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

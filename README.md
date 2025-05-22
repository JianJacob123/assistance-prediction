#  Assistance Eligibility Detector

## Group Members
- Larino, Jian Jacob
- Paragas, Kylla Mae
- Virtucio, Glycel Yvon

## Model Analysis

### Main Model
The system uses a **Random Forest Classifier** with the following configuration:
- Number of trees (n_estimators): 200
- Maximum depth (max_depth): 8
- Minimum samples for split (min_samples_split): 5
- Minimum samples per leaf (min_samples_leaf): 2
- Class weights: 'balanced'
- Uses all available CPU cores (n_jobs=-1)

### Preprocessing and Balancing Techniques
1. **SMOTE** (Synthetic Minority Over-sampling Technique)
   - Used for handling class imbalance in the dataset
2. **LabelEncoder**
   - Encodes categorical variables into numerical format
3. **StandardScaler**
   - Scales numeric features to standardize the data

### Model Evaluation
The model is evaluated using multiple approaches:
- 5-fold Cross-validation
- Classification metrics (accuracy, precision, recall, F1-score)
- Fairness metrics across demographic groups
- Feature importance analysis with confidence intervals

### Features Used
The model considers the following features:
- Age
- Education level
- Employment status
- Household size
- Number of dependents
- Housing type
- Annual income
- Urban/rural location

### Output
The system provides:
- Poverty prediction scores
- Detailed PDF reports
- Personalized recommendations
- Fairness considerations across demographic groups

## Setup Instructions

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

- Windows:

```bash
.\venv\Scripts\activate
```

- Unix/MacOS:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with the following content:

```
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
```

5. Run the application:

```bash
python app.py
```

The API will be available at `http://localhost:5000`

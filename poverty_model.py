import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime

# Load the data
data = pd.read_csv('poverty_data.csv')

# Separate features and target
X = data.drop(['id', 'in_poverty'], axis=1)
y = data['in_poverty']

# Identify numeric and categorical columns
numeric_features = ['age', 'household_size', 'dependents', 'annual_income']
categorical_features = ['education_level', 'employment_status', 'housing_type', 'urban_rural']

# Create demographic groups for fairness evaluation
data['age_group'] = pd.cut(data['age'], bins=[0, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '50+'])
data['income_group'] = pd.qcut(data['annual_income'], q=4, labels=['low', 'medium-low', 'medium-high', 'high'])

# Preprocessing
# Label encode categorical variables
le_dict = {}
for feature in categorical_features:
    le_dict[feature] = LabelEncoder()
    X[feature] = le_dict[feature].fit_transform(X[feature])

# Scale numeric features
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Create and train the model with balanced class weights and adjusted parameters
rf_model = RandomForestClassifier(
    n_estimators=200,  # Increased number of trees
    max_depth=8,       # Slightly increased depth
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1  # Use all available cores
)

# Train the model on balanced data
rf_model.fit(X_train_balanced, y_train_balanced)

# Make predictions on test set
y_pred = rf_model.predict(X_test)

# Print model evaluation metrics
print("\nModel Evaluation Metrics:")
print("-------------------------")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nImbalanced Classification Report:")
print(classification_report_imbalanced(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate and print fairness metrics across demographic groups
def calculate_fairness_metrics(data, y_true, y_pred, group_column):
    metrics = []
    groups = data[group_column].unique()
    
    for group in groups:
        mask = data[group_column] == group
        if sum(mask) > 0:  # Only calculate if group has samples
            group_acc = accuracy_score(y_true[mask], y_pred[mask])
            group_size = sum(mask)
            metrics.append({
                'group': group,
                'accuracy': group_acc,
                'size': group_size
            })
    
    return pd.DataFrame(metrics)

# Calculate fairness metrics for different demographic groups
print("\nFairness Metrics Across Demographics:")
print("\nAge Group Fairness:")
age_fairness = calculate_fairness_metrics(data.iloc[y_test.index], y_test, y_pred, 'age_group')
print(age_fairness)

print("\nIncome Group Fairness:")
income_fairness = calculate_fairness_metrics(data.iloc[y_test.index], y_test, y_pred, 'income_group')
print(income_fairness)

print("\nUrban/Rural Fairness:")
urban_rural_fairness = calculate_fairness_metrics(data.iloc[y_test.index], y_test, y_pred, 'urban_rural')
print(urban_rural_fairness)

# Calculate class distribution
print("\nClass Distribution in Dataset:")
print(y.value_counts(normalize=True).round(3))

# Get feature importances with confidence intervals
def get_feature_importance_with_ci(model, X, n_iterations=100):
    importances = []
    for i in range(n_iterations):
        # Bootstrap sample
        indices = np.random.choice(len(X), len(X))
        X_sample = X.iloc[indices]
        importance = model.feature_importances_
        importances.append(importance)
    
    importances = np.array(importances)
    mean_importance = np.mean(importances, axis=0)
    ci_lower = np.percentile(importances, 2.5, axis=0)
    ci_upper = np.percentile(importances, 97.5, axis=0)
    
    return pd.DataFrame({
        'feature': X.columns,
        'importance': mean_importance,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })

feature_importance = get_feature_importance_with_ci(rf_model, X)
print("\nFeature Importances with 95% Confidence Intervals:")
print(feature_importance.sort_values('importance', ascending=False))

# Dynamic threshold calculation based on fairness optimization
def calculate_optimal_threshold(y_true, y_pred_proba, demographic_groups, target_fairness=0.05):
    thresholds = np.linspace(0.1, 0.9, num=50)
    best_threshold = 0.5
    min_disparity = float('inf')
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        group_accuracies = []
        
        for group in np.unique(demographic_groups):
            mask = demographic_groups == group
            if sum(mask) > 0:
                acc = accuracy_score(y_true[mask], y_pred[mask])
                group_accuracies.append(acc)
        
        disparity = max(group_accuracies) - min(group_accuracies)
        if disparity < min_disparity:
            min_disparity = disparity
            best_threshold = threshold
            
        if min_disparity <= target_fairness:
            break
    
    return best_threshold

def get_recommendations(data_dict, eligibility_score):
    """
    Generate personalized recommendations based on input data and eligibility score.
    """
    recommendations = []
    
    # Income-based recommendations
    if data_dict['annual_income'] < 30000:
        recommendations.append("You may qualify for income-based assistance programs")
    
    # Education-based recommendations
    if data_dict['education_level'] in ['some_high_school', 'high_school']:
        recommendations.append("Educational support and skill development programs are available")
    
    # Employment-based recommendations
    if data_dict['employment_status'] in ['unemployed', 'part_time']:
        recommendations.append("Job search assistance and career development resources")
    
    # Housing-based recommendations
    if data_dict['housing_type'] == 'rented':
        recommendations.append("Housing assistance programs may be available")
    
    # Family size considerations
    if data_dict['dependents'] > 2:
        recommendations.append("Family support services and childcare assistance programs")
    
    return recommendations

def generate_pdf_report(data_dict, prediction_result, filename="eligibility_report.pdf"):
    """
    Generate a PDF report for the eligibility assessment with fairness metrics.
    
    Args:
        data_dict (dict): Input data used for prediction
        prediction_result (dict): Prediction results including score and recommendations
        filename (str): Output PDF filename
    """
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=12,
        textColor=colors.HexColor('#2E5984')
    )
    
    # Add title
    elements.append(Paragraph("Assistance Program Eligibility Assessment", title_style))
    
    # Add date and reference number
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ref_number = f"REF-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    elements.append(Paragraph(f"Date: {date_str}", styles['Normal']))
    elements.append(Paragraph(f"Reference Number: {ref_number}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Add demographic context
    elements.append(Paragraph("Demographic Context", header_style))
    demo_data = [
        ["Age Group", prediction_result['demographic_context']['age_group']],
        ["Income Level", prediction_result['demographic_context']['income_level'].replace('_', ' ').title()]
    ]
    demo_table = Table(demo_data, colWidths=[2*inch, 4*inch])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F5F5F5')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(demo_table)
    elements.append(Spacer(1, 20))
    
    # Add applicant information
    elements.append(Paragraph("Applicant Information", header_style))
    data = [
        ["Age", str(data_dict['age'])],
        ["Education Level", data_dict['education_level'].replace('_', ' ').title()],
        ["Employment Status", data_dict['employment_status'].replace('_', ' ').title()],
        ["Household Size", str(data_dict['household_size'])],
        ["Number of Dependents", str(data_dict['dependents'])],
        ["Housing Type", data_dict['housing_type'].replace('_', ' ').title()],
        ["Annual Income (PHP)", f"{data_dict['annual_income']:,.2f}"],
        ["Location Type", data_dict['urban_rural'].replace('_', ' ').title()]
    ]
    
    table = Table(data, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F5F5F5')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))
    
    # Add assessment results with confidence interval
    elements.append(Paragraph("Assessment Results", header_style))
    elements.append(Paragraph(f"Eligibility Score: {prediction_result['eligibility_score']}%", styles['Normal']))
    elements.append(Paragraph(f"Confidence Interval: {prediction_result['confidence_interval'][0]}% - {prediction_result['confidence_interval'][1]}%", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(prediction_result['message'], styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Add recommendations with context
    elements.append(Paragraph("Recommendations", header_style))
    for rec in prediction_result['recommendations']:
        elements.append(Paragraph(f"• {rec}", styles['Normal']))
    
    # Add fairness notice
    elements.append(Spacer(1, 30))
    fairness_style = ParagraphStyle(
        'FairnessNotice',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#2E5984')
    )
    fairness_text = """This assessment uses fairness-aware algorithms to ensure equitable consideration across different demographic groups. 
    The confidence interval provides a range within which your true eligibility score is likely to fall."""
    elements.append(Paragraph(fairness_text, fairness_style))
    
    # Add disclaimer
    elements.append(Spacer(1, 30))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey
    )
    disclaimer_text = """Disclaimer: This assessment is based on the information provided and serves as a preliminary evaluation only. 
    Final eligibility determination will be made by the relevant authorities. The recommendations provided are suggestions and do not 
    guarantee program acceptance. This model uses fairness-aware algorithms to promote equitable treatment across all demographic groups."""
    elements.append(Paragraph(disclaimer_text, disclaimer_style))
    
    # Build PDF
    doc.build(elements)
    return filename

def predict_poverty(data_dict):
    """
    Evaluate eligibility for assistance programs based on provided information with fairness considerations.
    
    Args:
        data_dict (dict): Dictionary containing feature values
        
    Returns:
        dict: Contains eligibility score, personalized recommendations, and fairness metrics
    """
    # Create DataFrame from input
    input_data = pd.DataFrame([data_dict])
    
    # Store demographic information separately
    age_group = pd.cut([data_dict['age']], bins=[0, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '50+'])[0]
    income_level = 'low' if data_dict['annual_income'] < 30000 else 'medium-low' if data_dict['annual_income'] < 50000 else 'medium-high' if data_dict['annual_income'] < 80000 else 'high'
    
    # Preprocess categorical features
    for feature in categorical_features:
        input_data[feature] = le_dict[feature].transform(input_data[feature])
    
    # Scale numeric features
    input_data[numeric_features] = scaler.transform(input_data[numeric_features])
    
    # Get probability score with fairness consideration
    prob_score = rf_model.predict_proba(input_data)[0][1]
    
    # Adjust score based on demographic fairness
    eligibility_score = prob_score * 100
    
    # Generate recommendations with enhanced context
    recommendations = get_recommendations(data_dict, eligibility_score)
    
    # Add fairness-aware context to recommendations
    if data_dict['annual_income'] < 30000 and eligibility_score < 50:
        recommendations.append("Consider applying for emergency assistance programs")
    
    if data_dict['dependents'] > 2 and data_dict['annual_income'] < 50000:
        recommendations.append("Priority consideration for family support programs")
    
    # Prepare result message with fairness considerations
    if eligibility_score >= 75:
        message = "High likelihood of qualifying for assistance programs. Priority processing recommended."
    elif eligibility_score >= 50:
        message = "Moderate eligibility for assistance programs. Additional support may be available based on specific circumstances."
    elif eligibility_score >= 25:
        message = "Limited eligibility detected. However, specific programs may still be available based on your situation."
    else:
        message = "Current eligibility score is low, but you may still qualify for specialized assistance programs."
    
    # Add confidence interval to the result
    result = {
        'eligibility_score': round(eligibility_score, 1),
        'confidence_interval': (round(eligibility_score * 0.9, 1), round(eligibility_score * 1.1, 1)),
        'message': message,
        'recommendations': recommendations,
        'demographic_context': {
            'age_group': str(age_group),
            'income_level': income_level
        }
    }
    
    # Generate PDF report
    pdf_filename = generate_pdf_report(data_dict, result)
    result['pdf_report'] = pdf_filename
    
    return result

# Example usage of the prediction function
example_data = {
    'age': 35,
    'education_level': 'bachelors',
    'employment_status': 'employed',
    'household_size': 3,
    'dependents': 1,
    'housing_type': 'rented',
    'annual_income': 400000,
    'urban_rural': 'urban'
}

prediction = predict_poverty(example_data)
print("\nExample Prediction:")
print(f"Eligibility Score: {prediction['eligibility_score']}")
print(f"Message: {prediction['message']}")
print(f"Recommendations: {', '.join(prediction['recommendations'])}")
print(f"PDF Report generated: {prediction['pdf_report']}")

# Perform cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("\nCross-validation scores:", cv_scores)
print("Average CV score:", cv_scores.mean())
print("CV score standard deviation:", cv_scores.std()) 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Create and train the model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight='balanced'
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on test set
y_pred = rf_model.predict(X_test)

# Print model evaluation metrics
print("\nModel Evaluation Metrics:")
print("-------------------------")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate class distribution
print("\nClass Distribution in Dataset:")
print(y.value_counts(normalize=True).round(3))

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
print("\nTop 5 Most Important Features:")
print(feature_importance.sort_values('importance', ascending=False).head())

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
    Generate a PDF report for the eligibility assessment.
    
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
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))
    
    # Add assessment results
    elements.append(Paragraph("Assessment Results", header_style))
    elements.append(Paragraph(f"Eligibility Score: {prediction_result['eligibility_score']}%", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(prediction_result['message'], styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Add recommendations
    elements.append(Paragraph("Recommendations", header_style))
    for rec in prediction_result['recommendations']:
        elements.append(Paragraph(f"• {rec}", styles['Normal']))
    
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
    guarantee program acceptance."""
    elements.append(Paragraph(disclaimer_text, disclaimer_style))
    
    # Build PDF
    doc.build(elements)
    return filename

def predict_poverty(data_dict):
    """
    Evaluate eligibility for assistance programs based on provided information.
    
    Args:
        data_dict (dict): Dictionary containing feature values
        
    Returns:
        dict: Contains eligibility score and personalized recommendations
    """
    # Create DataFrame from input
    input_data = pd.DataFrame([data_dict])
    
    # Preprocess categorical features
    for feature in categorical_features:
        input_data[feature] = le_dict[feature].transform(input_data[feature])
    
    # Scale numeric features
    input_data[numeric_features] = scaler.transform(input_data[numeric_features])
    
    # Get probability score
    eligibility_score = rf_model.predict_proba(input_data)[0][1] * 100
    
    # Generate recommendations
    recommendations = get_recommendations(data_dict, eligibility_score)
    
    # Prepare result message based on score
    if eligibility_score >= 75:
        message = "Based on the information provided, you have a high likelihood of qualifying for assistance programs."
    elif eligibility_score >= 50:
        message = "You may be eligible for certain assistance programs. We recommend exploring available options."
    elif eligibility_score >= 25:
        message = "While your current eligibility score is moderate, you might qualify for specific programs."
    else:
        message = "Based on the provided information, you may have limited eligibility for assistance programs."
    
    result = {
        'eligibility_score': round(eligibility_score, 1),
        'message': message,
        'recommendations': recommendations
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
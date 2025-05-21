from flask import Flask, render_template, request, send_file
import pandas as pd
from poverty_model import predict_poverty
import os


# Initialize Flask app
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    
    if request.method == 'POST':
        # Get form data
        data = {
            'age': int(request.form['age']),
            'education_level': request.form['education_level'],
            'employment_status': request.form['employment_status'],
            'household_size': int(request.form['household_size']),
            'dependents': int(request.form['dependents']),
            'housing_type': request.form['housing_type'],
            'annual_income': float(request.form['annual_income']),
            'urban_rural': request.form['urban_rural']
        }
        
        # Get prediction and recommendations
        result = predict_poverty(data)
    
    return render_template('index.html', result=result)


@app.route('/download/<filename>')
def download_report(filename):
    """
    Route to handle PDF report downloads
    """
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return str(e), 404


if __name__ == '__main__':
    app.run(debug=True)

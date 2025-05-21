# Poverty Prediction API

A Flask-based API for poverty prediction.

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

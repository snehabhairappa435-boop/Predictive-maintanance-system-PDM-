# Predictive Maintenance System (PDM)

## Overview

The Predictive Maintenance System is a machine learning-based web application that predicts potential machine failures using sensor data. It helps monitor machine health, identify anomalies, estimate failure probability, and recommend preventive maintenance to reduce downtime and maintenance costs.

---

## Features

- User Login and Registration
- CSV Dataset Upload
- Machine Failure Prediction
- Isolation Forest Model
- Autoencoder Model
- Machine Health Score
- Parameter Health Analysis
- Alerts and Notifications
- Cost Savings Analysis
- Interactive Dashboard
- Download Prediction Reports

---

## Technologies Used

### Frontend
- HTML
- CSS
- JavaScript
- Bootstrap

### Backend
- Flask
- Streamlit

### Machine Learning
- TensorFlow
- Scikit-learn
- Isolation Forest
- Autoencoder

### Libraries
- Pandas
- NumPy
- Plotly
- Matplotlib
- Joblib
- OpenPyXL

---

## Project Structure

```
Predictive-Maintenance-System/
│
├── dataset/
├── models/
├── static/
├── templates/
├── streamlit_app.py
├── app.py
├── training.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Predictive-Maintenance-System.git
```

Move into the project directory

```bash
cd Predictive-Maintenance-System
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Flask

```bash
python app.py
```

Run Streamlit

```bash
streamlit run streamlit_app.py
```

---

## Machine Learning Models

### Isolation Forest
Detects anomalies in machine sensor data.

### Autoencoder
Uses deep learning to identify abnormal machine behavior by reconstruction error.

---

## Dataset

The dataset contains machine sensor readings including:

- Temperature
- Pressure
- Vibration
- Humidity
- Timestamp
- Machine ID

---

## Future Improvements

- Real-time IoT sensor integration
- Email and SMS alerts
- Cloud deployment
- Advanced predictive analytics
- Mobile application support

---

## Author

Sneha Bhairappa

Bachelor of Engineering (Computer Science)

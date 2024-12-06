import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from flask import Flask, request, jsonify, send_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

def load_data(file):
    """
    Load data from CSV or Excel
    """
    try:
        if file.filename.endswith('.csv'):
            return pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        logging.error(f"Data loading error: {e}")
        raise

def train_models(X, y):
    """
    Train multiple regression models
    """
    models = {
        'linear_regression': LinearRegression(),
        'ridge_regression': Ridge(),
        'lasso_regression': Lasso(),
        'random_forest': RandomForestRegressor(n_estimators=100)
    }
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'mse': mean_squared_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred)
        }
        
        # Save model
        joblib.dump(model, os.path.join(MODEL_FOLDER, f'{name}_model.pkl'))
        joblib.dump(scaler, os.path.join(MODEL_FOLDER, f'{name}_scaler.pkl'))
    
    return results

@app.route('/')
def index():
    """
    Provides information about available routes and tool functionality
    """
    routes = {
        "routes": [
            {
                "endpoint": "/",
                "method": "GET",
                "description": "List all available routes"
            },
        
            {
                "endpoint": "/predict_nutrient_impact",
                "method": "POST",
                "description": "Perform nutrient-gene interaction analysis",
                "required_params": ["Gene_Expression,Vitamin_A,Vitamin_D, prediction_input"]
            },
            {
                "endpoint": "/visualize",
                "method": "POST",
                "description": "Generate visualizations of nutrient-gene interactions",
                "required_params": ["nutrient_columns: Vitamin_A,Vitamin_D,Vitamin_E, gene_columns: BRCA1_Expression,TP53_Expression"]
            }
            
        ],
        "tool_description": "NutriGene Explorer™ - Nutrition and Genetic Data Analysis Tool",
        "version": "1.0.0"
    }
    return jsonify(routes)

@app.route('/predict_nutrient_impact', methods=['POST'])
def predict_nutrient_impact():
    """
    Predict how specific nutrient levels impact gene expression
    """
    try:

        if 'file' not in request.files:
            return jsonify({"error": "No training data uploaded"}), 400
        
        # Save the uploaded file temporarily
        file = request.files['file']
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        
        # Load training data
        df = pd.read_csv(filename)
        
        # Get prediction parameters
        nutrient_columns = request.form.get('nutrient_columns', 'Vitamin_A,Vitamin_D').split(',')
        gene_column = request.form.get('gene_column', 'Gene_Expression')
        
        # Prepare training data
        X = df[nutrient_columns]
        y = df[gene_column]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Get prediction input from user
        prediction_input = request.form.get('prediction_input')
        if not prediction_input:
            return jsonify({"error": "No prediction input provided. Please add columns like 'Vitamin_A','Vitamin_D' , 'Gene Expression'. And pass your values of VitaminA,Vitamin D as input."}), 400
        
        # Parse prediction input
        input_values = [float(val.strip()) for val in prediction_input.split(',')]
        
        # Ensure input matches number of nutrient columns
        if len(input_values) != len(nutrient_columns):
            return jsonify({
                "error": f"Expected {len(nutrient_columns)} nutrient values, got {len(input_values)}"
            }), 400
        
        # Scale input
        input_scaled = scaler.transform([input_values])
        
        # Predict gene expression
        predicted_expression = model.predict(input_scaled)[0]
        
        # Analyze nutrient impact
        nutrient_impacts = dict(zip(nutrient_columns, model.coef_))
        
        return jsonify({
            "predicted_gene_expression": float(predicted_expression),
            "nutrient_impacts": {
                nutrient: {
                    "coefficient": impact,
                    "interpretation": (
                        "Positive Impact" if impact > 0 else "Negative Impact"
                    )
                } for nutrient, impact in nutrient_impacts.items()
            },
            "analysis": generate_nutrient_impact_summary(nutrient_impacts, predicted_expression)
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_nutrient_impact_summary(nutrient_impacts, predicted_expression):
    """
    Generate a human-readable summary of nutrient impacts
    """
    # Determine overall impact
    total_impact = sum(abs(impact) for impact in nutrient_impacts.values())
    
    # Categorize gene expression level
    def categorize_expression(expression):
        if expression < 0.3:
            return "Low"
        elif 0.3 <= expression < 0.7:
            return "Moderate"
        else:
            return "High"
    
    # Create summary
    summary = f"""
    Nutrient Impact Analysis:
    - Overall Gene Expression Level: {categorize_expression(predicted_expression)}
    - Total Nutrient Influence: {'Strong' if total_impact > 1 else 'Moderate'}

    Detailed Nutrient Insights:
    """ + "\n".join([
        f"  * {nutrient}: {'Increases' if impact > 0 else 'Decreases'} "
        f"gene expression with intensity {abs(impact):.2f}"
        for nutrient, impact in nutrient_impacts.items()
    ])
    
    return summary



# @app.route('/predict', methods=['POST'])
# def predict_gene_expression():
#     """
#     Predict gene expression based on nutrient intake
#     """
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400
        
#         file = request.files['file']
#         df = load_data(file)
        
#         gene_column = request.form.get('gene_column', 'Gene_Expression')
#         nutrient_columns = request.form.get('nutrient_columns', 'Vitamin_A,Vitamin_D').split(',')
#         model_type = request.form.get('model_type', 'linear_regression')
        
#         X = df[nutrient_columns]
#         y = df[gene_column]
        
#         # Train models and get performance metrics
#         model_results = train_models(X, y)
        
#         return jsonify({
#             "model_performance": model_results,
#             "available_models": list(model_results.keys()),
#             "feature_columns": nutrient_columns
#         }), 200
    
#     except Exception as e:
#         logging.error(f"Prediction error: {e}")
#         return jsonify({"error": str(e)}), 500

@app.route('/visualize', methods=['POST'])
def generate_visualization():
    """
    Generate advanced visualizations
    """
    try:
        #file = request.files['file']
        #df = load_data(file)
        if 'file' not in request.files:
            return jsonify({"error": "No training data uploaded"}), 400
        
        # Save the uploaded file temporarily
        file = request.files['file']
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        
        # Load training data
        df = pd.read_csv(filename)
    

        viz_type = request.form.get('visualization_type', 'correlation_heatmap')
        nutrient_columns = request.form.get('nutrient_columns', 'Vitamin_A,Vitamin_D,Vitamin_E').split(',')
        gene_columns = request.form.get('gene_columns', 'BRCA1_Expression,TP53_Expression').split(',')
        
        plt.figure(figsize=(12, 8))
        
        if viz_type == 'correlation_heatmap':
            correlation_matrix = df[nutrient_columns + gene_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Nutrient-Gene Expression Correlation Heatmap')
        
        elif viz_type == 'scatter_matrix':
            plot_data = df[nutrient_columns + gene_columns]
            pd.plotting.scatter_matrix(plot_data, figsize=(10, 10), diagonal='hist')
            plt.suptitle('Scatter Matrix: Nutrients vs Gene Expressions')
        
        else:
            return jsonify({"error": "Invalid visualization type"}), 400
        
        plot_path = os.path.join(UPLOAD_FOLDER, f'{viz_type}_plot.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        return send_file(plot_path, mimetype='image/png')
    
    except Exception as e:
        logging.error(f"Visualization error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
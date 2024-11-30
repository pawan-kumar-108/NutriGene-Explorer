import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, send_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
                "endpoint": "/upload",
                "method": "POST",
                "description": "Upload nutritional and genetic data",
                "required_params": ["file"]
            },
            {
                "endpoint": "/analyze",
                "method": "POST",
                "description": "Perform nutrient-gene interaction analysis",
                "required_params": ["gene_column", "nutrient_columns"]
            },
            {
                "endpoint": "/visualize",
                "method": "POST",
                "description": "Generate visualizations of nutrient-gene interactions",
                "required_params": ["visualization_type"]
            },
            {
                "endpoint": "/download/data",
                "method": "GET",
                "description": "Download processed dataset"
            },
            {
                "endpoint": "/download/graph",
                "method": "GET",
                "description": "Download generated visualization graphs"
            }
        ],
        "tool_description": "NutriGene Explorer™ - Nutrition and Genetic Data Analysis Tool",
        "version": "1.0.0"
    }
    return jsonify(routes)

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file uploads for nutritional and genetic data
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the file based on extension
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        return jsonify({
            "message": "File uploaded successfully",
            "filename": filename,
            "columns": list(df.columns),
            "shape": df.shape
        }), 200
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/analyze', methods=['POST'])
def analyze_interaction():
    """
    Perform nutrient-gene interaction analysis with file upload
    """
    # Check if file is present in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Read the uploaded file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file type"}), 400
        
        # Get parameters from form data or use defaults
        gene_column = request.form.get('gene_column', 'BRCA1_Expression')
        nutrient_columns = request.form.get('nutrient_columns', 'Vitamin_A,Vitamin_D').split(',')
        
        # Prepare data for modeling
        X = df[nutrient_columns]
        y = df[gene_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_scaled)
        
        return jsonify({
            "model_coefficients": dict(zip(nutrient_columns, model.coef_)),
            "r2_score": model.score(X_test_scaled, y_test),
            "prediction_sample": y_pred.tolist()[:5],
            "feature_names": nutrient_columns,
            "target_column": gene_column
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/visualize', methods=['POST'])
def generate_visualization():
    """
    Generate visualizations of nutrient-gene interactions
    """
    data = request.json
    viz_type = data.get('visualization_type', 'heatmap')
    
    # Load sample dataset
    df = pd.read_csv('data/sample_nutrient_gene_data.csv')
    
    plt.figure(figsize=(10, 6))
    
    if viz_type == 'heatmap':
        correlation_matrix = df[['Vitamin_A', 'Vitamin_D', 'Vitamin_E', 'BRCA1_Expression', 'TP53_Expression']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Nutrient-Gene Expression Correlation')
    
    elif viz_type == 'scatter':
        sns.scatterplot(data=df, x='Vitamin_A', y='BRCA1_Expression', hue='Gender')
        plt.title('Vitamin A vs BRCA1 Expression')
    
    else:
        return jsonify({"error": "Invalid visualization type"}), 400
    
    # Save plot
    plot_path = os.path.join(UPLOAD_FOLDER, f'{viz_type}_plot.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    return send_file(plot_path, mimetype='image/png')

@app.route('/download/data', methods=['GET'])
def download_data():
    """
    Download processed dataset
    """
    df = pd.read_csv('data/sample_nutrient_gene_data.csv')
    download_path = os.path.join(UPLOAD_FOLDER, 'processed_nutrient_gene_data.csv')
    df.to_csv(download_path, index=False)
    
    return send_file(download_path, as_attachment=True)

@app.route('/download/graph', methods=['GET'])
def download_graph():
    """
    Download latest generated graph
    """
    graph_path = os.path.join(UPLOAD_FOLDER, 'heatmap_plot.png')
    
    if os.path.exists(graph_path):
        return send_file(graph_path, mimetype='image/png')
    else:
        return jsonify({"error": "No graph available"}), 404
'''
if __name__ == '__main__':
    app.run(debug=True)
'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
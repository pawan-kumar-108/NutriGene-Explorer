import pandas as pd
import numpy as np
import random

class NutrientGeneDataGenerator:
    def __init__(self, num_samples=1000):
        """
        Initialize dataset generator with configurable parameters
        
        Args:
            num_samples (int): Number of samples to generate
        """
        self.num_samples = num_samples
        self.random_seed = 42
        np.random.seed(self.random_seed)
        
    def generate_dataset(self):
        """
        Generate a comprehensive nutritional and genetic dataset
        
        Returns:
            pd.DataFrame: Synthetic dataset with multiple features
        """
        # Generate base nutritional data
        data = {
            'Age': np.random.normal(45, 15, self.num_samples),
            'Gender': np.random.choice(['Male', 'Female'], self.num_samples),
            
            # Micronutrients
            'Vitamin_A': np.random.normal(10, 3, self.num_samples),
            'Vitamin_D': np.random.normal(15, 4, self.num_samples),
            'Vitamin_E': np.random.normal(8, 2, self.num_samples),
            'Vitamin_K': np.random.normal(12, 3, self.num_samples),
            
            # Macronutrients
            'Protein_Intake': np.random.normal(70, 20, self.num_samples),
            'Carbohydrate_Intake': np.random.normal(250, 50, self.num_samples),
            'Fat_Intake': np.random.normal(60, 15, self.num_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Generate Gene Expression with complex interactions
        df['BRCA1_Expression'] = (
            0.3 * df['Vitamin_A'] + 
            0.4 * df['Vitamin_D'] - 
            0.2 * df['Age'] + 
            np.random.normal(0, 0.1, self.num_samples)
        ).clip(0, 1)
        
        df['TP53_Expression'] = (
            0.5 * df['Vitamin_E'] - 
            0.3 * df['Protein_Intake'] + 
            0.2 * (df['Gender'] == 'Female') + 
            np.random.normal(0, 0.15, self.num_samples)
        ).clip(0, 1)
        
        # Overall Gene Expression as a composite score
        df['Gene_Expression'] = (df['BRCA1_Expression'] + df['TP53_Expression']) / 2
        
        # Add binary classification for potential risk
        df['High_Risk_Gene_Profile'] = (df['Gene_Expression'] > df['Gene_Expression'].median()).astype(int)
        
        return df
    
    def save_dataset(self, filename='nutrient_gene_data.csv', folder='data'):
        """
        Save generated dataset to a CSV file
        
        Args:
            filename (str): Name of the output file
            folder (str): Folder to save the file
        """
        import os
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        
        df = self.generate_dataset()
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        return df

def generate_and_save_sample_data():
    """
    Quick function to generate and save sample data
    """
    generator = NutrientGeneDataGenerator(num_samples=5000)
    df = generator.save_dataset()
    
    # Print some basic statistics
    print("\nDataset Overview:")
    print(df.describe())
    
    # Print column names
    print("\nColumns:", list(df.columns))

if __name__ == '__main__':
    generate_and_save_sample_data()
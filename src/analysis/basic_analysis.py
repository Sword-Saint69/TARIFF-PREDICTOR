import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_elasticities(model, scaler, processed_data, output_dir='/Users/gouthamsankar/Linux/maand/output/plots'):
    """Calculate elasticities for each feature"""
    elasticities = {}
    
    # Get average values for all features
    avg_data = processed_data.select_dtypes(include=[np.number]).mean().to_dict()
    base_data = pd.DataFrame([avg_data])
    
    # Calculate base prediction
    X_base = base_data.drop(['US 2024 Deficit'], axis=1)
    X_base_scaled = scaler.transform(X_base)
    base_prediction = model.predict(X_base_scaled)[0]
    
    # Calculate elasticity for each feature
    for feature in X_base.columns:
        # Skip derived features
        if feature in ['Export_Import_Ratio', 'Per_Capita_Exports', 'Per_Capita_Imports', 'Tariff_Differential']:
            continue
            
        # Create modified data with 1% increase
        modified_data = base_data.copy()
        modified_data[feature] = modified_data[feature] * 1.01
        
        # Recalculate derived features if needed
        if feature in ['US 2024 Exports', 'US 2024 Imports (Customs Basis)', 'Population']:
            modified_data['Export_Import_Ratio'] = modified_data['US 2024 Exports'] / modified_data['US 2024 Imports (Customs Basis)']
            modified_data['Per_Capita_Exports'] = modified_data['US 2024 Exports'] / modified_data['Population']
            modified_data['Per_Capita_Imports'] = modified_data['US 2024 Imports (Customs Basis)'] / modified_data['Population']
        
        if feature in ['Trump Tariffs Alleged', 'Trump Response']:
            modified_data['Tariff_Differential'] = modified_data['Trump Tariffs Alleged'] - modified_data['Trump Response']
        
        # Make prediction
        X_mod = modified_data.drop(['US 2024 Deficit'], axis=1)
        X_mod_scaled = scaler.transform(X_mod)
        mod_prediction = model.predict(X_mod_scaled)[0]
        
        # Calculate percent change in prediction
        pct_change = (mod_prediction - base_prediction) / abs(base_prediction) * 100
        
        # Store elasticity
        elasticities[feature] = pct_change
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.bar(elasticities.keys(), elasticities.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Trade Deficit Elasticities')
    plt.ylabel('% Change in Deficit for 1% Increase in Variable')
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, 'elasticities.png'))
    
    return elasticities

def analyze_country_clusters(processed_data, output_dir='/Users/gouthamsankar/Linux/maand/output/plots'):
    """Group countries by trade patterns"""
    # Select relevant features
    features = ['US 2024 Exports', 'US 2024 Imports (Customs Basis)', 
                'Trump Tariffs Alleged', 'Trump Response', 'Population',
                'Export_Import_Ratio']
    
    # Prepare data for clustering
    cluster_data = processed_data[['Country'] + features].copy()
    
    # Normalize data for clustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    X_cluster = cluster_data[features]
    X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
    
    # Find optimal number of clusters
    from sklearn.metrics import silhouette_score
    
    silhouette_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_cluster_scaled)
        silhouette_scores.append(silhouette_score(X_cluster_scaled, cluster_labels))
    
    optimal_k = np.argmax(silhouette_scores) + 2
    
    # Perform clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_data['Cluster'] = kmeans.fit_predict(X_cluster_scaled)
    
    # Analyze clusters
    cluster_summary = cluster_data.groupby('Cluster').agg({
        'Country': 'count',
        'US 2024 Exports': 'mean',
        'US 2024 Imports (Customs Basis)': 'mean',
        'Trump Tariffs Alleged': 'mean',
        'Trump Response': 'mean',
        'Export_Import_Ratio': 'mean'
    }).rename(columns={'Country': 'Count'})
    
    # Get example countries for each cluster
    cluster_examples = {}
    for cluster in cluster_data['Cluster'].unique():
        countries = cluster_data[cluster_data['Cluster'] == cluster]['Country'].values[:5]
        cluster_examples[cluster] = ', '.join(countries)
    
    # Plot clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=cluster_data, 
        x='US 2024 Exports', 
        y='US 2024 Imports (Customs Basis)',
        hue='Cluster',
        size='Population',
        sizes=(20, 500),
        alpha=0.7
    )
    plt.title('Country Clusters by Trade Patterns')
    plt.xscale('log')
    plt.yscale('log')
    for i, row in cluster_data.iterrows():
        plt.annotate(row['Country'], (row['US 2024 Exports'], row['US 2024 Imports (Customs Basis)']), 
                     fontsize=8, alpha=0.7)
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, 'country_clusters.png'))
    
    return cluster_data, cluster_summary, cluster_examples

def analyze_prediction_errors(model, scaler, processed_data):
    """Analyze prediction errors by country characteristics"""
    # Make predictions on all data
    X = processed_data.drop(['Country', 'US 2024 Deficit'], axis=1)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    
    # Calculate errors
    error_data = processed_data[['Country', 'US 2024 Deficit']].copy()
    error_data['Predicted'] = predictions
    error_data['Error'] = error_data['US 2024 Deficit'] - error_data['Predicted']
    error_data['Abs_Error'] = error_data['Error'].abs()
    error_data['Pct_Error'] = (error_data['Error'] / error_data['US 2024 Deficit'].abs()) * 100
    
    # Join with original features
    error_analysis = pd.merge(error_data, processed_data, on=['Country', 'US 2024 Deficit'])
    
    # Analyze correlations between errors and features - only use numeric columns
    numeric_error_data = error_analysis.select_dtypes(include=[np.number])
    error_correlations = numeric_error_data.corr()['Abs_Error'].sort_values(ascending=False)
    
    return error_analysis, error_correlations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def analyze_hey_classification_relationships(input_file, output_dir='.'):
    """
    Analyze relationships between Hey's classification and other mineral properties
    
    Parameters:
    -----------
    input_file : str
        Path to the CSV file with Hey's classification columns
    output_dir : str
        Directory to save the visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading the input file: {input_file}")
    # Try different encoding options if needed
    try:
        df = pd.read_csv(input_file)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_file, encoding='latin1')
            print("Using latin1 encoding")
        except:
            df = pd.read_csv(input_file, encoding='utf-8-sig')
            print("Using utf-8-sig encoding")
    
    # Convert potentially problematic columns to string type
    string_columns = ['Crystal Systems', 'Space Groups', 'Paragenetic Modes', 'Chemistry Elements']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Convert age column to numeric, handling errors
    if 'Oldest Known Age (Ma)' in df.columns:
        df['Oldest Known Age (Ma)'] = pd.to_numeric(df['Oldest Known Age (Ma)'], errors='coerce')
    
    # Check if Hey's classification columns exist
    if 'Hey Classification ID' not in df.columns or 'Hey Classification Name' not in df.columns:
        print("Hey's classification columns not found in the input file")
        return
    
    # 1. Age distribution by Hey's classification
    analyze_age_distribution(df, output_dir)
    
    # 2. Analyze relationship between crystal systems and Hey's classification
    analyze_crystal_systems(df, output_dir)
    
    # 3. Analyze space groups by Hey's classification
    analyze_space_groups(df, output_dir)
    
    # 4. Analyze paragenetic modes by Hey's classification
    analyze_paragenetic_modes(df, output_dir)
    
    # 5. Element distribution by Hey's classification
    analyze_element_distribution(df, output_dir)
    
    print("\nAnalysis complete. All visualizations saved to:", output_dir)
    return df

def analyze_age_distribution(df, output_dir):
    """Analyze mineral age distribution by Hey's classification"""
    
    # Filter out missing age data and unclassified minerals
    age_data = df[pd.notna(df['Oldest Known Age (Ma)']) & (df['Hey Classification ID'] != '0')]
    
    # Get the top 8 Hey's classifications by count
    top_classifications = df['Hey Classification ID'].value_counts().head(8).index.tolist()
    age_data = age_data[age_data['Hey Classification ID'].isin(top_classifications)]
    
    # Check if we have enough data
    if len(age_data) < 10:
        print("Not enough age data for analysis")
        return
    
    # Create ID to name mapping
    id_to_name = {id_val: f"{id_val}: {name[:20]}..." for id_val, name in 
                  df[['Hey Classification ID', 'Hey Classification Name']].drop_duplicates().values}
    
    # Replace IDs with readable names
    age_data = age_data.copy()  # Create a copy to avoid SettingWithCopyWarning
    age_data['Classification'] = age_data['Hey Classification ID'].map(id_to_name)
    
    # Create box plot of ages by classification
    plt.figure(figsize=(14, 10))  # Increased height to accommodate labels
    sns.boxplot(
        x='Classification',
        y='Oldest Known Age (Ma)',
        data=age_data,
        hue='Classification',
        palette='viridis',
        legend=False
    )
    
    plt.title('Distribution of Mineral Ages by Hey\'s Classification')
    plt.xlabel('Hey\'s Classification')
    plt.ylabel('Oldest Known Age (Ma)')
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout with more padding
    plt.subplots_adjust(bottom=0.3, top=0.9)
    
    # Save the figure
    age_box_path = os.path.join(output_dir, 'hey_classification_age_boxplot.png')
    plt.savefig(age_box_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create violin plot to better see the distribution
    plt.figure(figsize=(14, 10))  # Increased height to accommodate labels
    sns.violinplot(
        x='Classification',
        y='Oldest Known Age (Ma)',
        data=age_data,
        hue='Classification',
        palette='viridis',
        inner='quartile',
        legend=False
    )
    
    plt.title('Distribution of Mineral Ages by Hey\'s Classification (Violin Plot)')
    plt.xlabel('Hey\'s Classification')
    plt.ylabel('Oldest Known Age (Ma)')
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout with more padding
    plt.subplots_adjust(bottom=0.3, top=0.9)
    
    # Save the figure
    age_violin_path = os.path.join(output_dir, 'hey_classification_age_violin.png')
    plt.savefig(age_violin_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Age distribution analysis saved to:\n- {age_box_path}\n- {age_violin_path}")

def analyze_crystal_systems(df, output_dir):
    """Analyze relationship between crystal systems and Hey's classification"""
    
    # Filter out missing crystal system data and unclassified minerals
    crystal_data = df[(df['Crystal Systems'] != 'nan') & (df['Hey Classification ID'] != '0')]
    
    # Get the top 8 Hey's classifications by count
    top_classifications = df['Hey Classification ID'].value_counts().head(8).index.tolist()
    crystal_data = crystal_data[crystal_data['Hey Classification ID'].isin(top_classifications)]
    
    # Check if we have enough data
    if len(crystal_data) < 10:
        print("Not enough crystal system data for analysis")
        return
    
    # Create ID to name mapping
    id_to_name = {id_val: f"{id_val}: {name[:20]}..." for id_val, name in 
                  df[['Hey Classification ID', 'Hey Classification Name']].drop_duplicates().values}
    
    # Create a crosstab of Hey classification vs crystal system
    crystal_cross = pd.crosstab(
        crystal_data['Hey Classification ID'], 
        crystal_data['Crystal Systems'],
        normalize='index'
    ) * 100  # Convert to percentages
    
    # Replace IDs with readable names
    crystal_cross.index = [id_to_name.get(idx, idx) for idx in crystal_cross.index]
    
    # Plot as a stacked bar chart
    plt.figure(figsize=(16, 10))
    crystal_cross.plot(
        kind='bar', 
        stacked=True,
        figsize=(16, 8),
        colormap='tab20'
    )
    
    plt.title('Crystal Systems Distribution by Hey\'s Classification')
    plt.xlabel('Hey\'s Classification')
    #plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Crystal System', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the figure
    crystal_path = os.path.join(output_dir, 'hey_classification_crystal_systems.png')
    plt.savefig(crystal_path, dpi=300)
    plt.close()
    
    # Create a heatmap for a different visualization
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        crystal_cross,
        annot=False,
        cmap='viridis',
        fmt='.1f'
    )
    
    plt.title('Crystal Systems by Hey\'s Classification (Heatmap)')
    plt.tight_layout()
    
    # Save the figure
    crystal_heatmap_path = os.path.join(output_dir, 'hey_classification_crystal_heatmap.png')
    plt.savefig(crystal_heatmap_path, dpi=300)
    plt.close()
    
    print(f"Crystal systems analysis saved to:\n- {crystal_path}\n- {crystal_heatmap_path}")

def analyze_space_groups(df, output_dir):
    """Analyze space groups by Hey's classification"""
    
    # Filter out missing space group data and unclassified minerals
    space_data = df[(df['Space Groups'] != 'nan') & (df['Hey Classification ID'] != '0')]
    
    # Get the top 5 Hey's classifications by count
    top_classifications = df['Hey Classification ID'].value_counts().head(5).index.tolist()
    space_data = space_data[space_data['Hey Classification ID'].isin(top_classifications)]
    
    # Check if we have enough data
    if len(space_data) < 10:
        print("Not enough space group data for analysis")
        return
    
    # Create ID to name mapping
    id_to_name = {id_val: f"{id_val}: {name[:20]}..." for id_val, name in 
                  df[['Hey Classification ID', 'Hey Classification Name']].drop_duplicates().values}
    
    # Since there are many space groups, let's get the top 15 most common ones
    top_space_groups = space_data['Space Groups'].value_counts().head(15).index.tolist()
    space_data_filtered = space_data[space_data['Space Groups'].isin(top_space_groups)].copy()  # Create a copy
    
    # Replace IDs with readable names
    space_data_filtered['Classification'] = space_data_filtered['Hey Classification ID'].map(id_to_name)
    
    # Create a crosstab
    space_cross = pd.crosstab(
        space_data_filtered['Classification'], 
        space_data_filtered['Space Groups'],
        normalize='index'
    ) * 100  # Convert to percentages
    
    # Create a heatmap
    plt.figure(figsize=(18, 12))  # Increased size for better visibility
    sns.heatmap(
        space_cross,
        annot=True,
        cmap='viridis',
        fmt='.1f'
    )
    
    plt.title('Top Space Groups by Hey\'s Classification')
    
    # Adjust layout with more padding
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.2, right=0.9)
    
    # Save the figure
    space_path = os.path.join(output_dir, 'hey_classification_space_groups.png')
    plt.savefig(space_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Space groups analysis saved to: {space_path}")

def analyze_paragenetic_modes(df, output_dir):
    """Analyze paragenetic modes by Hey's classification"""
    
    # Filter out missing paragenetic mode data and unclassified minerals
    para_data = df[(df['Paragenetic Modes'] != 'nan') & (df['Hey Classification ID'] != '0')]
    
    # Get the top 5 Hey's classifications by count
    top_classifications = df['Hey Classification ID'].value_counts().head(5).index.tolist()
    para_data = para_data[para_data['Hey Classification ID'].isin(top_classifications)]
    
    # Check if we have enough data
    if len(para_data) < 10:
        print("Not enough paragenetic mode data for analysis")
        return
    
    # Create ID to name mapping
    id_to_name = {id_val: f"{id_val}: {name[:20]}..." for id_val, name in 
                  df[['Hey Classification ID', 'Hey Classification Name']].drop_duplicates().values}
    
    # Create a copy to avoid SettingWithCopyWarning
    para_data = para_data.copy()
    
    # Paragenetic modes can be complex strings with multiple modes
    # Let's extract main categories: Biotic vs Abiotic and Primary vs Secondary
    para_data['Is Biotic'] = para_data['Paragenetic Modes'].str.contains('Biotic').fillna(False)
    para_data['Is Primary'] = para_data['Paragenetic Modes'].str.contains('Primary').fillna(False)
    para_data['Is Secondary'] = para_data['Paragenetic Modes'].str.contains('Secondary').fillna(False)
    
    # Replace IDs with readable names
    para_data['Classification'] = para_data['Hey Classification ID'].map(id_to_name)
    
    # Create biotic/abiotic distribution
    biotic_cross = pd.crosstab(
        para_data['Classification'], 
        para_data['Is Biotic'],
        normalize='index'
    ) * 100  # Convert to percentages
    
    # Rename columns for clarity
    biotic_cross.columns = ['Abiotic', 'Biotic']
    
    # Plot biotic vs abiotic
    plt.figure(figsize=(14, 10))
    biotic_cross.plot(
        kind='bar', 
        stacked=True,
        colormap='viridis'
    )
    
    plt.title('Biotic vs Abiotic Distribution by Hey\'s Classification')
    plt.xlabel('Hey\'s Classification')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Origin')
    
    # Adjust layout with more padding
    plt.subplots_adjust(bottom=0.3, top=0.9)
    
    # Save the figure
    biotic_path = os.path.join(output_dir, 'hey_classification_biotic.png')
    plt.savefig(biotic_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create primary/secondary distribution
    primary_data = para_data[para_data['Is Primary'] | para_data['Is Secondary']]
    
    if not primary_data.empty:
        # Create a new DataFrame for primary/secondary analysis
        primary_analysis = pd.DataFrame(index=primary_data['Classification'].unique())
        
        # Initialize columns with zeros
        primary_analysis['Primary'] = 0.0
        primary_analysis['Secondary'] = 0.0
        
        # Calculate percentages for primary and secondary
        for classification in primary_analysis.index:
            class_data = primary_data[primary_data['Classification'] == classification]
            total = len(class_data)
            if total > 0:
                primary_analysis.loc[classification, 'Primary'] = float((class_data['Is Primary'].sum() / total) * 100)
                primary_analysis.loc[classification, 'Secondary'] = float((class_data['Is Secondary'].sum() / total) * 100)
        
        # Ensure all values are numeric
        primary_analysis = primary_analysis.astype(float)
        
        # Plot primary vs secondary
        plt.figure(figsize=(14, 10))
        primary_analysis.plot(
            kind='bar', 
            stacked=True,
            colormap='tab10'
        )
        
        plt.title('Primary vs Secondary Distribution by Hey\'s Classification')
        plt.xlabel('Hey\'s Classification')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Formation')
        
        # Adjust layout with more padding
        plt.subplots_adjust(bottom=0.3, top=0.9)
        
        # Save the figure
        primary_path = os.path.join(output_dir, 'hey_classification_primary.png')
        plt.savefig(primary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Paragenetic modes analysis saved to:\n- {biotic_path}\n- {primary_path}")
    else:
        print(f"Paragenetic modes analysis saved to:\n- {biotic_path}")

def analyze_element_distribution(df, output_dir):
    """Analyze element distribution by Hey's classification"""
    
    # Filter out missing element data and unclassified minerals
    element_data = df[(df['Chemistry Elements'] != 'nan') & (df['Hey Classification ID'] != '0')]
    
    # Get the top 8 Hey's classifications by count
    top_classifications = df['Hey Classification ID'].value_counts().head(8).index.tolist()
    element_data = element_data[element_data['Hey Classification ID'].isin(top_classifications)]
    
    # Check if we have enough data
    if len(element_data) < 10:
        print("Not enough element data for analysis")
        return
    
    # Create ID to name mapping
    id_to_name = {id_val: f"{id_val}: {name[:20]}..." for id_val, name in 
                  df[['Hey Classification ID', 'Hey Classification Name']].drop_duplicates().values}
    
    # Create a copy to avoid SettingWithCopyWarning
    element_data = element_data.copy()
    
    # Replace IDs with readable names
    element_data['Classification'] = element_data['Hey Classification ID'].map(id_to_name)
    
    # Process the element strings to count occurrence of each element
    all_elements = set()
    for elements in element_data['Chemistry Elements'].dropna():
        all_elements.update(elements.split())
    
    # Sort elements
    all_elements = sorted(list(all_elements))
    
    # Get the 20 most common elements
    element_counts = {}
    for element in all_elements:
        element_counts[element] = element_data['Chemistry Elements'].str.contains(f'\\b{element}\\b').sum()
    
    top_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    top_element_names = [e[0] for e in top_elements]
    
    # Create a matrix of elements vs Hey's classifications
    element_matrix = pd.DataFrame(index=element_data['Classification'].unique(), columns=top_element_names)
    
    # Initialize with zeros
    for col in element_matrix.columns:
        element_matrix[col] = 0.0
    
    # Calculate percentages
    for classification in element_matrix.index:
        class_data = element_data[element_data['Classification'] == classification]
        total_minerals = len(class_data)
        
        if total_minerals > 0:
            for element in top_element_names:
                element_count = class_data['Chemistry Elements'].str.contains(f'\\b{element}\\b').sum()
                element_matrix.loc[classification, element] = float((element_count / total_minerals) * 100)
    
    # Ensure all values are numeric
    element_matrix = element_matrix.astype(float)
    
    # Sort the matrix by classification
    element_matrix = element_matrix.loc[sorted(element_matrix.index)]
    
    # Create a heatmap
    plt.figure(figsize=(18, 12))
    sns.heatmap(
        element_matrix,
        annot=True,
        cmap='viridis',
        fmt='.1f'
    )
    
    plt.title('Element Distribution by Hey\'s Classification (Percentage)')
    
    # Adjust layout with more padding
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.2, right=0.9)
    
    # Save the figure
    element_path = os.path.join(output_dir, 'hey_classification_elements.png')
    plt.savefig(element_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Element distribution analysis saved to: {element_path}")

def plot_hey_classification_distribution(df, output_dir, **kwargs):
    """Create a pie or bar chart showing the distribution of Hey's classifications"""
    
    # Get chart type from kwargs, default to bar
    chart_type = kwargs.get('chart_type', 'bar')
    
    # Filter out unclassified minerals
    classified_data = df[df['Hey Classification ID'] != '0']
    
    # Count occurrences of each classification
    classification_counts = classified_data['Hey Classification ID'].value_counts()
    
    # Create ID to name mapping
    id_to_name = {id_val: f"{id_val}: {name[:20]}..." for id_val, name in 
                  df[['Hey Classification ID', 'Hey Classification Name']].drop_duplicates().values}
    
    # Replace IDs with readable names
    classification_counts.index = classification_counts.index.map(id_to_name)
    
    # Sort by count
    classification_counts = classification_counts.sort_values(ascending=False)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    if chart_type == 'pie':
        # Create pie chart
        plt.pie(
            classification_counts,
            labels=classification_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.85
        )
        plt.title('Distribution of Hey\'s Classifications (Pie Chart)')
        plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
        
    else:  # bar chart
        # Create bar chart
        classification_counts.plot(
            kind='bar',
            color='viridis',
            width=0.8
        )
        plt.title('Distribution of Hey\'s Classifications (Bar Chart)')
        plt.xlabel('Hey\'s Classification')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.3, top=0.9)
    
    # Save the figure
    chart_type_str = 'pie' if chart_type == 'pie' else 'bar'
    output_path = os.path.join(output_dir, f'hey_classification_distribution_{chart_type_str}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Hey's classification distribution {chart_type_str} chart saved to: {output_path}")

if __name__ == "__main__":
    input_file = "RRUFF_Export_with_Hey_Classification.csv"
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        input_file = input("Enter the path to the CSV file with Hey's classification: ")
        
    if os.path.exists(input_file):
        analyze_hey_classification_relationships(input_file)
    else:
        print("File not found. Please run the add_hey_classification.py script first.")

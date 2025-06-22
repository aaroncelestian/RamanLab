import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import numpy as np

def plot_hey_classification_distribution(input_file, output_dir='.'):
    """
    Create visualizations of Hey's classification distribution from the processed CSV
    
    Parameters:
    -----------
    input_file : str
        Path to the CSV file with Hey's classification columns
    output_dir : str
        Directory to save the visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading the input file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Check if Hey's classification columns exist
    if 'Hey Classification ID' not in df.columns or 'Hey Classification Name' not in df.columns:
        print("Hey's classification columns not found in the input file")
        return
    
    # Count classifications
    hey_counts = df['Hey Classification ID'].value_counts().reset_index()
    hey_counts.columns = ['Hey Classification ID', 'Count']
    
    # Merge with names
    hey_names = df[['Hey Classification ID', 'Hey Classification Name']].drop_duplicates()
    hey_distribution = pd.merge(hey_counts, hey_names, on='Hey Classification ID')
    
    # Sort by ID and exclude unclassified
    hey_distribution = hey_distribution[hey_distribution['Hey Classification ID'] != '0'].sort_values('Count', ascending=False)
    
    # Calculate percentages
    total_minerals = len(df)
    classified_minerals = total_minerals - df[df['Hey Classification ID'] == '0'].shape[0]
    hey_distribution['Percentage'] = hey_distribution['Count'] / total_minerals * 100
    
    # Print summary
    print("\nHey's Classification Distribution:")
    print(f"Total minerals: {total_minerals}")
    print(f"Classified minerals: {classified_minerals} ({classified_minerals/total_minerals*100:.1f}%)")
    print(f"Unclassified minerals: {total_minerals - classified_minerals} ({(total_minerals - classified_minerals)/total_minerals*100:.1f}%)")
    
    # Print top 10 categories
    print("\nTop 10 Hey's Classification Categories:")
    for _, row in hey_distribution.head(10).iterrows():
        print(f"{row['Hey Classification ID']}: {row['Hey Classification Name']}: {row['Count']} minerals ({row['Percentage']:.1f}%)")
    
    # Create visualizations
    
    # 1. Bar chart of top 15 categories
    plt.figure(figsize=(14, 8))
    top_n = 15
    top_categories = hey_distribution.head(top_n)
    
    # Create shortened labels
    max_label_length = 40
    labels = [f"{row['Hey Classification ID']}: {row['Hey Classification Name'][:max_label_length]}{'...' if len(row['Hey Classification Name']) > max_label_length else ''}" 
              for _, row in top_categories.iterrows()]
    
    # Bar plot
    sns.barplot(
        x='Count', 
        y=labels,
        data=top_categories,
        palette='viridis'
    )
    
    plt.title(f'Top {top_n} Hey\'s Classification Categories')
    plt.xlabel('Number of Minerals')
    plt.ylabel('Classification Category')
    plt.tight_layout()
    
    bar_chart_path = os.path.join(output_dir, 'hey_classification_top15_bar.png')
    plt.savefig(bar_chart_path, dpi=300)
    plt.close()
    
    # 2. Pie chart of top categories
    plt.figure(figsize=(12, 12))
    
    # Get top categories and group the rest as "Other"
    top_n_pie = 8
    top_categories_pie = hey_distribution.head(top_n_pie).copy()
    other_count = hey_distribution.iloc[top_n_pie:]['Count'].sum()
    
    # Add "Other" category if needed
    if other_count > 0:
        other_row = pd.DataFrame({
            'Hey Classification ID': ['Other'],
            'Count': [other_count],
            'Hey Classification Name': ['Other Categories'],
            'Percentage': [hey_distribution.iloc[top_n_pie:]['Percentage'].sum()]
        })
        pie_data = pd.concat([top_categories_pie, other_row])
    else:
        pie_data = top_categories_pie
    
    # Create shortened labels with percentages
    pie_labels = [f"{row['Hey Classification ID']}: {row['Hey Classification Name'][:25]}{'...' if len(row['Hey Classification Name']) > 25 else ''} ({row['Percentage']:.1f}%)" 
                 for _, row in pie_data.iterrows()]
    
    # Pie plot
    plt.pie(
        pie_data['Count'],
        labels=pie_labels,
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        explode=[0.05] * len(pie_data),  # Slight separation for all slices
        wedgeprops={'edgecolor': 'white'}
    )
    
    plt.axis('equal')
    plt.title('Distribution of Hey\'s Classification Categories')
    
    pie_chart_path = os.path.join(output_dir, 'hey_classification_pie.png')
    plt.savefig(pie_chart_path, dpi=300)
    plt.close()
    
    # 3. Create a histogram of crystal systems by Hey's classification
    plt.figure(figsize=(16, 10))
    
    # Get top 6 Hey classifications
    top_classifications = hey_distribution.head(6)['Hey Classification ID'].tolist()
    
    # Filter dataset for these classifications and filter out missing crystal systems
    crystal_data = df[df['Hey Classification ID'].isin(top_classifications)]
    crystal_data = crystal_data[~crystal_data['Crystal Systems'].isna()]
    
    # Create a crosstab of Hey classification vs crystal system
    crystal_cross = pd.crosstab(
        crystal_data['Hey Classification ID'], 
        crystal_data['Crystal Systems'],
        normalize='index'
    ) * 100  # Convert to percentages
    
    # Use a readable name for the Hey classification
    id_to_name = {row['Hey Classification ID']: f"{row['Hey Classification ID']}: {row['Hey Classification Name'][:30]}..." 
                  for _, row in top_categories.iterrows()}
    
    crystal_cross.index = [id_to_name.get(idx, idx) for idx in crystal_cross.index]
    
    # Plot as a stacked bar chart
    crystal_cross.plot(
        kind='bar', 
        stacked=True,
        figsize=(16, 8),
        colormap='tab20'
    )
    
    plt.title('Crystal Systems Distribution by Hey\'s Classification')
    plt.xlabel('Hey\'s Classification')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Crystal System', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    crystal_path = os.path.join(output_dir, 'hey_classification_crystal_systems.png')
    plt.savefig(crystal_path, dpi=300)
    plt.close()
    
    print(f"\nVisualizations saved to:")
    print(f"- Bar chart: {bar_chart_path}")
    print(f"- Pie chart: {pie_chart_path}")
    print(f"- Crystal systems chart: {crystal_path}")
    
    return hey_distribution

if __name__ == "__main__":
    input_file = "RRUFF_Export_with_Hey_Classification.csv"
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        input_file = input("Enter the path to the CSV file with Hey's classification: ")
        
    if os.path.exists(input_file):
        plot_hey_classification_distribution(input_file)
    else:
        print("File not found. Please run the add_hey_classification.py script first.")

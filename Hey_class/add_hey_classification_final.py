import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import seaborn as sns
from collections import Counter

# Define Hey's classification categories
HEY_CATEGORIES = {
    '1': 'Elements and Alloys (including the arsenides, antimonides and bismuthides of Cu, Ag and Au)',
    '2': 'Carbides, Nitrides, Silicides and Phosphides',
    '3': 'Sulphides, Selenides, Tellurides, Arsenides and Bismuthides',
    '4': 'Oxysulphides',
    '5': 'Sulphosalts - Sulpharsenites and Sulphobismuthites',
    '6': 'Sulphosalts - Sulphostannates, Sulphogermanates, Sulpharsenates, Sulphantimonates, Sulphovanadates and Sulphohalides',
    '7': 'Oxides and Hydroxides',
    '8': 'Halides',
    '9': 'Borates',
    '10': 'Borates with other anions',
    '11': 'Carbonates',
    '12': 'Carbonates with other anions',
    '13': 'Nitrates',
    '14': 'Silicates not Containing Aluminum',
    '15': 'Silicates of Aluminum',
    '16': 'Silicates Containing Aluminum and other Metals',
    '17': 'Silicates Containing other Anions',
    '18': 'Niobates and Tantalates',
    '19': 'Phosphates',
    '20': 'Arsenates',
    '21': 'Vanadates',
    '22': 'Phosphates, Arsenates or Vanadates with other Anions',
    '23': 'Arsenites',
    '24': 'Antimonates and Antimonites',
    '25': 'Sulphates',
    '26': 'Sulphates with Halide',
    '27': 'Sulphites, Chromates, Molybdates and Tungstates',
    '28': 'Selenites, Selenates, Tellurites, and Tellurates',
    '29': 'Iodates',
    '30': 'Thiocyanates',
    '31': 'Oxalates, Citrates, Mellitates and Acetates',
    '32': 'Hydrocarbons, Resins and other Organic Compounds'
}

def clean_chemistry(chemistry):
    """
    Clean up chemical formulas by removing superscripts, subscripts, and special characters
    """
    if not isinstance(chemistry, str):
        return ""
    
    # Remove superscripts
    chemistry = re.sub(r'\^[0-9+\-]+\^', '', chemistry)
    
    # Remove subscripts
    chemistry = re.sub(r'_[0-9]+_', '', chemistry)
    
    # Remove hydration dots
    chemistry = chemistry.replace('·', '')
    
    return chemistry

def contains(pattern, text):
    """
    Check if a pattern exists in text
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(pattern, text))

def classify_mineral(chemistry, elements):
    """
    Classify a mineral according to Hey's classification system based on its chemistry and elements
    
    Parameters:
    -----------
    chemistry : str
        RRUFF Chemistry (concise) formula
    elements : str
        Comma-separated list of elements
    
    Returns:
    --------
    dict
        Dictionary with id and name of the Hey classification
    """
    # Default classification
    default_classification = {'id': '0', 'name': 'Unclassified'}
    
    # Handle empty inputs
    if not isinstance(chemistry, str) or not isinstance(elements, str):
        return default_classification
    
    # Split elements string into array and clean
    element_list = [el.strip() for el in elements.split(',')]
    
    # Clean the chemistry for pattern matching
    clean = clean_chemistry(chemistry)
    
    # First check for specific anion groups in the chemistry formula
    # 25, 26. Sulfates
    if contains('SO4', clean) or contains('SO_4', clean) or contains('\\(SO', clean):
        if any(el in ['F', 'Cl', 'Br', 'I'] for el in element_list):
            return {'id': '26', 'name': HEY_CATEGORIES['26']}
        return {'id': '25', 'name': HEY_CATEGORIES['25']}
    
    # 11, 12. Carbonates
    if contains('CO3', clean) or contains('CO_3', clean) or contains('\\(CO', clean):
        if (any(el in ['F', 'Cl', 'Br', 'I'] for el in element_list) or 
            contains('OH', clean) or contains('SO4', clean) or contains('PO4', clean)):
            return {'id': '12', 'name': HEY_CATEGORIES['12']}
        return {'id': '11', 'name': HEY_CATEGORIES['11']}
    
    # 13. Nitrates
    if contains('NO3', clean) or contains('NO_3', clean):
        return {'id': '13', 'name': HEY_CATEGORIES['13']}
    
    # 14, 15, 16, 17. Silicates
    if (contains('SiO', clean) or contains('Si2O', clean) or contains('Si3O', clean) or 
        contains('Si4O', clean) or contains('Si5O', clean) or contains('Si6O', clean) or
        contains('Si7O', clean) or contains('Si8O', clean) or contains('Si9O', clean) or
        contains('Si10O', clean) or contains('Si11O', clean) or contains('Si12O', clean)):
        # First check for specific aluminum silicate patterns where Al is part of the silicate structure
        if (contains('Si3Al2O10', clean) or contains('Si3Al2O_10', clean) or
            contains('Si9Al6O30', clean) or contains('Si9Al6O_30', clean) or
            contains('Al4Si4O16', clean) or contains('Al4Si4O_16', clean) or
            contains('Si2Al2O8', clean) or contains('Si2Al2O_8', clean) or
            contains('Si3AlO8', clean) or contains('Si3AlO_8', clean) or
            contains('Si2AlO5', clean) or contains('Si2AlO_5', clean) or
            contains('SiAlO4', clean) or contains('SiAlO_4', clean) or
            contains('SiAl2O5', clean) or contains('SiAl2O_5', clean) or
            contains('SiAl4O8', clean) or contains('SiAl4O_8', clean)):
            # These are true aluminum silicates where Al is part of the silicate structure
            if any(el in ['F', 'Cl', 'Br', 'I', 'S', 'P', 'As', 'V'] for el in element_list):
                return {'id': '17', 'name': HEY_CATEGORIES['17']}
            return {'id': '15', 'name': HEY_CATEGORIES['15']}
        
        # Check for specific silicate patterns
        if contains('SiO2', clean) or contains('SiO_2', clean):  # Framework silicates
            if 'Al' in element_list:
                # Check if Al is part of the silicate structure
                if (contains('AlSiO', clean) or contains('AlSi2O', clean) or 
                    contains('Al2SiO', clean) or contains('Al2Si2O', clean)):
                    if any(el in ['F', 'Cl', 'Br', 'I', 'S', 'P', 'As', 'V'] for el in element_list):
                        return {'id': '17', 'name': HEY_CATEGORIES['17']}
                    return {'id': '15', 'name': HEY_CATEGORIES['15']}
                # Al is just another cation, not part of silicate structure
                return {'id': '14', 'name': HEY_CATEGORIES['14']}
            return {'id': '14', 'name': HEY_CATEGORIES['14']}
        
        if contains('Si2O5', clean) or contains('Si2O_5', clean):  # Phyllosilicates
            if 'Al' in element_list:
                # Check if Al is part of the silicate structure
                if (contains('AlSiO', clean) or contains('AlSi2O', clean) or 
                    contains('Al2SiO', clean) or contains('Al2Si2O', clean)):
                    if any(el in ['F', 'Cl', 'Br', 'I', 'S', 'P', 'As', 'V'] for el in element_list):
                        return {'id': '17', 'name': HEY_CATEGORIES['17']}
                    return {'id': '15', 'name': HEY_CATEGORIES['15']}
                # Al is just another cation, not part of silicate structure
                return {'id': '14', 'name': HEY_CATEGORIES['14']}
            return {'id': '14', 'name': HEY_CATEGORIES['14']}
        
        if contains('Si2O7', clean) or contains('Si2O_7', clean):  # Sorosilicates
            if 'Al' in element_list:
                # Check if Al is part of the silicate structure
                if (contains('AlSiO', clean) or contains('AlSi2O', clean) or 
                    contains('Al2SiO', clean) or contains('Al2Si2O', clean)):
                    if any(el in ['F', 'Cl', 'Br', 'I', 'S', 'P', 'As', 'V'] for el in element_list):
                        return {'id': '17', 'name': HEY_CATEGORIES['17']}
                    return {'id': '15', 'name': HEY_CATEGORIES['15']}
                # Al is just another cation, not part of silicate structure
                return {'id': '14', 'name': HEY_CATEGORIES['14']}
            return {'id': '14', 'name': HEY_CATEGORIES['14']}
        
        if contains('SiO4', clean) or contains('SiO_4', clean):  # Nesosilicates
            if 'Al' in element_list:
                # Check if Al is part of the silicate structure
                if (contains('AlSiO', clean) or contains('AlSi2O', clean) or 
                    contains('Al2SiO', clean) or contains('Al2Si2O', clean)):
                    if any(el in ['F', 'Cl', 'Br', 'I', 'S', 'P', 'As', 'V'] for el in element_list):
                        return {'id': '17', 'name': HEY_CATEGORIES['17']}
                    return {'id': '15', 'name': HEY_CATEGORIES['15']}
                # Al is just another cation, not part of silicate structure
                return {'id': '14', 'name': HEY_CATEGORIES['14']}
            return {'id': '14', 'name': HEY_CATEGORIES['14']}
        
        if contains('Si3O8', clean) or contains('Si3O_8', clean):  # Tektosilicates
            if 'Al' in element_list:
                # Check if Al is part of the silicate structure
                if (contains('AlSiO', clean) or contains('AlSi2O', clean) or 
                    contains('Al2SiO', clean) or contains('Al2Si2O', clean)):
                    if any(el in ['F', 'Cl', 'Br', 'I', 'S', 'P', 'As', 'V'] for el in element_list):
                        return {'id': '17', 'name': HEY_CATEGORIES['17']}
                    return {'id': '15', 'name': HEY_CATEGORIES['15']}
                # Al is just another cation, not part of silicate structure
                return {'id': '14', 'name': HEY_CATEGORIES['14']}
            return {'id': '14', 'name': HEY_CATEGORIES['14']}
        
        if contains('Si3O9', clean) or contains('Si3O_9', clean):  # Cyclosilicates
            if 'Al' in element_list:
                # Check if Al is part of the silicate structure
                if (contains('AlSiO', clean) or contains('AlSi2O', clean) or 
                    contains('Al2SiO', clean) or contains('Al2Si2O', clean)):
                    if any(el in ['F', 'Cl', 'Br', 'I', 'S', 'P', 'As', 'V'] for el in element_list):
                        return {'id': '17', 'name': HEY_CATEGORIES['17']}
                    return {'id': '15', 'name': HEY_CATEGORIES['15']}
                # Al is just another cation, not part of silicate structure
                return {'id': '14', 'name': HEY_CATEGORIES['14']}
            return {'id': '14', 'name': HEY_CATEGORIES['14']}
        
        if contains('Si4O11', clean) or contains('Si4O_11', clean):  # Inosilicates
            if 'Al' in element_list:
                # Check if Al is part of the silicate structure
                if (contains('AlSiO', clean) or contains('AlSi2O', clean) or 
                    contains('Al2SiO', clean) or contains('Al2Si2O', clean)):
                    if any(el in ['F', 'Cl', 'Br', 'I', 'S', 'P', 'As', 'V'] for el in element_list):
                        return {'id': '17', 'name': HEY_CATEGORIES['17']}
                    return {'id': '15', 'name': HEY_CATEGORIES['15']}
                # Al is just another cation, not part of silicate structure
                return {'id': '14', 'name': HEY_CATEGORIES['14']}
            return {'id': '14', 'name': HEY_CATEGORIES['14']}
        
        if contains('Si6O18', clean) or contains('Si6O_18', clean):  # Cyclosilicates
            if 'Al' in element_list:
                # Check if Al is part of the silicate structure
                if (contains('AlSiO', clean) or contains('AlSi2O', clean) or 
                    contains('Al2SiO', clean) or contains('Al2Si2O', clean)):
                    if any(el in ['F', 'Cl', 'Br', 'I', 'S', 'P', 'As', 'V'] for el in element_list):
                        return {'id': '17', 'name': HEY_CATEGORIES['17']}
                    return {'id': '15', 'name': HEY_CATEGORIES['15']}
                # Al is just another cation, not part of silicate structure
                return {'id': '14', 'name': HEY_CATEGORIES['14']}
            return {'id': '14', 'name': HEY_CATEGORIES['14']}
        
        if contains('Si8O20', clean) or contains('Si8O_20', clean):  # Phyllosilicates
            if 'Al' in element_list:
                # Check if Al is part of the silicate structure
                if (contains('AlSiO', clean) or contains('AlSi2O', clean) or 
                    contains('Al2SiO', clean) or contains('Al2Si2O', clean)):
                    if any(el in ['F', 'Cl', 'Br', 'I', 'S', 'P', 'As', 'V'] for el in element_list):
                        return {'id': '17', 'name': HEY_CATEGORIES['17']}
                    return {'id': '15', 'name': HEY_CATEGORIES['15']}
                # Al is just another cation, not part of silicate structure
                return {'id': '14', 'name': HEY_CATEGORIES['14']}
            return {'id': '14', 'name': HEY_CATEGORIES['14']}
        
        if contains('Si12O30', clean) or contains('Si12O_30', clean):  # Cyclosilicates
            if 'Al' in element_list:
                # Check if Al is part of the silicate structure
                if (contains('AlSiO', clean) or contains('AlSi2O', clean) or 
                    contains('Al2SiO', clean) or contains('Al2Si2O', clean)):
                    if any(el in ['F', 'Cl', 'Br', 'I', 'S', 'P', 'As', 'V'] for el in element_list):
                        return {'id': '17', 'name': HEY_CATEGORIES['17']}
                    return {'id': '15', 'name': HEY_CATEGORIES['15']}
                # Al is just another cation, not part of silicate structure
                return {'id': '14', 'name': HEY_CATEGORIES['14']}
            return {'id': '14', 'name': HEY_CATEGORIES['14']}
        
        if contains('Si12O36', clean) or contains('Si12O_36', clean):  # Cyclosilicates
            if 'Al' in element_list:
                # Check if Al is part of the silicate structure
                if (contains('AlSiO', clean) or contains('AlSi2O', clean) or 
                    contains('Al2SiO', clean) or contains('Al2Si2O', clean)):
                    if any(el in ['F', 'Cl', 'Br', 'I', 'S', 'P', 'As', 'V'] for el in element_list):
                        return {'id': '17', 'name': HEY_CATEGORIES['17']}
                    return {'id': '15', 'name': HEY_CATEGORIES['15']}
                # Al is just another cation, not part of silicate structure
                return {'id': '14', 'name': HEY_CATEGORIES['14']}
            return {'id': '14', 'name': HEY_CATEGORIES['14']}
        
        # Generic silicate case for any other Si-O ratio
        if 'Al' in element_list:
            # Check if Al is part of the silicate structure
            if (contains('AlSiO', clean) or contains('AlSi2O', clean) or 
                contains('Al2SiO', clean) or contains('Al2Si2O', clean)):
                if any(el in ['F', 'Cl', 'Br', 'I', 'S', 'P', 'As', 'V'] for el in element_list):
                    return {'id': '17', 'name': HEY_CATEGORIES['17']}
                return {'id': '15', 'name': HEY_CATEGORIES['15']}
            # Al is just another cation, not part of silicate structure
            return {'id': '14', 'name': HEY_CATEGORIES['14']}
        return {'id': '14', 'name': HEY_CATEGORIES['14']}
    
    # 30. Thiocyanates
    if (contains('SCN', clean) or contains('S-CN', clean) or contains('S(CN)', clean) or
        ('S' in element_list and 'C' in element_list and 'N' in element_list)):
        return {'id': '30', 'name': HEY_CATEGORIES['30']}
    
    # 20, 23. Arsenates and Arsenites
    if contains('AsO4', clean) or contains('AsO_4', clean) or contains('\\(AsO', clean):
        # 22. Mixed anions or with other anions
        if (any(el in ['F', 'Cl', 'Br', 'I', 'S', 'OH'] for el in element_list) or 
            contains('OH', clean) or contains('SO4', clean) or contains('CO3', clean)):
            return {'id': '22', 'name': HEY_CATEGORIES['22']}
        # 20. Arsenates
        return {'id': '20', 'name': HEY_CATEGORIES['20']}
    
    if contains('AsO3', clean) or contains('AsO_3', clean) or contains('\\(AsO3', clean):
        # 23. Arsenites
        return {'id': '23', 'name': HEY_CATEGORIES['23']}
    
    # 19, 21, 22. Phosphates and Vanadates
    if contains('PO4', clean) or contains('PO_4', clean) or contains('\\(PO', clean):
        # 22. Mixed anions or with other anions
        if (any(el in ['F', 'Cl', 'Br', 'I', 'S', 'OH'] for el in element_list) or 
            contains('OH', clean) or contains('SO4', clean) or contains('CO3', clean)):
            return {'id': '22', 'name': HEY_CATEGORIES['22']}
        # 19. Phosphates
        return {'id': '19', 'name': HEY_CATEGORIES['19']}
    
    if contains('VO4', clean) or contains('VO_4', clean) or contains('\\(VO', clean):
        # 22. Mixed anions or with other anions
        if (any(el in ['F', 'Cl', 'Br', 'I', 'S', 'OH'] for el in element_list) or 
            contains('OH', clean) or contains('SO4', clean) or contains('CO3', clean)):
            return {'id': '22', 'name': HEY_CATEGORIES['22']}
        # 21. Vanadates
        return {'id': '21', 'name': HEY_CATEGORIES['21']}
    
    # 3, 4, 5, 6. Sulfides and related
    if 'S' in element_list:
        # 4. Oxysulphides
        if 'O' in element_list and not contains('SO4', clean) and not contains('SO3', clean):
            return {'id': '4', 'name': HEY_CATEGORIES['4']}
        
        # 5, 6. Sulphosalts
        if any(el in ['As', 'Sb', 'Bi'] for el in element_list) and any(el not in ['S', 'Se', 'Te', 'As', 'Sb', 'Bi', 'O', 'H'] for el in element_list):
            if any(el in ['Sn', 'Ge', 'V'] for el in element_list):
                return {'id': '6', 'name': HEY_CATEGORIES['6']}
            return {'id': '5', 'name': HEY_CATEGORIES['5']}
        
        # 3. Sulphides, Selenides, Tellurides, Arsenides and Bismuthides
        if not contains('SO4', clean) and not contains('SO3', clean):
            return {'id': '3', 'name': HEY_CATEGORIES['3']}
    
    # 7. Oxides and hydroxides
    if (('O' in element_list and len(element_list) <= 3) or 
        (contains('OH', clean) and not contains('BO', clean) and 
         not contains('SiO', clean) and not contains('PO', clean) and 
         not contains('AsO', clean) and not contains('VO', clean))):
        return {'id': '7', 'name': HEY_CATEGORIES['7']}
    
    # 8. Halides
    if any(el in ['F', 'Cl', 'Br', 'I'] for el in element_list) and 'O' not in element_list and 'S' not in element_list and not contains('BO', clean) and not contains('SiO', clean):
        return {'id': '8', 'name': HEY_CATEGORIES['8']}
    
    # 9, 10. Borates
    if 'B' in element_list and 'O' in element_list:
        if any(el in ['F', 'Cl', 'Br', 'I', 'S', 'P', 'As', 'V'] for el in element_list) or contains('OH', clean):
            return {'id': '10', 'name': HEY_CATEGORIES['10']}
        return {'id': '9', 'name': HEY_CATEGORIES['9']}
    
    # 18. Niobates and Tantalates
    if 'Nb' in element_list or 'Ta' in element_list:
        return {'id': '18', 'name': HEY_CATEGORIES['18']}
    
    # 24. Antimonates and Antimonites
    if ('Sb' in element_list and 'O' in element_list) or contains('SbO3', clean) or contains('SbO4', clean):
        return {'id': '24', 'name': HEY_CATEGORIES['24']}
    
    # 27. Sulphites, Chromates, Molybdates and Tungstates
    if contains('SO3', clean) or contains('CrO4', clean) or 'Mo' in element_list or 'W' in element_list:
        return {'id': '27', 'name': HEY_CATEGORIES['27']}
    
    # 28. Selenites, Selenates, Tellurites, and Tellurates
    if ('Se' in element_list or 'Te' in element_list) and 'O' in element_list:
        return {'id': '28', 'name': HEY_CATEGORIES['28']}
    
    # 29. Iodates
    if contains('IO3', clean) or contains('IO4', clean):
        return {'id': '29', 'name': HEY_CATEGORIES['29']}
    
    # 31. Oxalates, Citrates, Mellitates and Acetates
    if contains('C2O4', clean) or contains('acetate', clean) or contains('citrate', clean) or contains('mellitate', clean):
        return {'id': '31', 'name': HEY_CATEGORIES['31']}
    
    # 32. Hydrocarbons, Resins and other Organic Compounds
    if (('C' in element_list and 'H' in element_list and 'O' not in element_list) or
        contains('C31H32', clean) or contains('C[0-9]+H[0-9]+', clean)):
        return {'id': '32', 'name': HEY_CATEGORIES['32']}
    
    # 1. Elements and Alloys (including the arsenides, antimonides and bismuthides of Cu, Ag and Au)
    if len(element_list) == 1 or (len(element_list) == 2 and all(el in ['Cu', 'Ag', 'Au', 'Fe', 'Ni', 'Co', 'Pt', 'Pd', 'Ir', 'Os', 'Ru', 'Rh'] for el in element_list)):
        return {'id': '1', 'name': HEY_CATEGORIES['1']}
    
    # Special case for Cu, Ag, Au with As, Sb, Bi
    if any(el in ['Cu', 'Ag', 'Au'] for el in element_list) and any(el in ['As', 'Sb', 'Bi'] for el in element_list) and 'S' not in element_list:
        return {'id': '1', 'name': HEY_CATEGORIES['1']}
    
    # 2. Carbides, Nitrides, Silicides and Phosphides
    if (('C' in element_list and 'O' not in element_list and 'H' not in element_list) or
        ('N' in element_list and 'O' not in element_list and 'H' not in element_list) or
        ('Si' in element_list and 'O' not in element_list) or
        ('P' in element_list and 'O' not in element_list)):
        return {'id': '2', 'name': HEY_CATEGORIES['2']}
    
    # Default case
    return default_classification

def add_hey_classification(input_file, output_file):
    """
    Add Hey's classification to a mineral dataset
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str
        Path to the output CSV file
    """
    print(f"Reading the input file: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Check if required columns exist
    required_columns = ['Mineral Name', 'RRUFF Chemistry (concise)', 'Chemistry Elements']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    print(f"Classifying {len(df)} minerals...")
    
    # Add Hey's classification columns
    df['Hey Classification ID'] = '0'
    df['Hey Classification Name'] = 'Unclassified'
    
    # Track statistics
    classifications = []
    
    # Classify each mineral
    for i, row in df.iterrows():
        mineral_name = row['Mineral Name']
        chemistry = row['RRUFF Chemistry (concise)']
        elements = row['Chemistry Elements']
        
        # Skip if missing data
        if pd.isna(chemistry) or pd.isna(elements):
            continue
        
        # Classify the mineral
        classification = classify_mineral(chemistry, elements)
        
        # Update the dataframe
        df.at[i, 'Hey Classification ID'] = classification['id']
        df.at[i, 'Hey Classification Name'] = classification['name']
        
        # Track the classification
        classifications.append(classification['id'])
        
        # Log progress
        if (i + 1) % 1000 == 0 or i == len(df) - 1:
            print(f"Processed {i + 1}/{len(df)} minerals ({(i + 1) / len(df) * 100:.1f}%)")
    
    # Save the updated dataset
    print(f"Saving the updated dataset to: {output_file}")
    df.to_csv(output_file, index=False)
    
    # Display statistics
    classification_counts = Counter(classifications)
    total_classified = sum(classification_counts.values())
    
    print("\nClassification Statistics:")
    print(f"Total minerals: {len(df)}")
    print(f"Classified: {total_classified} ({total_classified / len(df) * 100:.1f}%)")
    print(f"Unclassified: {len(df) - total_classified} ({(len(df) - total_classified) / len(df) * 100:.1f}%)")
    
    print("\nBreakdown by Hey's Classification:")
    for id, count in sorted(classification_counts.items(), key=lambda x: int(x[0])):
        if id == '0':
            continue
        print(f"{id}: {HEY_CATEGORIES[id]}: {count} minerals ({count / len(df) * 100:.1f}%)")
    
    return df

def visualize_hey_classification(df, output_dir='.'):
    """
    Create visualizations of Hey's classification distribution
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with Hey's classification columns
    output_dir : str
        Directory to save the visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Count classifications
    classification_counts = Counter(df['Hey Classification ID'])
    
    # Remove unclassified
    if '0' in classification_counts:
        del classification_counts['0']
    
    # Sort by count
    sorted_counts = sorted(classification_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare data for plotting
    categories = [f"{id}: {HEY_CATEGORIES[id][:20]}..." for id, _ in sorted_counts[:10]]
    counts = [count for _, count in sorted_counts[:10]]
    
    # Bar chart
    plt.figure(figsize=(12, 8))
    plt.barh(categories, counts)
    plt.title('Top 10 Hey\'s Classification Categories')
    plt.xlabel('Number of Minerals')
    plt.tight_layout()
    bar_chart_path = os.path.join(output_dir, 'hey_classification_top10.png')
    plt.savefig(bar_chart_path)
    plt.close()
    
    # Pie chart
    plt.figure(figsize=(12, 12))
    labels = [f"{id}: {HEY_CATEGORIES[id][:15]}..." for id, _ in sorted_counts[:8]]
    labels.append('Other')
    sizes = [count for _, count in sorted_counts[:8]]
    sizes.append(sum(count for _, count in sorted_counts[8:]))
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Distribution of Hey\'s Classification Categories')
    pie_chart_path = os.path.join(output_dir, 'hey_classification_pie.png')
    plt.savefig(pie_chart_path)
    plt.close()
    
    print(f"\nVisualizations saved to:\n- {bar_chart_path}\n- {pie_chart_path}")
    
    return bar_chart_path, pie_chart_path

def main():
    """
    Main function to run the Hey's classification process
    """
    # Default file paths
    input_file = "RRUFF_Export_20250427_214850.csv"
    output_file = "RRUFF_Export_with_Hey_Classification.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        input_file = input("Enter the path to the input CSV file: ")
        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
            return
    
    # Add Hey's classification
    df = add_hey_classification(input_file, output_file)
    
    if df is not None:
        # Visualize the results
        visualize_hey_classification(df)
        
        print("\nProcess completed successfully!")
        print(f"Updated dataset saved to: {output_file}")
    else:
        print("Error processing the dataset.")

if __name__ == "__main__":
    main()
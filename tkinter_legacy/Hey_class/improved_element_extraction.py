import re
import pandas as pd
from typing import List, Set

def extract_elements_from_formula(formula: str) -> List[str]:
    """
    Extract all chemical elements from a formula string.
    
    This function handles various formula formats including:
    - Subscripts and superscripts (both with and without underscores/carets)
    - Parentheses and brackets
    - Hydration notation (·, •)
    - Charge notation
    
    Parameters:
    -----------
    formula : str
        Chemical formula string
        
    Returns:
    --------
    List[str]
        Sorted list of unique element symbols
    """
    if not isinstance(formula, str) or not formula.strip():
        return []
    
    # Clean the formula
    clean_formula = formula.strip()
    
    # Remove charge indicators and formatting
    clean_formula = re.sub(r'\^[0-9+\-]+\^', '', clean_formula)  # Remove ^2+^, ^3-^, etc.
    clean_formula = re.sub(r'_[0-9]+_', '', clean_formula)       # Remove _2_, _3_, etc.
    clean_formula = re.sub(r'[<>]', '', clean_formula)           # Remove < >
    clean_formula = clean_formula.replace('·', ' ')              # Replace hydration dots
    clean_formula = clean_formula.replace('•', ' ')              # Replace bullet dots
    
    # Pattern to match element symbols
    # Element symbols: Capital letter followed by optional lowercase letter
    element_pattern = r'[A-Z][a-z]?'
    
    # Find all element symbols
    elements = re.findall(element_pattern, clean_formula)
    
    # Remove common non-element patterns that might be captured
    non_elements = {'OH', 'CO', 'SO', 'PO', 'NO', 'BO', 'As', 'Sb', 'Bi'}  # Keep As, Sb, Bi as they are elements
    non_elements = {'OH', 'CO', 'SO', 'PO', 'NO', 'BO'}  # Actually, let's be more careful
    
    # Filter out obvious non-elements and validate
    valid_elements = []
    for element in elements:
        # Skip if it's a known non-element pattern
        if element in non_elements:
            continue
        
        # Add to valid elements
        valid_elements.append(element)
    
    # Handle special cases where we need to extract from compound groups
    # CO3 -> C, O
    if 'CO3' in clean_formula or 'CO_3' in clean_formula or '(CO3)' in clean_formula:
        valid_elements.extend(['C', 'O'])
    
    # SO4 -> S, O  
    if 'SO4' in clean_formula or 'SO_4' in clean_formula or '(SO4)' in clean_formula:
        valid_elements.extend(['S', 'O'])
    
    # PO4 -> P, O
    if 'PO4' in clean_formula or 'PO_4' in clean_formula or '(PO4)' in clean_formula:
        valid_elements.extend(['P', 'O'])
    
    # NO3 -> N, O
    if 'NO3' in clean_formula or 'NO_3' in clean_formula or '(NO3)' in clean_formula:
        valid_elements.extend(['N', 'O'])
    
    # BO3, BO4 -> B, O
    if 'BO3' in clean_formula or 'BO_3' in clean_formula or 'BO4' in clean_formula or 'BO_4' in clean_formula:
        valid_elements.extend(['B', 'O'])
    
    # AsO3, AsO4 -> As, O
    if 'AsO3' in clean_formula or 'AsO_3' in clean_formula or 'AsO4' in clean_formula or 'AsO_4' in clean_formula:
        valid_elements.extend(['As', 'O'])
    
    # OH -> O, H
    if 'OH' in clean_formula:
        valid_elements.extend(['O', 'H'])
    
    # H2O -> H, O
    if 'H2O' in clean_formula or 'H_2_O' in clean_formula:
        valid_elements.extend(['H', 'O'])
    
    # Remove duplicates and sort
    unique_elements = sorted(list(set(valid_elements)))
    
    return unique_elements

def validate_element_list(elements: List[str]) -> List[str]:
    """
    Validate and clean a list of element symbols.
    
    Parameters:
    -----------
    elements : List[str]
        List of potential element symbols
        
    Returns:
    --------
    List[str]
        Validated list of element symbols
    """
    # Common element symbols (not exhaustive, but covers most minerals)
    valid_elements = {
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
        'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    }
    
    return [el for el in elements if el in valid_elements]

def test_element_extraction():
    """Test the element extraction function with various formula formats."""
    test_cases = [
        ("NaPb^2+^_2_(CO_3_)_2_(OH)", ["C", "H", "Na", "O", "Pb"]),
        ("Ni^2+^C_31_H_32_N_4_", ["C", "H", "N", "Ni"]),
        ("CaCO3", ["C", "Ca", "O"]),
        ("CaSO4·2H2O", ["Ca", "H", "O", "S"]),
        ("KAlSi3O8", ["Al", "K", "O", "Si"]),
        ("Fe2O3", ["Fe", "O"]),
        ("Mg3Si4O10(OH)2", ["H", "Mg", "O", "Si"]),
        ("Ca5(PO4)3(OH)", ["Ca", "H", "O", "P"]),
        ("SiO2", ["O", "Si"]),
        ("NaCl", ["Cl", "Na"]),
    ]
    
    print("Testing Element Extraction:")
    print("-" * 60)
    
    for formula, expected in test_cases:
        result = extract_elements_from_formula(formula)
        status = "✓" if result == expected else "✗"
        print(f"{status} {formula}")
        print(f"   Expected: {expected}")
        print(f"   Got:      {result}")
        if result != expected:
            missing = set(expected) - set(result)
            extra = set(result) - set(expected)
            if missing:
                print(f"   Missing:  {sorted(list(missing))}")
            if extra:
                print(f"   Extra:    {sorted(list(extra))}")
        print()

def update_chemistry_elements_column(input_csv: str, output_csv: str):
    """
    Update the Chemistry Elements column in the CSV with improved element extraction.
    
    Parameters:
    -----------
    input_csv : str
        Path to input CSV file
    output_csv : str
        Path to output CSV file
    """
    # Read the CSV
    df = pd.read_csv(input_csv)
    
    print(f"Processing {len(df)} minerals...")
    
    # Extract elements from RRUFF Chemistry (concise)
    new_elements = []
    for _, row in df.iterrows():
        chemistry = row.get('RRUFF Chemistry (concise)', '')
        elements = extract_elements_from_formula(chemistry)
        new_elements.append(', '.join(elements))
    
    # Update the Chemistry Elements column
    df['Chemistry Elements'] = new_elements
    
    # Save the updated CSV
    df.to_csv(output_csv, index=False)
    
    print(f"Updated CSV saved to: {output_csv}")
    
    # Show some examples of changes
    print("\nSample improvements:")
    print("-" * 40)
    for i in range(min(5, len(df))):
        print(f"Mineral: {df.iloc[i]['Mineral Name']}")
        print(f"Chemistry: {df.iloc[i]['RRUFF Chemistry (concise)']}")
        print(f"Elements: {df.iloc[i]['Chemistry Elements']}")
        print()

if __name__ == "__main__":
    test_element_extraction() 
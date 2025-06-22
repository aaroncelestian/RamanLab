import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from typing import Dict, List, Tuple, Optional

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

class ImprovedHeyClassifier:
    """
    Improved Hey Classification system with better accuracy and comprehensive logic.
    """
    
    def __init__(self):
        self.native_elements = {
            'metals': ['Cu', 'Ag', 'Au', 'Fe', 'Ni', 'Co', 'Pt', 'Pd', 'Ir', 'Os', 'Ru', 'Rh', 'Pb', 'Zn', 'Sn', 'Hg'],
            'semimetals': ['As', 'Sb', 'Bi', 'Te', 'Se'],
            'nonmetals': ['C', 'S']
        }
        
        self.halides = ['F', 'Cl', 'Br', 'I']
        self.chalcogens = ['S', 'Se', 'Te']
        self.pnictogens = ['As', 'Sb', 'Bi']
        
        # Common anion groups with their patterns (more comprehensive)
        self.anion_patterns = {
            'sulfate': [r'SO4', r'SO_4_', r'\(SO4\)', r'\(SO_4_\)', r'S\^6\+\^O_4_', r'S\^6\+\^O4'],
            'carbonate': [r'CO3', r'CO_3_', r'\(CO3\)', r'\(CO_3_\)', r'C\^?\^?O_3_', r'CO_3'],
            'nitrate': [r'NO3', r'NO_3_', r'\(NO3\)', r'\(NO_3_\)'],
            'phosphate': [r'PO4', r'PO_4_', r'\(PO4\)', r'\(PO_4_\)', r'P\^?\^?O_4_'],
            'arsenate': [r'AsO4', r'AsO_4_', r'\(AsO4\)', r'\(AsO_4_\)', r'As\^5\+\^O_4_', r'As\^5\+\^O4'],
            'arsenite': [r'AsO3', r'AsO_3_', r'\(AsO3\)', r'\(AsO_3_\)'],
            'vanadate': [r'VO4', r'VO_4_', r'\(VO4\)', r'\(VO_4_\)'],
            'chromate': [r'CrO4', r'CrO_4_', r'\(CrO4\)', r'\(CrO_4_\)'],
            'molybdate': [r'MoO4', r'MoO_4_', r'\(MoO4\)', r'\(MoO_4_\)'],
            'tungstate': [r'WO4', r'WO_4_', r'\(WO4\)', r'\(WO_4_\)'],
            'sulfite': [r'SO3', r'SO_3_', r'\(SO3\)', r'\(SO_3_\)', r'S\^4\+\^O_2_'],
            'iodate': [r'IO3', r'IO_3_', r'IO4', r'IO_4_'],
            'borate': [r'BO3', r'BO_3_', r'BO4', r'BO_4_', r'B2O3', r'B_2_O_3_'],
        }
        
        # Silicate structural patterns
        self.silicate_patterns = {
            'framework': [r'SiO2', r'SiO_2', r'Si\d*O\d*', r'AlSi\d*O\d*'],
            'chain': [r'Si2O6', r'Si_2_O_6', r'Si4O11', r'Si_4_O_11'],
            'sheet': [r'Si2O5', r'Si_2_O_5', r'Si4O10', r'Si_4_O_10'],
            'ring': [r'Si3O9', r'Si_3_O_9', r'Si6O18', r'Si_6_O_18'],
            'isolated': [r'SiO4', r'SiO_4'],
            'double': [r'Si2O7', r'Si_2_O_7']
        }
    
    def clean_chemistry(self, chemistry: str) -> str:
        """Clean up chemical formulas by removing formatting artifacts."""
        if not isinstance(chemistry, str):
            return ""
        
        # Remove superscripts and subscripts
        chemistry = re.sub(r'\^[0-9+\-]+\^', '', chemistry)
        chemistry = re.sub(r'_[0-9]+_', '', chemistry)
        
        # Remove hydration dots and special characters
        chemistry = chemistry.replace('·', '').replace('•', '')
        chemistry = re.sub(r'[<>]', '', chemistry)
        
        return chemistry.strip()
    
    def parse_elements(self, elements_str: str) -> List[str]:
        """Parse comma-separated elements string into a clean list."""
        if not isinstance(elements_str, str):
            return []
        
        elements = [el.strip() for el in elements_str.split(',')]
        return [el for el in elements if el and el != 'nan']
    
    def contains_anion_group(self, chemistry: str, anion_type: str) -> bool:
        """Check if chemistry contains specific anion group."""
        if anion_type not in self.anion_patterns:
            return False
        
        patterns = self.anion_patterns[anion_type]
        for pattern in patterns:
            if re.search(pattern, chemistry, re.IGNORECASE):
                return True
        return False
    
    def contains_silicate(self, chemistry: str) -> Tuple[bool, str]:
        """Check if chemistry contains silicate and determine type."""
        for silicate_type, patterns in self.silicate_patterns.items():
            for pattern in patterns:
                if re.search(pattern, chemistry, re.IGNORECASE):
                    return True, silicate_type
        
        # General silicate check
        if re.search(r'Si.*O', chemistry, re.IGNORECASE):
            return True, 'general'
        
        return False, ''
    
    def analyze_aluminum_role(self, chemistry: str, elements: List[str]) -> str:
        """
        Determine if aluminum is part of the silicate framework or just a cation.
        Returns: 'framework', 'cation', or 'none'
        """
        if 'Al' not in elements:
            return 'none'
        
        # Check for aluminum-silicon framework patterns
        framework_patterns = [
            r'Al\d*Si\d*O\d*',  # AlSiO patterns
            r'Si\d*Al\d*O\d*',  # SiAlO patterns
            r'Al2Si\d*O\d*',    # Al2Si patterns
            r'Si\d*Al2O\d*',    # SiAl2 patterns
        ]
        
        for pattern in framework_patterns:
            if re.search(pattern, chemistry, re.IGNORECASE):
                return 'framework'
        
        # If Al is present but not in framework patterns, it's likely a cation
        return 'cation'
    
    def has_other_anions(self, elements: List[str], chemistry: str) -> bool:
        """Check for presence of other anions (F, Cl, Br, I, S, P, As, V)."""
        other_anion_elements = set(['F', 'Cl', 'Br', 'I', 'S', 'P', 'As', 'V'])
        
        # Check elements (but exclude S if it's part of sulfate)
        if any(el in other_anion_elements for el in elements):
            # If S is present, check if it's part of sulfate
            if 'S' in elements and self.contains_anion_group(chemistry, 'sulfate'):
                # Remove S from consideration and check others
                other_elements = [el for el in elements if el in other_anion_elements and el != 'S']
                if other_elements:
                    return True
            else:
                return True
        
        # Check for OH groups (but be more selective for silicates)
        if re.search(r'OH', chemistry, re.IGNORECASE):
            # For silicates, OH is common and shouldn't always trigger "other anions"
            # Only trigger if there are significant amounts of OH or other indicators
            if re.search(r'\(OH\)_?[2-9]', chemistry) or re.search(r'OH.*OH', chemistry):
                return True
        
        return False
    
    def classify_silicate(self, chemistry: str, elements: List[str]) -> Dict[str, str]:
        """Classify silicate minerals according to Hey's system."""
        al_role = self.analyze_aluminum_role(chemistry, elements)
        has_other = self.has_other_anions(elements, chemistry)
        
        # Category 17: Silicates with other anions
        if has_other:
            return {'id': '17', 'name': HEY_CATEGORIES['17']}
        
        # Category 15: Aluminum silicates (Al in framework)
        if al_role == 'framework':
            return {'id': '15', 'name': HEY_CATEGORIES['15']}
        
        # Category 16: Silicates with Al and other metals (Al as cation)
        if al_role == 'cation':
            # Check if there are other metals besides Al
            metals = [el for el in elements if el not in ['Si', 'O', 'H', 'Al']]
            if metals:
                return {'id': '16', 'name': HEY_CATEGORIES['16']}
            else:
                return {'id': '15', 'name': HEY_CATEGORIES['15']}
        
        # Category 14: Silicates not containing aluminum
        return {'id': '14', 'name': HEY_CATEGORIES['14']}
    
    def classify_sulfur_bearing(self, chemistry: str, elements: List[str]) -> Optional[Dict[str, str]]:
        """Classify sulfur-bearing minerals."""
        if 'S' not in elements:
            return None
        
        # Check for sulfates first
        if self.contains_anion_group(chemistry, 'sulfate'):
            if any(el in self.halides for el in elements):
                return {'id': '26', 'name': HEY_CATEGORIES['26']}
            return {'id': '25', 'name': HEY_CATEGORIES['25']}
        
        # Check for sulfites
        if self.contains_anion_group(chemistry, 'sulfite'):
            return {'id': '27', 'name': HEY_CATEGORIES['27']}
        
        # Check for oxysulfides (S + O but not SO4/SO3)
        if 'O' in elements and not self.contains_anion_group(chemistry, 'sulfate') and not self.contains_anion_group(chemistry, 'sulfite'):
            return {'id': '4', 'name': HEY_CATEGORIES['4']}
        
        # Check for sulphosalts
        if any(el in self.pnictogens for el in elements):
            # Check for specific sulphosalt types
            if any(el in ['Sn', 'Ge', 'V'] for el in elements) or any(el in self.halides for el in elements):
                return {'id': '6', 'name': HEY_CATEGORIES['6']}
            return {'id': '5', 'name': HEY_CATEGORIES['5']}
        
        # General sulfides
        return {'id': '3', 'name': HEY_CATEGORIES['3']}
    
    def classify_mineral(self, chemistry: str, elements_str: str) -> Dict[str, str]:
        """
        Main classification function with improved logic.
        
        Parameters:
        -----------
        chemistry : str
            Chemical formula
        elements_str : str
            Comma-separated list of elements
        
        Returns:
        --------
        dict
            Dictionary with 'id' and 'name' of the Hey classification
        """
        # Default classification
        default = {'id': '0', 'name': 'Unclassified'}
        
        # Handle empty inputs
        if not isinstance(chemistry, str) or not isinstance(elements_str, str):
            return default
        
        # Clean and parse inputs
        clean_chem = self.clean_chemistry(chemistry)
        elements = self.parse_elements(elements_str)
        
        if not elements:
            return default
        
        # 1. Elements and Alloys
        if len(elements) == 1:
            return {'id': '1', 'name': HEY_CATEGORIES['1']}
        
        # Special case for Cu, Ag, Au with As, Sb, Bi (but not sulfides)
        if (any(el in ['Cu', 'Ag', 'Au'] for el in elements) and 
            any(el in self.pnictogens for el in elements) and 
            'S' not in elements):
            return {'id': '1', 'name': HEY_CATEGORIES['1']}
        
        # Native alloys
        if all(el in self.native_elements['metals'] for el in elements):
            return {'id': '1', 'name': HEY_CATEGORIES['1']}
        
        # 2. Carbides, Nitrides, Silicides and Phosphides (no oxygen)
        if 'O' not in elements and 'H' not in elements:
            if 'C' in elements:
                return {'id': '2', 'name': HEY_CATEGORIES['2']}
            if 'N' in elements:
                return {'id': '2', 'name': HEY_CATEGORIES['2']}
            if 'Si' in elements and not any(el in self.chalcogens for el in elements):
                return {'id': '2', 'name': HEY_CATEGORIES['2']}
            if 'P' in elements and not any(el in self.chalcogens for el in elements):
                return {'id': '2', 'name': HEY_CATEGORIES['2']}
        
        # 30. Thiocyanates
        if ('S' in elements and 'C' in elements and 'N' in elements and 
            re.search(r'SCN|CNS', chemistry, re.IGNORECASE)):
            return {'id': '30', 'name': HEY_CATEGORIES['30']}
        
        # 32. Organic compounds (be more specific to avoid false positives)
        if ('C' in elements and 'H' in elements):
            # Check if it's truly organic (not just CO3 + OH)
            # Look for C-H bonds or organic patterns
            if (re.search(r'C\d*H\d+', chemistry) and 
                not self.contains_anion_group(chemistry, 'carbonate') and
                not self.contains_anion_group(chemistry, 'sulfate') and
                not self.contains_anion_group(chemistry, 'phosphate')):
                return {'id': '32', 'name': HEY_CATEGORIES['32']}
        
        # 31. Organic salts
        if (re.search(r'oxalate|citrate|acetate|mellitate', chemistry, re.IGNORECASE) or
            self.contains_anion_group(chemistry, 'oxalate')):
            return {'id': '31', 'name': HEY_CATEGORIES['31']}
        
        # Priority order: Check anion groups first, then other categories
        # Use original chemistry for pattern matching, cleaned for other checks
        
        # 13. Nitrates
        if self.contains_anion_group(chemistry, 'nitrate'):
            return {'id': '13', 'name': HEY_CATEGORIES['13']}
        
        # 11, 12. Carbonates
        if self.contains_anion_group(chemistry, 'carbonate'):
            if (any(el in self.halides for el in elements) or 
                re.search(r'OH', chemistry) or 
                self.contains_anion_group(chemistry, 'sulfate') or
                self.contains_anion_group(chemistry, 'phosphate')):
                return {'id': '12', 'name': HEY_CATEGORIES['12']}
            return {'id': '11', 'name': HEY_CATEGORIES['11']}
        
        # Sulfur-bearing minerals (categories 3-6, 25-27)
        sulfur_class = self.classify_sulfur_bearing(chemistry, elements)
        if sulfur_class:
            return sulfur_class
        
        # 19, 20, 21, 22. Phosphates, Arsenates, Vanadates
        mixed_anions = (any(el in self.halides for el in elements) or 
                       re.search(r'OH', chemistry) or
                       self.contains_anion_group(chemistry, 'sulfate') or
                       self.contains_anion_group(chemistry, 'carbonate'))
        
        if self.contains_anion_group(chemistry, 'phosphate'):
            if mixed_anions:
                return {'id': '22', 'name': HEY_CATEGORIES['22']}
            return {'id': '19', 'name': HEY_CATEGORIES['19']}
        
        if self.contains_anion_group(chemistry, 'arsenate'):
            if mixed_anions:
                return {'id': '22', 'name': HEY_CATEGORIES['22']}
            return {'id': '20', 'name': HEY_CATEGORIES['20']}
        
        if self.contains_anion_group(chemistry, 'vanadate'):
            if mixed_anions:
                return {'id': '22', 'name': HEY_CATEGORIES['22']}
            return {'id': '21', 'name': HEY_CATEGORIES['21']}
        
        # 23. Arsenites
        if self.contains_anion_group(chemistry, 'arsenite'):
            return {'id': '23', 'name': HEY_CATEGORIES['23']}
        
        # 8. Halides (only halides, no oxygen or sulfur)
        if (any(el in self.halides for el in elements) and 
            'O' not in elements and 'S' not in elements):
            return {'id': '8', 'name': HEY_CATEGORIES['8']}
        
        # 14-17. Silicates (check before oxides)
        is_silicate, silicate_type = self.contains_silicate(chemistry)
        if is_silicate:
            return self.classify_silicate(chemistry, elements)
        
        # 7. Oxides and Hydroxides
        if ('O' in elements and len(elements) <= 3) or re.search(r'OH', chemistry):
            # Exclude if it contains anion groups
            if not any(self.contains_anion_group(chemistry, anion) for anion in self.anion_patterns.keys()):
                return {'id': '7', 'name': HEY_CATEGORIES['7']}
        
        # 9, 10. Borates
        if 'B' in elements and 'O' in elements:
            if (any(el in self.halides for el in elements) or 
                re.search(r'OH', chemistry) or
                any(el in ['S', 'P', 'As', 'V'] for el in elements)):
                return {'id': '10', 'name': HEY_CATEGORIES['10']}
            return {'id': '9', 'name': HEY_CATEGORIES['9']}
        
        # 18. Niobates and Tantalates
        if 'Nb' in elements or 'Ta' in elements:
            return {'id': '18', 'name': HEY_CATEGORIES['18']}
        
        # 24. Antimonates and Antimonites
        if ('Sb' in elements and 'O' in elements):
            return {'id': '24', 'name': HEY_CATEGORIES['24']}
        
        # 27. Chromates, Molybdates, Tungstates
        if ('Cr' in elements or 'Mo' in elements or 'W' in elements) and 'O' in elements:
            return {'id': '27', 'name': HEY_CATEGORIES['27']}
        
        # 28. Selenites, Selenates, Tellurites, Tellurates
        if ('Se' in elements or 'Te' in elements) and 'O' in elements:
            return {'id': '28', 'name': HEY_CATEGORIES['28']}
        
        # 29. Iodates
        if self.contains_anion_group(chemistry, 'iodate'):
            return {'id': '29', 'name': HEY_CATEGORIES['29']}
        
        return default

def test_classifier():
    """Test the improved classifier with some examples."""
    classifier = ImprovedHeyClassifier()
    
    test_cases = [
        ("SiO2", "Si, O", "14"),  # Quartz - silicate without Al
        ("KAlSi3O8", "K, Al, Si, O", "15"),  # Feldspar - Al silicate
        ("CaCO3", "Ca, C, O", "11"),  # Calcite - carbonate
        ("CaSO4·2H2O", "Ca, S, O, H", "25"),  # Gypsum - sulfate
        ("FeS2", "Fe, S", "3"),  # Pyrite - sulfide
        ("NaCl", "Na, Cl", "8"),  # Halite - halide
        ("Fe2O3", "Fe, O", "7"),  # Hematite - oxide
        ("Mg3Si4O10(OH)2", "Mg, Si, O, H", "17"),  # Talc - silicate with OH
        ("Ca5(PO4)3(OH)", "Ca, P, O, H", "22"),  # Apatite - phosphate with OH
        ("Au", "Au", "1"),  # Gold - native element
    ]
    
    print("Testing Improved Hey Classifier:")
    print("-" * 50)
    
    for chemistry, elements, expected_id in test_cases:
        result = classifier.classify_mineral(chemistry, elements)
        status = "✓" if result['id'] == expected_id else "✗"
        print(f"{status} {chemistry} ({elements}) -> {result['id']}: {result['name'][:50]}...")
    
    print("-" * 50)

def create_validation_report(input_csv: str, output_file: str = "validation_report.txt"):
    """
    Create a validation report comparing old vs new classifications.
    """
    classifier = ImprovedHeyClassifier()
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Apply new classification
    new_classifications = []
    for _, row in df.iterrows():
        chemistry = row.get('RRUFF Chemistry (concise)', '')
        elements = row.get('Chemistry Elements', '')
        result = classifier.classify_mineral(chemistry, elements)
        new_classifications.append(int(result['id']))  # Convert to int to match CSV format
    
    df['New Hey Classification ID'] = new_classifications
    
    # Compare with existing classifications
    if 'Hey Classification ID' in df.columns:
        df['Classification Changed'] = df['Hey Classification ID'] != df['New Hey Classification ID']
        changes = df[df['Classification Changed']]
        
        print(f"Validation Report:")
        print(f"Total minerals: {len(df)}")
        print(f"Classifications changed: {len(changes)}")
        print(f"Accuracy: {((len(df) - len(changes)) / len(df) * 100):.1f}%")
        
        # Save detailed report
        with open(output_file, 'w') as f:
            f.write("Hey Classification Validation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total minerals analyzed: {len(df)}\n")
            f.write(f"Classifications changed: {len(changes)}\n")
            f.write(f"Accuracy: {((len(df) - len(changes)) / len(df) * 100):.1f}%\n\n")
            
            if len(changes) > 0:
                f.write("Changed Classifications:\n")
                f.write("-" * 30 + "\n")
                for _, row in changes.head(20).iterrows():
                    f.write(f"Mineral: {row['Mineral Name']}\n")
                    f.write(f"Chemistry: {row['RRUFF Chemistry (concise)']}\n")
                    f.write(f"Elements: {row['Chemistry Elements']}\n")
                    f.write(f"Old: {row['Hey Classification ID']} -> New: {row['New Hey Classification ID']}\n")
                    f.write("\n")
        
        print(f"Detailed report saved to: {output_file}")
        return changes
    else:
        print("No existing Hey Classification found for comparison")
        return None

if __name__ == "__main__":
    test_classifier() 
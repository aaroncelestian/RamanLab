# Numpy Array Safety Guide - Quick Reference

## 🚨 **The #1 Rule: NEVER use direct boolean evaluation on numpy arrays**

```python
# ❌ NEVER DO THIS - Will cause ambiguous truth value error
if not numpy_array:
if numpy_array:
if numpy_array and other_condition:
```

## ✅ **Safe Patterns (Copy & Paste Ready)**

### **Check if array is empty/None:**
```python
if array is None or len(array) == 0:
    # Handle empty case
```

### **Check if array has data:**
```python
if array is not None and len(array) > 0:
    # Handle non-empty case
```

### **Full validation (safest for mixed types):**
```python
if (array is None or 
    not hasattr(array, '__len__') or 
    len(array) == 0):
    # Handle invalid/empty case
```

### **Helper function (recommended):**
```python
def is_valid_array(self, arr):
    """Check if array is valid and non-empty."""
    return (arr is not None and 
            hasattr(arr, '__len__') and 
            len(arr) > 0)

# Usage:
if self.is_valid_array(self.peaks):
    # Safe to use peaks
```

## 🔍 **Quick Debug Commands**

When you suspect ambiguous truth value errors:

```bash
# Search for dangerous patterns
grep -n "if not.*array" your_file.py
grep -n "if.*array:" your_file.py
grep -n "not.*peaks" your_file.py
grep -n "not.*params" your_file.py
```

## 💡 **Remember:**
- After `curve_fit()`, parameters become numpy arrays
- Always initialize: `self.peaks = np.array([])`
- Use explicit length checks instead of boolean evaluation
- Test with both empty and populated arrays

---
*Keep this guide handy when working with numpy arrays in boolean contexts!* 
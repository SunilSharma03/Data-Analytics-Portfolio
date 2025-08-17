#!/usr/bin/env python3
"""
Setup Verification Script

This script verifies that all required packages for the predictive modeling
project are installed and working correctly.
"""

import sys
import importlib
from typing import List, Dict, Tuple

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed and can be imported."""
    try:
        if import_name is None:
            import_name = package_name
        
        module = importlib.import_module(import_name)
        
        # Try to get version
        version = "unknown"
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'version'):
            version = module.version
        
        return True, version
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error importing: {str(e)}"

def main():
    """Main verification function."""
    print("🔍 Predictive Modeling Setup Verification")
    print("=" * 50)
    
    # Core packages to check
    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("tensorflow", "tensorflow"),
        ("prophet", "prophet"),
        ("statsmodels", "statsmodels"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("loguru", "loguru"),
    ]
    
    # Check each package
    results = []
    all_passed = True
    
    for package_name, import_name in packages:
        is_available, version_or_error = check_package(package_name, import_name)
        results.append((package_name, is_available, version_or_error))
        
        if is_available:
            print(f"✅ {package_name}: {version_or_error}")
        else:
            print(f"❌ {package_name}: {version_or_error}")
            all_passed = False
    
    print("\n" + "=" * 50)
    
    # Summary
    if all_passed:
        print("🎉 All checks passed! Your setup is ready for predictive modeling.")
    else:
        print("⚠️ Some packages are missing. Install with: pip install -r requirements/requirements.txt")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

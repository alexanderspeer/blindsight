"""
Configuration Validation Script
Verifies that all weights come from config and no hardcoded values exist
"""

import re
from pathlib import Path

def check_hardcoded_weights():
    """Check for hardcoded weight values in code files"""
    print("\nVALIDATING CONFIGURATION")
    
    files_to_check = [
        'v1_column.py',
        'v1_model.py',
        'neurons.py',
        'pipeline.py'
    ]
    
    # Patterns that indicate potential hardcoded weights
    suspicious_patterns = [
        (r'\bweight\s*=\s*[0-9]{2,4}\b', 'Direct weight assignment'),
        (r'\.weight\s*=\s*[0-9]{2,4}', 'Attribute weight assignment'),
        (r'\b(800|1200)\b', 'Old hardcoded values (800, 1200)'),
    ]
    
    issues_found = []
    
    for filename in files_to_check:
        filepath = Path(__file__).parent / filename
        if not filepath.exists():
            print(f"WARNING: File not found: {filename}")
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments and strings
            if line.strip().startswith('#'):
                continue
            if '"""' in line or "'''" in line:
                continue
                
            for pattern, description in suspicious_patterns:
                if re.search(pattern, line):
                    # Check if it's using config
                    if 'V1_ARCHITECTURE' in line or 'config' in line.lower():
                        continue
                    # Check if it's in a docstring
                    if 'Args:' in line or 'Returns:' in line or 'Example' in line:
                        continue
                    
                    issues_found.append({
                        'file': filename,
                        'line': line_num,
                        'content': line.strip(),
                        'issue': description
                    })
    
    if issues_found:
        print("\nWARNING: POTENTIAL ISSUES FOUND:\n")
        for issue in issues_found:
            print(f"  {issue['file']}:{issue['line']}")
            print(f"    Issue: {issue['issue']}")
            print(f"    Line: {issue['content'][:80]}")
            print()
        return False
    else:
        print("\nPASS: No hardcoded weights found!")
        return True


def verify_config_values():
    """Verify that config.py has the correct values"""
    print("\nVERIFYING CONFIG VALUES")
    
    from config import V1_ARCHITECTURE
    
    expected_weights = {
        'weight_L4_to_L23': (100.0, 150.0),
        'weight_L23_to_L5': (150.0, 250.0),
        'weight_L5_to_L6': (100.0, 200.0),
    }
    
    all_good = True
    
    for weight_name, (min_val, max_val) in expected_weights.items():
        actual_value = V1_ARCHITECTURE.get(weight_name)
        
        if actual_value is None:
            print(f"FAIL: {weight_name}: NOT FOUND in config!")
            all_good = False
        elif actual_value < min_val or actual_value > max_val:
            print(f"WARNING: {weight_name}: {actual_value} (expected range: {min_val}-{max_val})")
            all_good = False
        else:
            print(f"PASS: {weight_name}: {actual_value}")
    
    # Check L2/3 parameters
    print("\nL2/3 Parameters:")
    l23_params = {
        'L23_v_threshold': (-57.0, -52.0),
        'L23_tau_membrane': (20.0, 30.0),
        'L23_bias_current': (15.0, 35.0),
    }
    
    for param_name, (min_val, max_val) in l23_params.items():
        actual_value = V1_ARCHITECTURE.get(param_name)
        
        if actual_value is None:
            print(f"FAIL: {param_name}: NOT FOUND in config!")
            all_good = False
        elif actual_value < min_val or actual_value > max_val:
            print(f"WARNING: {param_name}: {actual_value} (expected range: {min_val}-{max_val})")
        else:
            print(f"PASS: {param_name}: {actual_value}")
    
    return all_good


def check_no_rollback_logic():
    """Verify no automatic weight adjustment or rollback logic exists"""
    print("\nCHECKING FOR ROLLBACK/ADJUSTMENT LOGIC")
    
    files_to_check = [
        'v1_column.py',
        'v1_model.py',
        'pipeline.py'
    ]
    
    dangerous_patterns = [
        r'rollback',
        r'adjust.*weight',
        r'if.*active.*:.*weight',
        r'rebuild.*column',
        r'weight.*-=',
        r'weight.*\+=',
    ]
    
    issues_found = []
    
    for filename in files_to_check:
        filepath = Path(__file__).parent / filename
        if not filepath.exists():
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith('#'):
                continue
                
            for pattern in dangerous_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues_found.append({
                        'file': filename,
                        'line': line_num,
                        'content': line.strip(),
                        'pattern': pattern
                    })
    
    if issues_found:
        print("\nWARNING: POTENTIAL ROLLBACK LOGIC FOUND:\n")
        for issue in issues_found:
            print(f"  {issue['file']}:{issue['line']}")
            print(f"    Pattern: {issue['pattern']}")
            print(f"    Line: {issue['content'][:80]}")
            print()
        return False
    else:
        print("\nPASS: No rollback or automatic adjustment logic found!")
        return True


def main():
    """Run all validation checks"""
    print("\nV1 MODEL CONFIGURATION VALIDATION")
    
    checks = [
        ("Hardcoded weights", check_hardcoded_weights),
        ("Config values", verify_config_values),
        ("Rollback logic", check_no_rollback_logic),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\nERROR: Error in {check_name}: {e}")
            results.append((check_name, False))
    
    # Print summary
    print("\nVALIDATION SUMMARY")
    
    all_passed = all(result for _, result in results)
    
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {check_name}")
    
    
    if all_passed:
        print("\nPASS: ALL CHECKS PASSED - Configuration is clean!")
        print("\nYou can now run: python test_static_image.py")
    else:
        print("\nWARNING: SOME CHECKS FAILED - Review issues above")
    
    print()
    return all_passed


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)


"""
Script to automatically fix common pylint issues in qsopt package
"""
import os
import re
from pathlib import Path

def fix_trailing_whitespace(content):
    """Remove trailing whitespace from lines."""
    lines = content.split('\n')
    fixed_lines = [line.rstrip() for line in lines]
    return '\n'.join(fixed_lines)

def ensure_final_newline(content):
    """Ensure file ends with a single newline."""
    if not content.endswith('\n'):
        return content + '\n'
    # Remove extra trailing newlines but keep one
    content = content.rstrip('\n') + '\n'
    return content

def fix_file(filepath):
    """Fix common issues in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Apply fixes
        content = fix_trailing_whitespace(content)
        content = ensure_final_newline(content)
        
        # Only write if changed
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    return False

def main():
    """Fix all Python files in src/qsopt."""
    src_dir = Path("src/qsopt")
    
    fixed_count = 0
    for py_file in src_dir.rglob("*.py"):
        if fix_file(py_file):
            print(f"Fixed: {py_file}")
            fixed_count += 1
    
    print(f"\nTotal files fixed: {fixed_count}")

if __name__ == "__main__":
    main()

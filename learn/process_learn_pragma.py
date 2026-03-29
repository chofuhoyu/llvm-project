#!/usr/bin/env python3
"""
Process #pragma learn print directives in C/C++ source files.
This script parses source files and inserts printf statements
and necessary #include directives.

Usage:
    python process_learn_pragma.py <source_file>
"""

import re
import sys
import os
import tempfile
import shutil


def is_cpp_file(filename):
    """Determine if a file is C++ based on extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ['.cpp', '.cxx', '.cc', '.C', '.hpp', '.hxx']


def process_file(input_file):
    """Process a single source file."""
    print(f"Processing file: {input_file}")
    
    # Read the file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if it's a C++ file
    is_cpp = is_cpp_file(input_file)
    
    # Check for existing stdio includes
    has_stdio = False
    include_pattern = re.compile(r'#\s*include\s*(<stdio\.h>|"stdio\.h"|<cstdio>|"cstdio")')
    if include_pattern.search(content):
        has_stdio = True
        print("  ✓ Already has stdio.h/cstdio include")
    else:
        print("  - Will add stdio include")
    
    # Pattern to match #pragma learn print directives
    # Matches both:
    #   #pragma learn print
    #   #pragma learn print identifier
    # Must be at the beginning of a line (ignoring whitespace)
    pragma_pattern = re.compile(
        r'(^[ \t]*#\s*pragma\s+learn\s+print(?:\s+(\w+))?[ \t]*\n?)',
        re.MULTILINE
    )
    
    # Find all matches
    matches = list(pragma_pattern.finditer(content))
    
    if not matches:
        print("  No #pragma learn print directives found")
        return False
    
    print(f"  Found {len(matches)} #pragma learn print directive(s)")
    
    # Process the content from bottom to top (so positions don't shift)
    modified_content = content
    for match in reversed(matches):
        full_match = match.group(1)
        identifier = match.group(2)
        
        # Build the printf statement
        if identifier:
            printf_stmt = f'    printf("#pragma learn print directive received identifier {identifier}\\n");\n'
        else:
            printf_stmt = '    printf("#pragma learn print directive received\\n");\n'
        
        # Insert after the pragma
        insert_pos = match.end()
        modified_content = modified_content[:insert_pos] + printf_stmt + modified_content[insert_pos:]
        print(f"  - Inserted printf at position {insert_pos}")
    
    # Add the include if needed
    if not has_stdio:
        include_stmt = '#include <cstdio>\n' if is_cpp else '#include <stdio.h>\n'
        modified_content = include_stmt + modified_content
        print(f"  - Added {include_stmt.strip()}")
    
    # Write to a temporary file first
    temp_fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(input_file)[1])
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        # Replace the original file
        shutil.move(temp_path, input_file)
        print(f"\n✓ Successfully modified: {input_file}")
        return True
        
    except Exception as e:
        os.unlink(temp_path)
        print(f"✗ Error writing file: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    success = process_file(input_file)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

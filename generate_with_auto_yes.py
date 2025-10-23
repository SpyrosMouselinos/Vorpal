#!/usr/bin/env python3
"""
Wrapper to run generation scripts with automatic 'yes' confirmation.
Used by run_full_pipeline.sh for unattended execution.
"""

import sys
import subprocess

if len(sys.argv) < 2:
    print("Usage: python generate_with_auto_yes.py <generation_script>")
    sys.exit(1)

script = sys.argv[1]

# Run the script with 'yes' piped to stdin
process = subprocess.Popen(
    ['python', script],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# Send 'yes' when prompted
stdout, _ = process.communicate(input='yes\n')
print(stdout)

sys.exit(process.returncode)


#!/bin/bash

# Script to run tests for the ZClip package

set -e  # Exit on any error

# Show Python version
echo "Using Python:"
python3 --version

# Run tests using hatch
echo "Running tests with hatch..."
hatch run test:default

echo "All tests completed successfully!"

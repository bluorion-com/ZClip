#!/bin/bash

# Script to run tests for the ZClip package

set -e  # Exit on any error

# Show Python version
echo "Using Python:"
python3 --version

# Run tests using hatch
if [ "$1" == "lightning" ]; then
    echo "Running tests with lightning support..."
    # Run tests using the test-lightning environment
    hatch run test-lightning:lightning
elif [ "$1" == "all" ]; then
    echo "Running all tests (including lightning if available)..."
    hatch run test-lightning:all
else
    echo "Running tests with default setup (excluding lightning)..."
    hatch run test:default
fi

echo "All tests completed successfully!"

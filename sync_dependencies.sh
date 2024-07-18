#!/bin/bash

# Export dependencies from Poetry to requirements.txt
poetry export -f requirements.txt --output requirements.txt
echo "Dependencies have been synced from pyproject.toml to requirements.txt and setup.py"

# Define linting-related packages and files
LINT_PACKAGES=black black-jupyter blacken-docs 
PYTHON_FILES=$(find . -type f -name '*.py')
IPYNB_FILES=$(find . -type f -name '*.ipynb')

# make a venv at ~/bind_gps_env
VENV="~/bind_gps_env/bin/activate"

lint:
	@echo "Running linting on all Python files..."
	black .
	isort .


permission:
	chmod -R 777 .
	chmod -R 777 .git

.PHONY: lint permission
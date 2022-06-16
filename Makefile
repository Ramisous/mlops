SHELL = /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv    : creates a virtual environment."
	@echo "style   : executes style formatting."
	@echo "clean   : cleans all unnecessary files."

style:
	black .
	python3 -m isort .
	flake8 
	

.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage	

# Test
.PHONY: test
test:
	pytest -m "not training"
	cd tests && great_expectations checkpoint run projects
	cd tests && great_expectations checkpoint run tags
	cd tests && great_expectations checkpoint run labeled_projects





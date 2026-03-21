
PYTHON=python
PIP=pip
NOTEBOOK=jupyter notebook

REQ=requirements.txt
BEST_MODEL_SCRIPT=src/train_best_gpr_variational.py

.PHONY: help install notebooks best-model clean check-artifacts tree

help:
	@echo "Available commands:"
	@echo "  make install         Install Python dependencies"
	@echo "  make notebooks       Launch Jupyter Notebook"
	@echo "  make best-model      Verify and rerun the final best GPR training logic"
	@echo "  make clean           Remove Python cache files and notebook checkpoints"
	@echo "  make check-artifacts Check that key folders and files exist"
	@echo "  make tree            Show expected project structure"

install:
	$(PIP) install -r $(REQ)

notebooks:
	$(NOTEBOOK)

best-model:
	$(PYTHON) $(BEST_MODEL_SCRIPT)

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

check-artifacts:
	@echo "Checking repository structure..."
	@test -d notebooks && echo "notebooks/ found" || echo "notebooks/ missing"
	@test -d artifacts && echo "artifacts/ found" || echo "artifacts/ missing"
	@test -d artifacts_bestPath && echo "artifacts_bestPath/ found" || echo "artifacts_bestPath/ missing"
	@test -f requirements.txt && echo "requirements.txt found" || echo "requirements.txt missing"
	@test -f $(BEST_MODEL_SCRIPT) && echo "$(BEST_MODEL_SCRIPT) found" || echo "$(BEST_MODEL_SCRIPT) missing"

tree:
	@echo "."
	@echo "├── notebooks/"
	@echo "├── src/"
	@echo "│   ├── Functions_PumpAI.py"
	@echo "│   └── train_best_gpr_variational.py"
	@echo "├── data/"
	@echo "├── artifacts/"
	@echo "├── artifacts_bestPath/"
	@echo "├── requirements.txt"
	@echo "├── Makefile"
	@echo "└── README.md"
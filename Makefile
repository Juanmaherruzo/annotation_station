CONDA_ENV   = sam_studio
BACKEND_DIR = backend
FRONTEND_DIR= frontend
BACKEND     = $(shell cygpath -w $(CURDIR)/$(BACKEND_DIR))
FRONTEND    = $(shell cygpath -w $(CURDIR)/$(FRONTEND_DIR))

.PHONY: install dev-backend dev-frontend clean

# Run once to install all dependencies
install:
	conda run -n $(CONDA_ENV) pip install -e "$(BACKEND_DIR)[dev]"
	cd $(FRONTEND_DIR) && npm install

# Start the FastAPI backend (run in its own terminal)
dev-backend:
	conda run -n $(CONDA_ENV) --no-capture-output \
	  cmd /c "set KMP_DUPLICATE_LIB_OK=TRUE && cd /d $(BACKEND) && uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

# Start the Vite dev server (run in its own terminal)
dev-frontend:
	cd $(FRONTEND_DIR) && conda run -n $(CONDA_ENV) --no-capture-output npx vite --port 5173

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(FRONTEND_DIR)/dist $(FRONTEND_DIR)/node_modules

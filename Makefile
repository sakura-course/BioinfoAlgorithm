.PHONY: list run

list:
	@echo "Available commands:"
	@echo "  run - Run the application using uv"

run:
	uv run src/main.py

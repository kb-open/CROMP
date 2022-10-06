pytest:
	pytest

install:
	python -m pip install -e .

pypi:
	python setup.py sdist
	twine upload dist/*

clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache

ut:
	python tests/cromp_tests.py

test: install ut


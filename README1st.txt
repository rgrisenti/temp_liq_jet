Installing packages locally
---------------------------

The required directory structure is

Project/                # Root project folder
├── src/                # Source folder
│   └── temp_liq_jet/   # Actual Python package
│       ├── __init__.py
│       ├── module1.py
│       └── module2.py
├── tests/              # Optional: tests
│   └── test_module1.py
├── README.md
├── LICENSE
├── pyproject.toml
└── setup.cfg           # Optional

The __init__.py can be empty, or can expose selected names (preferred):

	from .temp_liq_jet import KnudsenModel

First build the project,

python3.1x> python -m build

(
It can be tested locally by installing it via:

python3.1x> python -m pip install --upgrade dist/*.whl
)

Then, the project can be uploaded (after creating .pypirc in ~/),

  1. To TestPyPI

     python3.1x> python -m twine upload --repository testpypi dist/*

     and installed as

     python3.1x> python -m pip install --upgrade --index-url https://test.pypi.org/simple/ --no-deps temp_liq_jet

  2. To PyPI

     python3.1x> python -m twine upload dist/*

     and installed as usual,

     python3.1x> python -m pip install --upgrade temp_liq_jet

#!/bin/bash

pip install -e .
python3 setup.py sdist
python3 setup.py bdist_wheel --universal
twine upload dist/*
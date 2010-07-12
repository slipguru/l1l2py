#!/bin/bash -x

./clean_wc.sh

# Preparing nosetest output
echo '$ nosetests' > docs/tests.txt
nosetests &>> docs/tests.txt

cd docs
make html
make latex
cd _build/latex
make all-pdf &> /dev/null
cd -
cd ..

python setup.py sdist

cp -R docs/_build/html dist/l1l2py_html
cp docs/_build/latex/l1l2py.pdf dist/

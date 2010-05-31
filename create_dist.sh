#!/bin/bash -x

cd doc
make html
make latex
cd _build/latex
make all-pdf &> /dev/null
cd -
cd ..

python setup.py sdist

cp -R doc/_build/html dist/l1l2py_html
cp doc/_build/latex/l1l2py.pdf dist/

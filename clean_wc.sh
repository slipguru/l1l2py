#!/bin/bash -x

rm MANIFEST
rm -rf dist
rm -rf build

cd docs
make clean
cd ..


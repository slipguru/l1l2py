#!/bin/bash -x

rm MANIFEST
rm -rf dist
rm -rf build

cd doc
make clean
cd ..


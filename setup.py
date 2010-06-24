import os
from distutils.core import setup

from l1l2py import __version__ as version
README = os.path.join(os.path.dirname(__file__), 'README')
lines = open(README).readlines()
description = ''.join(lines[:2])
long_description = ''.join(lines[2:])

setup(
    name='L1L2Py',
    version=version,
    url='http://slipguru.disi.unige.it/Research/L1L2Py',
    description=description,
    long_description=long_description,
    keywords='feature selection, regularization, regression, classification,'
             'l1-norm, l2-norm',
    author='L1L2Py developers - SlipGURU',
    author_email='salvatore.masecchia@disi.unige.it',
    license='GNU GPL version 3',
    download_url = 'http://slipguru.disi.unige.it/Research/L1L2Py/L1L2Py-1.0.0.tar.gz',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],

    packages=['l1l2py', 'l1l2py.tests'],
    package_data={'l1l2py.tests': ['data.txt']},
    requires=['numpy (>=1.3.0)']
)

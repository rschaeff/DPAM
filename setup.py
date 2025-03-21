#!/usr/bin/env python3
"""
DPAM: Domain Parser for AlphaFold Models - Setup Configuration

This setup script configures the DPAM package for installation.
"""

import os
from setuptools import setup, find_packages

# Read version from __init__.py
with open(os.path.join('dpam', '__init__.py'), 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip("'\"")
            break
    else:
        version = '0.1.0'  # Default if not found

# Read long description from README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if not line.startswith('#') and line.strip()]

setup(
    name="dpam",
    version=version,
    author="DPAM Development Team",
    author_email="dpam-dev@example.com",
    description="Domain Parser for AlphaFold Models - Automated protein domain identification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/dpam",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.9',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'dpam-manager=dpam.bin.dpam-manager:main',
            'dpam-worker=dpam.cli.worker:main',
            'dpam-api=dpam.api.server:start_server',
            'dpam-monitor=dpam.grid.monitor:main',
        ],
    },
    include_package_data=True,
    package_data={
        'dpam': ['schema/*.sql', 'config.json.template'],
    },
    zip_safe=False,
)
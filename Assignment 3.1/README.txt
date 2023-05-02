1) open Anaconda terminal and run below command for create virtual environment
conda env create -f env.yaml

2) use below command to activate environment
conda activate myenv

3) for test run below command
python test.py

4) for import as package 
from my_package import train
from my_package import score
from my_package import ingest_data

5)install .whl file
py -m pip install --index-url https://test.pypi.org/simple/ --no-deps my_package-0.1-py3-none-any

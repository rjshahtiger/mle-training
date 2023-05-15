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
pip install  my_package-0.1-py3-none-any.whl

6)ingest_data
form my_package-0.1-py3-none-any import ingest_data
ingest_data.ingest_data(download_root)

7) train
form my_package-0.1-py3-none-any import train
train.train(read_train_file, output_model_path)

8)score
from my_package-0.1-py3-none-any import score
score.score(validate_file_path,pickle_file_path)

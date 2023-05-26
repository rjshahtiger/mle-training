#Deriving the latest base image
FROM python:latest


#Labels as key value pair
LABEL Maintainer="ronak"


# Any working directory can be chosen as per choice like '/' or '/home' etc
# i have chosen /usr/app/src
WORKDIR /usr/app/src

#to COPY the remote file at working directory in container
RUN apt update -y
RUN python -m pip install --upgrade pip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r ./requirements.txt
COPY file/ingest_data.py ./
COPY file/train.py ./
COPY file/score.py ./
COPY file/run.py ./
# Now the structure looks like this '/usr/app/src/test.py'


#CMD instruction should be used to run the software
#contained by your image, along with any arguments.

CMD [ "python", "./run.py" ]
#"./train.py", "./score.py" ]
# CMD echo python ./ingest_data.py; echo python ./train.py; echo python ./score.py 
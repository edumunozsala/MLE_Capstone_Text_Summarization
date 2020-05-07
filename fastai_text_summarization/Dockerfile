# Install the latest version of pytorch
FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime 
WORKDIR /root

# Installs pandas, and google-cloud-storage.
RUN pip install pandas google-cloud-storage numpy fastai==1.0.60

# The data for this sample has been publicly hosted on a GCS bucket.
# Download the data from the public Google Cloud Storage bucket for this sample
#RUN curl https://storage.googleapis.com/cloud-samples-data/ml-engine/sonar/sonar.all-data --output ./sonar.all-data

# Copies the trainer code to the docker image.
COPY model.py ./model.py
COPY data_process.py ./data_process.py
COPY task.py ./task.py

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "task.py"]
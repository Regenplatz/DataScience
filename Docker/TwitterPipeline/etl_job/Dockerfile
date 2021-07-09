# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /etl
WORKDIR /etl

# Copy the requirements file into the container at /etl
# (a text files with all the libraries you want to install)
# ADD requirements.txt /etl
COPY . /etl

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.org -r requirements.txt
# RUN curl install -r (cat requirements.txt)

# Run etl.py when the container launches
CMD ["python", "etl.py"]
# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Set the working directory to /tweety
WORKDIR /tweety

# Copy the requirements file into the container at /tweety
# (a text files with all the libraries you want to install)
#ADD requirements.txt /tweety
COPY . /tweety

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.org -r requirements.txt
#RUN curl install -r (cat requirements.txt)

# Run tweety.py when the container launches
CMD ["python", "tweety.py"]

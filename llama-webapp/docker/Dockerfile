# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
#COPY . /app
COPY . /app
#RUN ls /app

# Install any needed packages specified in requirements.txt
#RUN pip install -r requirements.txt
RUN pip install Flask==3.0.0 pydantic==1.10.8 llama-llm==0.0.15

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run the Flask application when the container launches
CMD ["python", "app.py"]

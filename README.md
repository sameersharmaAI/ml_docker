# Machine Learning Model Using Docker

## Overview of the Project
The project is a basic application of Machine Learning using Docker for its deployment. It uses the Iris dataset and implements a Decision Tree Classifier that predicts the species of an iris flower from its features. In this application, Flask a lightweight web framework used so that users can pass input to the model using an API endpoint.

## Instructions to Build and Run the Docker Container

1. **Clone the Repository**:
    
   - git clone https://github.com/sameersharmaAI/ml_docker.git
    

2. **Build the Docker Image**:
   Make sure that you are in the project directory and run the following command:
    
   - sudo docker build -t ml-app .
    

3. **Run the Docker Container**:
   Use following command to run the container:

   - sudo docker run -p 4000:80 ml-app
    

## Instructions to Test the ML Endpoint

1. **Access the Application**: 
   - Open your browser and navigate to http://localhost:4000 to see the running application.
   - Note: Use external IP address of the VM, if you are using any cloud services.

2. **Test the /predict Endpoint**: 
   Test the /predict endpoint using curl or Postman by sending a POST request with JSON data:
    
    - curl -X POST http://localhost:4000/predict -H "Content-Type: application/json" -d '{"input": [5.1, 3.5, 1.4, 0.2]}'
    - Replace the input values with the features of the iris flower you want to predict.

## Other Relevant Information

- Please ensure that the Docker is installed and running on your machine before you begin building or running the application.
- Install the requirements.txt dependencies.
- The code has a simple script `train_model.py`, which trains the Decision Tree Classifier and saves it as `model.pkl`.
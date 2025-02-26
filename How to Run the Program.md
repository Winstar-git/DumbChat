# Steps to Clone and Run the Program

Follow these steps to successfully clone and run the project:

## 1. Clone the Repository

First, create a file directory where you want to clone this repository to your local machine using Git:

---bash

cd Documents

mkdir Github

git clone https://github.com/Winstar-git/DumbChat.git

## 2. Navigate to the Project Directory

After cloning, go into the project directory:

---bash

cd DumbChat

## 3. Set Up the Virtual Environment

It's important to create a virtual environment to manage dependencies for the project.

---bash

python -m venv venv

source venv/Scripts/activate

## 4. Install Dependencies
With the virtual environment activated, you can install the required Python packages by running:

---bash

pip install -r requirements.txt

Other method is:
pip install transformers torch

## 5. Run the Application
Now you are ready to run the chatbot! Start the Python script:

---bash

python DumbGPT.py

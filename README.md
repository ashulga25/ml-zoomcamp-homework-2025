# ml-zoomcamp-homework-2025
## Repo for homework of ML zoomcamp 2025 course

# Install VS Code extension for `GitHib Codespaces`
# Configure environment terminal via PS1="> "

# Add required python libraries

$ pip install numpy pandas scikit-learn seaborn jupyter && pip list


# Start jupyter notebook remotely (ignore the pop-up) and we can use it locally
# The port-forwarding from remote VM on port 8888 is forwarded to a local machine
# So we can open a dynamic link on PORTS tab and it will open a browser window with running jupyter
# We can take the token from log print-out in terminal

$ jupyter notebook 

http://localhost:8888/tree?token=<some token>


# Next we can create a new folder in our repo vis VS Code and put there the solutions for the homework
# Inside the folder `01-intro` create a new notebook called `homework.ipynb` and add code:

import pandas as pd

df = pd.read_csv('https://raw/githubusercontent.com/alexeygriforiev/datasets/master/car_fuel_efficiency.csv')
df.head()

OR
wget https://raw/githubusercontent.com/alexeygriforiev/datasets/master/car_fuel_efficiency.csv





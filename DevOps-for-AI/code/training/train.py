"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import pickle
#from azureml.core import Workspace
from azureml.core import Workspace, Dataset
from azureml.core.run import Run
import os
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import json
import subprocess
from typing import Tuple, List
from pmdarima import auto_arima
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

# run_history_name = 'devops-ai'
# os.makedirs('./outputs', exist_ok=True)
# #ws.get_details()
# Start recording results to AML
# run = Run.start_logging(workspace = ws, history_name = run_history_name)
run = Run.get_submitted_run()


# Get a dataset from the workspace datasets collection
dataset = ws.datasets['airline']
data = dataset.to_pandas_dataframe()
data.index.freq = 'MS'
dataset_ts = data[['Thousands of Passengers']]


print("Running train.py")

# Randomly pic alpha
model = auto_arima(dataset_ts, trace=False, error_action='ignore', suppress_warnings=True,m=5, stepwise=True)


# Save model as part of the run history
model_name = "airline_autoArima_model.pkl"
# model_name = "."

with open(model_name, "wb") as file:
    joblib.dump(value=model, filename=model_name)

# upload the model file explicitly into artifacts
run.upload_file(name="./outputs/" + model_name, path_or_stream=model_name)
print("Uploaded the model {} to experiment {}".format(model_name, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)


# register the model
run.log_model(file_name = model_name)
print('Registered the model {} to run history {}'.format(model_name, run.history.name))


print("Following files are uploaded ")
print(run.get_file_names())
run.complete()

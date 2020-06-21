# Pulsar Prediction using the High Time Resolution Universe Survey 🌌

I'd like to preface this document by stating this is my first major project using Python. I aim to keep learning more and more everyday, and hope this project provides some benefit to the greater scientific community.

The purpose of this data mining application is to make use of the High Time Resolution Universe Survey using random forest classification to predict whether a new candidate is a pulsar or some other non-pulsar object (e.g., noise). For more information about pulsars, check out NASA's <ins>[website](https://www.nasa.gov/subject/8731/pulsars/)</ins>. Eight variables act as test parameters for this project, with four relating to the integrated profile of each candidate and four relating to the dispersion measure signal-to-noise ratio curve of each candidate. I'm by no means knowledgeable in astrophysics, but I do find the topic extremely interesting and hope you do too after using this statistical tool.

This project's script will open a `streamlit` application using a localhost in your default browser. An example video of what you'll see on this website can be found <ins>[here](Pulsar.gif)</ins>. The model will run by updating the test parameters located on the sidebar, or rerun using the [R] key. Output on this website includes model error summaries, model variable importances, and predicted class probabilities and insights.  

## 💻 Installation and Preparation

Please note that excerpts of code provided below are examples based on the <ins>[Pulsar.py](Pulsar.py)</ins> script. As a new learner myself, I suggest reading through my insights, mixing them with Google search and your own experiences, and then delving into the script and running the application.

For this project, I used <ins>[Python 3.8](https://www.python.org/downloads/)</ins>, the Microsoft Windows operating system, and Microsoft Excel. As such, these act as the prerequisites for running this application successfully. Going forward, please ensure everything you download or install for this project ends up in the correct location (e.g., the same folder). This includes the <ins>[Pulsar.csv](Pulsar.csv)</ins> data file.

Use <ins>[pip](https://pip.pypa.io/en/stable/)</ins> to install relevant packages to the folder designated above using the Command Prompt and correct PATH. For example:

```bash
pip install streamlit
pip install pandas
```
Please be sure to install each of the following packages: `streamlit`, `pandas`, `sklearn`, `numpy`, `matplotlib`, and `Pillow`.
## 📑 Script Breakdown
The current random seed for this project is 874. You may remove the random seed if fluctuating results are desired. You may also set the random seed to a value of your own preference (e.g., 12345 or 1337).
```python
import numpy as np

np.random.seed(874)
```
Importing the data and defining the predictor and outcome variable(s) is essential for this analysis. For example:
```python
import pandas as pd

data = pd.read_csv('Data.csv')

predictor_variables = data[['x1', 'x2', 'x3']]
outcome_variable = data[['y']]
outcome_variable = np.ravel(outcome_variable)
```
Next, parameters need to be defined based on the variables above. In the example below, the numerical values 0, 8, and 4 represent the minimum possible value, maximum possible value, and starting value for each parameter. I used averages for each variable as parameter starting values in this project.
```python
import streamlit as st

def design_parameters():

    x1 = st.sidebar.slider('Variable 1', 0, 8, 4)
    x2 = st.sidebar.slider('Variable 2', 0, 8, 4)
    x3 = st.sidebar.slider('Variable 3', 0, 8, 4)

    variables = {'Variable 1': x1, 'Variable 2': x2, 'Variable 3': x3}

    parameters = pd.DataFrame(variables, index=[0])
    return parameters
````
Following parameter design comes the fun part, at least statistically speaking. Random forest classification is a very cool, robust data mining technique and if you're unfamiliar with it, I thoroughly encourage you to read about the <ins>[method](https://en.wikipedia.org/wiki/Random_forest)</ins> further. Below we define model arguments, run the analysis, and create various outputs, such as a graph of variable importances.
```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

analysis = design_parameters()
random_forest = RandomForestClassifier(n_estimators = 100, max_features = 'sqrt', oob_score = True)

random_forest.fit(predictor_variables, outcome_variable)
prediction_result = random_forest.predict(analysis)
probability_result = random_forest.predict_proba(analysis)
error_result = str(round(1 - random_forest.oob_score_, 4))
importances = random_forest.feature_importances_

plt.figure()
plt.xticks(np.arange(0,0.5,0.05))
plt.title('Prediction Model')
plt.barh(range(predictors.shape[1]), importances, color = 'coral', align = 'center')
plt.yticks(range(predictors.shape[1]), ['Variable 1', 'Variable 2', 'Variable 3'])
plt.ylim([-1, predictors.shape[1]])
plt.xlabel('Importances')
plt.savefig('Importances.png')
````
Please note that graphs used in this project have hard-coded y-axis labels; if you alter the script or adapt it for your own purposes, I highly recommend removing lines 60, 76 and updating lines 62, 78 from <ins>[Pulsar.py](Pulsar.py)</ins> accordingly.

The remainder of the script for this project is all about taking output from the analysis and formatting it in a way that is tolerable to the typical research user. This includes a) header generation, b) model error reporting, c) image uploading for importances, d) stripping, splitting, and manipulating strings of probabilities, and e) using logic to automatically update insights given probabilities.

In general, typing headers and body text using the package `streamlit` is pretty straightforward, but here is a quick example:
```python
st.title('Primary Header')
st.subheader('Secondary Header')
st.write('Body Text')
```

To run the application, please type the following into the Command Prompt, again using the correct PATH:
```bash
streamlit run Pulsar.py
```
Please allow the application approximately 15 seconds to run; once complete, you'll see the results for a candidate that has the average score on each variable. Tweak each parameter accordingly and the application will then run the model again immediately.

## 📋 Next Steps
Although I feel this project is solid in its current state, I think one area of improvement would be to replace the sidebar sliders with numerical input boxes. The current sliders are not granular enough to capture every test value for parameters with very large ranges. To this end, I'm open to thoughts and opinions!

## 💡 Community Contribution
I'm always happy to receive feedback, recommendations, and/or requests from anyone, especially new learners. Please click <ins>[here](LICENSE.md)</ins> for information about this project's license.

## ❔ Project Support
Please let me know if you plan to make changes to this project, or adapt the script to a project of your own interest. We can certainly collaborate to make this process as painless as possible!

## 📚 Additional Resources
- To see what inspired my project, click <ins>[here](https://towardsdatascience.com/how-to-build-a-simple-machine-learning-web-app-in-python-68a45a0e0291)</ins> to access Towards Data Science <br/>
- To download the original data, click <ins>[here](https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star)</ins> to access Kaggle <br/>
- To learn more about calculating random forest classification in Python, click <ins>[here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)</ins> to access scikit-learn <br/>
- To learn more about calculating Mahalanobis distances in R, click <ins>[here](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/mahalanobis)</ins> to access R Documentation; additionally, <ins>[here](MahalanobisDistances.R)</ins> is the R script I personally used before beginning this project <br/>
- For easy-to-use text editing software, check out <ins>[Sublime Text](https://www.sublimetext.com/)</ins> for Python and <ins>[Atom](https://atom.io/)</ins> for Markdown

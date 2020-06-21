#Import packages and set the random seed

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(874)


#Read data file and organize variables for the analysis

pulsar = pd.read_csv('Pulsar.csv')

predictors = pulsar[['integrated_profile_mean', 'integrated_profile_standard_deviation', 'integrated_profile_excess_kurtosis', 'integrated_profile_skewness', 
	'dmsnr_curve_mean', 'dmsnr_curve_standard_deviation', 'dmsnr_curve_excess_kurtosis', 'dmsnr_curve_skewness']]
outcome_pulsar, outcome_outlier = pulsar[['outcome']], pulsar[['outlier']]
outcome_pulsar, outcome_outlier = np.ravel(outcome_pulsar), np.ravel(outcome_outlier)


#Establish parameters based on descriptive statistics

st.sidebar.header('Test Parameters')

def test_parameters():

    integrated_profile_mean = st.sidebar.slider('Integrated Profile - Mean', 5.82, 192.61, 111.08)
    integrated_profile_standard_deviation = st.sidebar.slider('Integrated Profile - Standard Deviation', 24.78, 98.77, 46.55)
    integrated_profile_excess_kurtosis = st.sidebar.slider('Integrated Profile - Excess Kurtosis', -1.87, 8.06, 0.48)
    integrated_profile_skewness = st.sidebar.slider('Integrated Profile - Skewness', -1.78, 68.09, 1.77)
    dmsnr_curve_mean = st.sidebar.slider('DM-SNR Curve - Mean', 0.22, 223.38, 12.61)
    dmsnr_curve_standard_deviation = st.sidebar.slider('DM-SNR Curve - Standard Deviation', 7.38, 110.63, 26.33)
    dmsnr_curve_excess_kurtosis = st.sidebar.slider('DM-SNR Curve - Excess Kurtosis', -3.13, 34.53, 8.30)
    dmsnr_curve_skewness = st.sidebar.slider('DM-SNR Curve - Skewness', -1.97, 1190.99, 104.86)

    variables = {'Integrated Profile - Mean': integrated_profile_mean, 'Integrated Profile - Standard Deviation': integrated_profile_standard_deviation, 
        'Integrated Profile - Excess Kurtosis': integrated_profile_excess_kurtosis, 'Integrated Profile - Skewness': integrated_profile_skewness, 
        'DM-SNR Curve - Mean': dmsnr_curve_mean, 'DM-SNR Curve - Standard Deviation': dmsnr_curve_standard_deviation, 
        'DM-SNR Curve - Excess Kurtosis': dmsnr_curve_excess_kurtosis, 'DM-SNR Curve - Skewness': dmsnr_curve_skewness}

    parameters = pd.DataFrame(variables, index=[0])
    return parameters


#Run random forest classification for outcome variables

analysis = test_parameters()
random_forest = RandomForestClassifier(n_estimators = 100, max_features = 0.55, oob_score = True)

random_forest.fit(predictors, outcome_pulsar)
prediction_pulsar = random_forest.predict(analysis)
estimate_pulsar = random_forest.predict_proba(analysis)
error_pulsar = str(round(1 - random_forest.oob_score_, 2)).ljust(4, '0')
importances_pulsar = random_forest.feature_importances_

plt.figure()
plt.xticks(np.arange(0,0.5,0.05))
sort = np.argsort(importances_pulsar)
plt.title('Pulsar Prediction Model')
plt.barh(range(predictors.shape[1]), importances_pulsar[sort], color = 'coral', align = 'center')
plt.yticks(range(predictors.shape[1]), ['DC - EK', 'DC - S', 'IP - SD', 'DC - SD', 'DC - M', 'IP - M', 'IP - S', 'IP - EK'])
plt.ylim([-1, predictors.shape[1]])
plt.xlabel('Importances')
plt.savefig('Pulsar_Importances.png')

random_forest.fit(predictors, outcome_outlier)
prediction_outlier = random_forest.predict(analysis)
estimate_outlier = random_forest.predict_proba(analysis)
error_outlier = str(round(1 - random_forest.oob_score_, 2)).ljust(4, '0')
importances_outlier = random_forest.feature_importances_

plt.figure()
plt.xticks(np.arange(0,0.5,0.05))
sort = np.argsort(importances_outlier)
plt.title('Multivariate Outlier Prediction Model')
plt.barh(range(predictors.shape[1]), importances_outlier[sort], color = 'deepskyblue', align = 'center')
plt.yticks(range(predictors.shape[1]), ['IP - M', 'IP - SD', 'IP - EK', 'DC - S', 'DC - SD', 'DC - EK', 'IP - S', 'DC - M'])
plt.ylim([-1, predictors.shape[1]])
plt.xlabel('Importances')
plt.savefig('Outlier_Importances.png')


#Build out results section and automatic insights

st.title('Pulsar Prediction using the High Time Resolution Universe Survey')
st.subheader('_A data mining application for astrophysics using Python_')

st.subheader('')
st.subheader('Overall Model Summaries')

st.write('Out-of-Bag Error (Pulsar Prediction Model) = ' + error_pulsar)
st.write('Out-of-Bag Error (Multivariate Outlier Prediction Model) = ' + error_outlier)
pulsar_graph = Image.open('Pulsar_Importances.png')
st.image(pulsar_graph)
outlier_graph = Image.open('Outlier_Importances.png')
st.image(outlier_graph)

st.subheader('Predicted Class Probabilities')

probability_pulsar = str(estimate_pulsar).strip('\'[]')
probability_pulsar = probability_pulsar.split()
probability_outlier = str(estimate_outlier).strip('\'[]')
probability_outlier = probability_outlier.split()
st.write('Pr(Non-Pulsar) = ' + probability_pulsar[0].ljust(4, '0'))
st.write('Pr(Pulsar) = ' + probability_pulsar[1].ljust(4, '0'))
st.write('Pr(Non-Multivariate Outlier) = ' + probability_outlier[1].ljust(4, '0'))
st.write('Pr(Multivariate Outlier) = ' + probability_outlier[0].ljust(4, '0'))

st.subheader('')
st.subheader('Predicted Class Results and Insights')

if prediction_pulsar == 'non-pulsar':
	st.write('This candidate is _probably_ a ' + str(prediction_pulsar).strip('\'[]') + ' object based on random forest classification.')
elif prediction_pulsar == 'pulsar':
	st.write('This candidate is _probably_ a ' + str(prediction_pulsar).strip('\'[]') + ' based on random forest classification.')

if prediction_outlier == 'multivariate outlier':
	st.write('Additionally, this candidate is _probably_ a ' + str(prediction_outlier).strip('\'[]') + ' based on Mahalanobis distances \
		and random forest classification.')
elif prediction_outlier == 'non-outlier':
	st.write('Additionally, this candidate is _probably not_ a multivariate outlier based on Mahalanobis distances \
		and random forest classification.')

#Last updated by Thomas J. Nicoletti on June 21, 2020
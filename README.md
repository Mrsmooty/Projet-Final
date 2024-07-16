# Projet-Final

DREEM 🧠
Dreem is a french neurotechnology startup which works on the Dreem headband, a tool for sleep analysis and enhancement to help people improve their sleep. The Dreem headband collects physiological activity during sleep, such as brain activity (EEG), respiration and heart rate. Physiological signals are analyzed throughout the night and allows to perform a detailed analysis of sleep. Sound stimulation is also provided to enhance sleep quality at different steps of the night : falling asleep, deep sleep and awakening.

Predicting sex ♀️♂️ from EEG data 🧠
Using deep convolutional neural networks, with unique potential to find subtle differences in apparent similar patterns, we explore if brain rhythms from either sex contain sex specific information. Here we show, in a ground truth scenario, that a deep neural net can predict sex from scalp electroencephalograms with an accuracy of >80% (p < 10−5), revealing that brain rhythms are sex specific.”

Dreem has a huge database composed of 1 000 000 of full night EEG recordings. We reproduced a dataset of good quality EEG signal to allow reproducing the described results. The main difference is the number of EEG channels used (7 vs 24 in the paper) but this is compensated by the face that 1) EEG channels are highly correlated in EEG recordings 2) we gathered more subjects than in the original study.

Project goals 📈
We try to predict the gender of someone based on 40 windows of 2 seconds taken during sleep.


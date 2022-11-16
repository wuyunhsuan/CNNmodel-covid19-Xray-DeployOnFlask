# CNNmodel-covid19-Xray-DeployOnFlask

reference
Kaggle Covid-19 Image Dataset
https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset
The original dataset seperates pics to train/test and has 3 different labels(covid/normal/Viral Pneumonia).
However, I combined train set with test set, and split data into new train/test set randomly(with sklearn.model_selection.train_test_split).

And after that I deploy a flask to show the prediction.
Users can upload a xray picture and press 'predict' to show the result.

**
Covid-19 Image Dataset
Covid19 file is too big to upload but can check on Kaggle.
**

## Predictor Function for Covid 19 Model

* Download the Pre-trained Model from link in root Readme or create the Model by using the Jupyter Notebook

* For Predict function, replace Image path and label in `Predict.py`

* For labels:
	* Covid Expected: 1
	* Non-Covid Expected: 0

* 4 Cases Returned
	* Prediction == 1 and Expectation == 1: True Positive
	* Prediction == 0 and Expectation == 1: False Negative
	* Prediction == 1 and Expectation == 0: False Positive
	* Prediction == 0 and Expectation == 0: True Negative
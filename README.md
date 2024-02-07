Cardiovascular Disease Prediction Using Machine Learning and XAI 
Abstract: Advances in supporting technology, particularly in big  data and machine learning, have given predictive analytics a lot of  attention in recent years. The healthcare sector has recently seen the  application of disease prediction. This project has been 
demonstrated the prediction of a heart attack among many others.  Heart disease, more commonly referred to as cardiovascular  disease, is a group of conditions that affect the heart and has  emerged as the world's leading cause of death in recent decades. It  connects a slew of heart disease risk factors to the urgent need for  precise, dependable, and practical methods for early diagnosis and  management. Data mining is a common method for analyzing a lot  of data in the healthcare industry because it is difficult to predict  cardiac disease. To avoid the risks associated with it and to inform  the patient well in advance, the procedure must be automated. Data  mining methods like XG-Boost , Logistic Regression, Decision  tree, Support Vector Classifier, Random Forest, K-neighbors  classifier, and Naive Bayes can be used to identify heart diseases.  We have demonstrated through this project that, when it comes to  predicting heart attacks, XG-BOOST and KNN performs better  than the other machine learning models mentioned above.  Consequently, the performance of several machine learning  algorithms is compared in this paper. The dataset's features play a  crucial role in any kind of prediction. The final prediction can be  influenced positively or negatively by features. We have got a  95.60% accuracy rate by using KNN algorithm and used  explainable AI technique to explain the reason of heart disease. The  XAI methods can be used to show how important features are. The  predictability of the model is also interpreted using a new method  in this paper. By utilizing the XAI technique LIME with the  assistance of the idea of a black box, this exploration leads the  KNN calculations expectation 
Keywords :Heart Disease; Prediction; Machine learning; Random Forest; KNN  algorithm; Support Vector Machine; Decision Tree; Logistic Regression; Naive  Bayes, XG Boost,XAI.
3	Investigation/Experiment, Result, Analysis, and Discussion
![image](https://github.com/UtshoData/heart-disease-prediction-using-ML-and-XAI/assets/157609050/7218f3a4-a23d-4866-aa6d-3acb8d1def80)
With an accuracy of 95.60 percent, the KNN model makes 100 right predictions and 92 incorrect predictions. By using this model, we have got a precision of 0.95, recall of 0.97, and f-score of
0.96 for negative results. But for positive results, we have got 0.97 precision,0.94 recall and 0.95 f-1 score. At last, we have got a 95.60% accuracy rate which we can show in Figure 4.3 and Figure 4.4.
![image](https://github.com/UtshoData/heart-disease-prediction-using-ML-and-XAI/assets/157609050/eaad8537-1b7f-478e-80f8-9821b1943aca)
![image](https://github.com/UtshoData/heart-disease-prediction-using-ML-and-XAI/assets/157609050/ffeb541f-88fe-40fe-bbe9-86a30b292ce1)
![image](https://github.com/UtshoData/heart-disease-prediction-using-ML-and-XAI/assets/157609050/f219b7e7-ed26-40fd-9e8b-22100f132a70)
![image](https://github.com/UtshoData/heart-disease-prediction-using-ML-and-XAI/assets/157609050/00ea5f68-5985-42cf-93ff-f605e1c82c9c)
In the paper, the heart attack rate is predicted for some different models. We have used Logistic Regression, Decision Tree, Random Forest, Support Vector Classifier, Naive Bayes Classifier, XG-Boost Classifier, and K-Neighbors Classifier algorithms to find the accuracy of a heart attack. We have shown the accuracy percentage of different algorithms in Fig 4.4.
A heart attack may vary depending on various circumstances. In machine learning, these circumstances are defined as features. The features that have the biggest impact on predictions are called feature importance. In some cases, a few attributes may decrease the accuracy level of a model. So, it is important to work with the correct attributes. Though we have used all the features, feature importance from the dataset is shown in Fig 4.6.
![image](https://github.com/UtshoData/heart-disease-prediction-using-ML-and-XAI/assets/157609050/f1da27f2-8695-435d-97a6-914a2ba5642b)

Figure14: Important Features of The Dataset
As shown in Fig 4.6, the feature cp which is chest pain has the most impact on the chances of having a heart attack.
We have taken a local example here. We can see that the top 4 features contributing to the prediction of the feature are old peak, ca, cp, and that.
Old-peak, which is the most important feature in this prediction is related to the electrocardiograph results of the patient. If the electrocardiograph shows an ST Depression (indicating that the patient is suffering from myocardial ischemia), the value of the old peak shows us the level of the ST Depression. The more the depression, the likelier it is that the patient is suffering from heart disease.
All the features have value to count. By analyzing all the value counts we predict the rate of a heart attack. All the features have value to count. By analyzing all the value counts we predict the rate of a heart attack.
 
 
![image](https://github.com/UtshoData/heart-disease-prediction-using-ML-and-XAI/assets/157609050/411b3630-d8d5-4f89-b49b-7f3e973fb4cf)

Figure 16: Value Counts of The Feature
In the paper, Fig 4.7 represents the more and fewer chances of heart attack from value counts. It means the dataset has 51.32% of '1' which predicts more chances of heart attack and 48.68% of '0' which predicts fewer chances of a heart attack.
![image](https://github.com/UtshoData/heart-disease-prediction-using-ML-and-XAI/assets/157609050/6a6261d8-6d3c-4ee8-ba33-a4ae1435847a)
Figure 17: Feature Importance Using LIME
In fig-4.8, for this patient, heart disease is predicted for orange features which are important features for heart disease whereas blue features are less important. Orange features are a more important feature for heart disease. These volumes are high for predicting heart disease.
 
![image](https://github.com/UtshoData/heart-disease-prediction-using-ML-and-XAI/assets/157609050/6a6261d8-6d3c-4ee8-ba33-a4ae1435847a)
 ![image](https://github.com/UtshoData/heart-disease-prediction-using-ML-and-XAI/assets/157609050/106cb6f8-4a5d-47a0-ae53-16f817f92fa7)


Figure 18: Feature Importance Using LIME
In fig-4.9, for this patient, heart disease is not predicted for blue features which are not important features for heart disease whereas orange features are more important. Here, the volume of blue features is higher than the orange volume. For that reason, our model is predicting "No Disease" for this particular patient.



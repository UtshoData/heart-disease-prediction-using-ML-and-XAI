Cardiovascular Disease Prediction Using Machine Learning and XAI 
Abstract: Advances in supporting technology, particularly in big  data and machine learning, have given predictive analytics a lot of  attention in recent years. The healthcare sector has recently seen the  application of disease prediction. This project has been 
demonstrated the prediction of a heart attack among many others.  Heart disease, more commonly referred to as cardiovascular  disease, is a group of conditions that affect the heart and has  emerged as the world's leading cause of death in recent decades. It  connects a slew of heart disease risk factors to the urgent need for  precise, dependable, and practical methods for early diagnosis and  management. Data mining is a common method for analyzing a lot  of data in the healthcare industry because it is difficult to predict  cardiac disease. To avoid the risks associated with it and to inform  the patient well in advance, the procedure must be automated. Data  mining methods like XG-Boost , Logistic Regression, Decision  tree, Support Vector Classifier, Random Forest, K-neighbors  classifier, and Naive Bayes can be used to identify heart diseases.  We have demonstrated through this project that, when it comes to  predicting heart attacks, XG-BOOST and KNN performs better  than the other machine learning models mentioned above.  Consequently, the performance of several machine learning  algorithms is compared in this paper. The dataset's features play a  crucial role in any kind of prediction. The final prediction can be  influenced positively or negatively by features. We have got a  95.60% accuracy rate by using KNN algorithm and used  explainable AI technique to explain the reason of heart disease. The  XAI methods can be used to show how important features are. The  predictability of the model is also interpreted using a new method  in this paper. By utilizing the XAI technique LIME with the  assistance of the idea of a black box, this exploration leads the  KNN calculations expectation 
Keywords :Heart Disease; Prediction; Machine learning; Random Forest; KNN  algorithm; Support Vector Machine; Decision Tree; Logistic Regression; Naive  Bayes, XG Boost,XAI.
3 Result and Analysis

3.1.1 Correlation and Attributes
Figure 8 depicts the correlation matrix of the dataset's attributes. The correlation matrix depicts the relationship between characteristics. The group is the most important factor to consider while diagnosing dementia. A group value larger than 0.5 is found in dementia patients. The higher the ASF and SES levels, the greater the risk of dementia, according to the correlation matrix. Males, on the other hand, are more likely than females to get dementia.
 
 

                                        Figure 8: Correlation and Attribute







This section of the thesis will describe the findings and analyze the results of the research. 4.1 Result Analysis of Heart Attack Prediction Using Multiple ML Algorithms In this paper, the chance of a heart attack is predicted using different models. On Jupyter Notebook which operates under Ubuntu 64 bits and is composed of a single core hyper threaded Intel Xeon processor@2.3 GHz and 8 GB of RAM, I performed all the computations. As Python programming has open-source packages, I used those to simulate our code and experiments. This work has used confusion matrices such as accuracy, sensitivity, specificity, and F1-score for the XGBoost Classifier and KNN algorithm. Accuracy is the percentage of total subjects classified correctly. Sensitivity is the proportion of those who do have the disease who test positive. Specificity is the proportion of those who do not have the disease who test negative. Sensitivity can also be identified as Recall. Precision is the number of subjects correctly identified as positive out of the total subjects identified as positive. F1-Score is a harmonic mean of precision and recall [42]. 
Accuracy=(TP +TN)/(Data Size)    
Precision=TP/(TP+FP)
RECALL=TP/(TP+FN)           
f1-score=2*(precision*Recall)/(precision+Recall)                                           
 Here, TP and FP denote the number of correctly and wrongly classified subjects having heart disease, respectively. Similarly, TN and FN denote the number of correctly and wrongly classified subjects not having heart disease, respectively [42]. 17 The paper is shown the confusion matrix which is contained the summary of prediction results of all instances for the XGBoost and KNN Classifier of the dataset used for both testing in Fig 4.1 and 4.2 respectively. The paper is also shown the performance of the XGBoost Classifier algorithm in Fig 4.3 and Fig 4.4 respectively.
        Figure 4 1:Confusion Matrix of KNN                     Figure 4 2:Confusion Matrix of XgB

   Figure 4 3:Performance Matrix of XGB                                  Figure 4 4: Performance Matrix of KNN

3.2 Accuracy  Comparison of All Models
 In the paper, the heart attack rate is predicted for some different models. We have used Logistic Regression, Decision Tree, Random Forest, Support Vector Classifier, Naive Bayes Classifier, XGBoost Classifier,  and KNeighbors Classifier algorithms to find the accuracy of a heart attack. We have shown the accuracy percentage of different algorithms in Fig 4.3. Here, XGBoost and KNN Classifier give the best performance with 100% and 96%  accuracy among all other algorithms.   
     
                                     
Figure 4 5:Accuracy Comparison

3.3 Feature Importance 
A heart attack may vary depending on various circumstances. In machine learning, these circumstances are defined as features. The features that have the biggest impact on predictions are called feature importance. In some cases, a few attributes may decrease the accuracy level of a model. So, it is important to work with the correct attributes. Though we have used all the features, feature importance from the dataset is shown in Fig 4.4. As shown in Fig 4.4, the feature cp which is chest pain has the most impact on the chances of having a heart attack. 
We have taken a local example here. We can see that the top 4 features contributing to the prediction of the feature are old peak, ca, cp, and that.
Oldpeak, which is the most important feature in this prediction is related to the electrocardiograph results of the patient. If the electrocardiograph shows an ST Depression (indicating that the patient is suffering from myocardial ischemia), the value of the old peak shows us the level of the ST Depression. The more the depression, the likelier it is that the patient is suffering from heart disease.

 
Figure 4 6:Important Features of The Dataset
3.5 Feature Analysis 
All the features have value to count. By analyzing all the value counts we predict the rate of a heart attack. In the paper, Fig 4.5 represents the more and less chances of heart attack from value counts. It means the dataset has 51.32% of ‘1’ which predicts more chances of heart attack and 48.68% of ‘0’ which predicts fewer chances of a heart attack. All the features have value to count. By analyzing all the value counts we predict the rate of a heart attack.
 

Figure 4 7:Value Counts of The Feature

3.6 Result Analysis of the Heart Attack Prediction Using LIME 
This section of the paper will explain heart attack prediction through the KNN Machine Learning algorithm using LIME. 
3.7 Feature Importance 
Which features give more contribution to the prediction, LIME method can give a visual representation of it. Using LIME is so beneficial because it can provide the feature importance by using 2 different methods. By using the show in a notebook() method Fig 4.6 and 4.7 shows the volume of the feature's impact. The figure consists of 3 types of representation- progress bar, bar chart, and table. In the figure, the progress bar indicates the range of which value varies and the actual prediction; the bar chart shows the features of their weights positive and negative to prediction; and the table represents the feature's importance by showing the actual feature values. Here, the orange color indicates the positive contribution and the blue color indicates the negative contribution toward the prediction [41].
 
Figure 4 8:Feature Importance Using LIME
In fig-4.6, for this patient, heart disease is predicted for orange features which are important features for heart disease whereas blue features are less important.
 
Figure 4 9:Feature Importance Using LIME
In fig-4.7, for this patient, heart disease is not predicted for blue features which are not important features for heart disease whereas orange features are more important.
3.8 Retrieve Features Importance
 LIME XAI method has the advantage of retrieving the features' importance shown in Fig 4.8. Here the first value of the tuple is condition and the second value is the feature value based on condition.
 
Figure 4 10:Retrieving Features Using LIME
 As we know, the KNN algorithm is a regression model. But this algorithm is also defined as a classifier. So if we want to retrieve the feature's importance for classifier task LIME will allow us to do so. Fig 4.8 shows that it returns a dictionary where the key is each class of task and the value is a list of feature indexes and their contribution in predicting that class.

# Viva Questions and Answers

**1. What is machine learning?** Teaching a computer to learn patterns from data and make predictions without writing fixed rules.

**2. What is supervised learning?** Learning from data that already has the correct answers (labels). Ours has Pass/Fail for every student.

**3. Why is this a classification problem?** The output is a category (Pass or Fail), not a continuous number.

**4. What is the target variable?** `result`: 1 = Pass, 0 = Fail.

**5. What are features?** Input columns used for prediction: study hours, attendance, previous score, assignment score, sleep hours.

**6. Why Logistic Regression?** It is the standard simple model for binary classification and gives the probability of passing. It also performed best in our test.

**7. Why Decision Tree?** Easy to understand and visualise; it works like if-else rules. It is a good baseline, but it overfits.

**8. Why Random Forest?** It combines many trees, so it is more stable and accurate than a single tree.

**9. What is overfitting?** The model memorises training data and performs badly on new data. Our Decision Tree had 100% training accuracy but 70% test accuracy.

**10. What is underfitting?** The model is too simple to capture the pattern, so it performs badly on both training and test data.

**11. What is train-test split?** Dividing data into a training part (to learn) and a test part (to check on unseen data).

**12. Why an 80/20 split?** It is a common standard: enough data to learn from (1200) and enough to test fairly (300). We used `stratify` so both parts keep the same pass rate.

**13. What is accuracy?** Correct predictions divided by total predictions. Ours (best model): 77.33%.

**14. What is precision?** Of the students predicted to pass, how many really passed. TP / (TP + FP).

**15. What is recall?** Of the students who really passed, how many we found. TP / (TP + FN).

**16. What is F1-score?** The balance (harmonic mean) of precision and recall.

**17. What is a confusion matrix?** A 2x2 table of TN, FP, FN, TP showing where the model is right and wrong.

**18. What is ROC-AUC?** The ROC curve plots true positive rate against false positive rate; AUC is the area under it. 0.5 = random guessing, 1.0 = perfect. Ours: 0.8604 for Logistic Regression.

**19. Why is Random Forest useful?** Many trees voting reduce overfitting and give more reliable results than one tree.

**20. What is feature importance?** A score showing how much each feature helps the model. In our Random Forest, Previous Exam Score was most important and Sleep Hours least.

**21. Why not Linear Regression?** It predicts continuous values and can give outputs below 0 or above 1; Logistic Regression is made for Pass/Fail.

**22. What is data leakage and how did you avoid it?** Information from test data influencing training. We split first, and the StandardScaler is inside a Pipeline so it is fitted on training data only.

**23. Why did you scale data for Logistic Regression only?** It is sensitive to feature scale; trees split on thresholds, so scaling does not matter for them.

**24. How was the best model chosen?** Automatically by the highest ROC-AUC on the test set, with F1-score as the tie-breaker.

**25. Is your dataset real?** No, it is synthetic (seed 42), with a realistic but noisy relationship between features and result. Results demonstrate the method, not real-world accuracy.

**26. How does the app predict?** It loads `best_model.joblib` (pipeline with scaler + model), takes the five inputs and shows the pipeline's real `predict_proba` probability.

**27. How could you improve the model?** Real data, more features, cross-validation, hyperparameter tuning, other models.

## Short explanation to say in the viva
"My project predicts whether a student will pass or fail using study hours, attendance, previous score, assignment score and sleep hours. It is a supervised binary classification problem. I generated a reproducible dataset of 1500 students, cleaned it, did EDA, and split it 80/20. I trained Logistic Regression, Decision Tree and Random Forest and evaluated them on unseen test data with accuracy, precision, recall, F1, confusion matrices and ROC curves. Logistic Regression was selected automatically as the best model with ROC-AUC 0.86 and 77% accuracy; the Decision Tree overfitted. The best model is saved with joblib and used in a Streamlit app to predict new students with probability."

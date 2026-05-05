# *Data Mining Course, 2026 Spring*

# **Assignment 2**

# ---

# April 09, 2026 

# **Overview**

In Assignment 2, we have **5 questions**. The first question is a continuation of Assignment 1\. For the remaining questions (2–5), we will analyze a new dataset “mobile\_price.csv”, which can be downloaded from E3.

# **Instructions**

Please submit your solution by **May 7, 11:59 PM.**   
**Late submissions are not accepted.** You can use any third-party libraries or AI tools to assist your work, **but if plagiarism or cheating on assignments, the grade will be 0\.**

**Submissions consist of two parts:** 

1. **Code**   
   1. Please host your **code on a public GitHub repository** (commit before deadline).   
2. **Report.**  
   1. Please ensure your public **GitHub link is included** within your report, and submit your **report via E3.**  
   2. **File name:** DM\_asg2\_{studentID}.pdf (e.g., DM\_asg2\_314551051.pdf)

Please write down your report in English. The TA will test whether your code can run successfully.

---

**1\. K-fold Cross-Validation  (15 points):**

| For Question 1, please continue using Real\_World\_Classification.ipynb. Before starting: First, please update linear\_model.py by downloading or replacing it with the latest version from [https://github.com/NYCU-Data-Mining/DM2026-Assignment-1/blob/main/model/linear\_model.py](https://github.com/NYCU-Data-Mining/DM2026-Assignment-1/blob/main/model/linear_model.py).  For this question, please use your preprocessed data from Assignment 1, Question 3\. Do not change the random\_seed to ensure consistency in evaluation results. |
| :---- |

1. **K-fold Cross-Validation:** Use scikit-learn to implement 5-fold cross-validation on the training data to determine the optimal hyperparameters.   
* Apply L2 regularization  
* Evaluate the following hyperparameter values (**16 combinations** in total)  
  * Learning rates: {0.005, 0.01, 0.1, 0.5}  
  * Regularization parameters (reg\_lambda in the code): {1.0, 2.0, 4.0, 8.0}  
* For each combination, compute the **average accuracy** over the 5 folds on the **training data**, and present the results in a 4×4 table.

\[Hint\]

* 5-fold cross-validation should be used on the training data.  
* Please use **sklearn.model\_selection.KFold** and  **sklearn.model\_selection.cross\_val\_score**: [https://scikit-learn.org/stable/modules/cross\_validation.html\#computing-cross-validated-metrics](https://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics)  
* Please fix the cv parameter in **sklearn.model\_selection.cross\_val\_score** like this:

  cross\_val\_score(cv=KFold(n\_splits=5, shuffle=True, random\_state=40),...)

2. **Following 1 (a),**  select the top two hyperparameter settings based on the training data (two hyperparameter combinations with the highest average accuracy). Use these two settings, train the model on the training data and evaluate it on the **testing data**. Report the **testing results** using the **“evaluate\_binary\_classifier”** function.

\[Hint\]

* Each evaluation (shown in a screenshot) contains the metrics:  Accuracy, Precision, Recall, F1 score.  
    
3. **Describe your observations in 1(a) and 1(b).**

---

| For Questions 2–5, please refer to the 'mobile\_price.csv' dataset available on E3.  This dataset maps various hardware specifications (features *X*, e.g., columns “battery\_power”, “ram”, etc.) to a mobile phone’s price range (target *y,* column *“*price\_range”). The column *“*price\_range” is categorized from 0 to 3, representing a scale from 'low cost' to 'very high cost.'  The objective is to use the mobile phone’s feature set *X* to accurately predict the target *y* (the price range, column *“*price\_range”). |
| :---- |

**2\. SVM  (15 points):**

1. Shuffle the data (“mobile\_price.csv”) using a random seed 42, and split it into training, validation, and testing sets with a ratio of 60% / 20% / 20% ratio. Then, train a SVM classifier with the regularization parameter **C= 1.0**. Report the accuracy and F1-score on training, validation and testing data, respectively.

\[Hint\]

* Use **random\_state** in **sklearn.model\_selection.train\_test\_split**  
* SVM classifier please use **sklearn.svm.SVC**


2. Explore different values of the regularization parameter C \= {0.001, 0.01, 0.1, … 100, 1000, 10000}. For each value of C, train the SVM model and evaluate its performance on the training, validation, and test sets using both accuracy and F1-score. Visualize the results to show how performance changes across different values of C.

\[Hint\]

* The results can be visualized using line plots, bar charts, or any other clear comparative visualization.


3. Following 2 (b), based on your observations, determine which model (which value of C) provides the best generalization performance, and explain your reasoning.

**3\. Association Rule Mining  (15 points):** 

| This question aims to use the FP-growth algorithm to analyze the relationship between selected features and the label in the dataset “mobile\_price.csv”. First, filter the dataset by selecting only the data samples where “price\_range” \= 1. Then, focus on the following four feature columns:	{“ram” , “int\_memory”, “px\_width”, “battery\_power”} For each of these features: Divide the values into three categories:  low, medium, and high First determine each feature’s value range (max \- min), and then divide the range into three intervals using a 3:4:3 ratio: low (bottom 30%), medium (middle 40%), and high (top 30%). After categorization, convert each data sample into a transaction format.  For example:For a data sample with “price\_range” \= 1, its original features and categorization results are:  ram int\_memory px\_width batter\_power original features 2549 7 756 842 categorization high low low low are then converted into the following transaction record: ram\_high, int\_memory\_low, px\_width\_low, batter\_power\_low  Please use the resulting transaction records to answer Question 3(a)-(c). |
| :---- |

1. Please list all frequent patterns showing support ≥ 0.3 using FP-growth.

\[Hint\]

* Please use the package **mlxtend**: [https://rasbt.github.io/mlxtend/user\_guide/frequent\_patterns/fpgrowth/](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/)

2. Following 3(a), please list all the association rules where support ≥ 0.3, confidence ≥ 0.4 and lift ≥ 0.8

3. Please describe your observations in 3(a) and 3(b).

 

**4\. PCA and K-Means (20 points):**

1. Split the data (“mobile\_price.csv”) into features and labels. Standardize the features using z-score standardization.

\[Hint\]

* Use  **sklearn.preprocessing.StandardScaler.**


2. Following 4(a), project the features onto 2 dimensions using PCA and visualize the scatterplot for the first two principal components of the data.

\[Hint\]

* Us **sklearn.decomposition.PCA**  
* Ensure that the scatter plot includes a legend indicating the class labels.

3. Following 4(a), apply K-means to cluster the data samples into 4 clusters **using all features**. Visualize the clustering results in a scatter plot (including a legend), and report the clustering performance.

\[Hint\]

* Use **sklearn.cluster.KMeans**  
* Use **sklearn.metrics.cluster.adjusted\_rand\_score** for evaluating clustering performance**.**


4. Following 4(b), apply K-means to cluster the data samples into 4 clusters using the **2-dimensional features obtained from PCA**. Visualize the clustering results with a scatter plot (including a legend), and report the clustering performance.

5. Please describe your observations in 4(c) and 4(d).

**5\. Enhancing K-means Clustering with Association Rule Mining (35 points)**.

| \[20260422\] Grading Criteria Points 0-10: Required results are incomplete (comparison between original K-means and your method)OR The model design description is not provided. Points 10–20: Required results are fully provided& Include a brief description of your model design. Points 20–35: Required results are fully provided& Include additional experimental analysis, a framework diagram, or other detailed analyses to support your model design. |
| :---- |

Please propose a method to improve the performance of **K-means using association rule mining**, and implement your approach **using all features** in the data (“mobile\_price.csv”).

Clearly describe your design in detail and also your inspirations (e.g., inspired by which techniques, papers, or data observations), and compare the performance of the original K-means and your improved method in terms of accuracy, precision, recall, and F1-score. 

To ensure the validity of your results, please report the **average performance** of **both the original K-means and your method** using the same set of random seeds by setting the random\_state in K-means:

seeds \= \[0, 10, 42, 100, 999\].

Also, you are free to draw your designed framework and provide any other more detailed analysis results.

# 

# **Reference** 

1. Codes of Question 1 are modified from: [https://github.com/anhquan0412/basic\_model\_scratch/tree/master](https://github.com/anhquan0412/basic_model_scratch/tree/master)  
2. Some questions of this assignment are based on concepts from Jinbo Shang, DSC148 [HW2](https://www.dropbox.com/scl/fi/8xvrdt5aohiajr3udmi1a/DSC148_W25_HW2.pdf?rlkey=9dpyp8xen5nz2ilhokivvv8fe&e=1&dl=0) and [HW3](https://www.dropbox.com/scl/fi/yq698t46zj5c1ab6u33rx/DSC148_W25_HW3.pdf?rlkey=ifqlupogtmm60k255200qpm5b&e=1&dl=0)  
3. The raw data of the mobile dataset is from:  
   [https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data?select=train.csv](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data?select=train.csv)

# **FAQ**

Below are some common questions and answers. If you have any other questions, please feel free to contact the TA at [nycu.dm.ta@gmail.com](mailto:nycu.dm.ta@gmail.com).

## 
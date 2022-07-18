# Twitter-sentimental-analysis-using-machine-learning
# **Introduction**
Sentiment analysis refers to identifying as well as classifying the sentiments that are expressed in the text source. Tweets are often useful in generating a vast amount of sentiment data upon analysis. These data are useful in understanding the opinion of the people about a variety of topics.

Therefore we need to develop an Automated Machine Learning Sentiment Analysis Model in order to compute the customer perception. Due to the presence of non-useful characters (collectively termed as the noise) along with useful data, it becomes difficult to implement models on them.

we aim to analyze the sentiment of the tweets provided from the Sentiment140 dataset by developing a machine learning pipeline involving the use of three classifiers (Logistic Regression, Bernoulli Naive Bayes, and SVM) along with using Term Frequency- Inverse Document Frequency (TF-IDF). The performance of these classifiers is then evaluated using accuracy and F1 Scores.

# **Problem Statement**
In this project, we try to implement a Twitter **sentiment analysis** model that helps to overcome the challenges of identifying the sentiments of the tweets. The necessary details regarding the dataset are:

The dataset provided is the Sentiment Dataset which consists of 50,000 tweets that have been extracted using the Twitter API. The various columns present in the dataset are:

* **target**: the polarity of the tweet (positive or negative)
* **ids**: Unique id of the tweet
* **date**: the date of the tweet
* **flag**: It refers to the query. If no such query exists then it is NO QUERY.
* **user**: It refers to the name of the user that tweeted
* **text**: It refers to the text of the tweet


# **Project pipeline**

The various steps involved in the **Machine Learning Pipeline** are :

* Import Necessary Dependencies
* Read and Load the Dataset
* Exploratory Data Analysis
* Plotting data
* Data Preprocessing
* Splitting our data into Train and Test Subset
* Transforming Dataset using TF-IDF Vectorizer
* Function for Model Evaluation
* Model Building
* Conclusion


# **Plotting data**
![image](https://user-images.githubusercontent.com/103111784/179452364-d8239eaa-4d79-4942-ae1a-88e4c04ab3a7.png)

**Now we will check the distribution of length of the tweets, in terms of words, in both train and test data.**
![image](https://user-images.githubusercontent.com/103111784/179452450-e74a97de-84dd-433b-92ab-c3a2a1628a61.png)

**Now let us plot a bar graph positive and negative tweets according word count in the tweet**
![image](https://user-images.githubusercontent.com/103111784/179452499-cc6e8f05-bb42-475c-8d85-6002f580329d.png)

# **Understanding the impact of Hashtags on tweets sentiment**
# **Non-Racist/Sexist Tweets**
![image](https://user-images.githubusercontent.com/103111784/179452605-d7b94e9b-5d1e-4260-974b-3f0727ee07f4.png)
      
# **Racist/Sexist Tweets**
![image](https://user-images.githubusercontent.com/103111784/179452716-9f95a4a1-3bed-44b1-8ec0-3966acc6c7cd.png)
      
# **CLEANING TWEET DATA**

  * Removing Twitter Handles (@user)
  * Removing Punctuations, Numbers, and Special Characters
  * Removing Short Words
      	
# **Text Normalization**

# **WORD CLOUD**
# **plot a cloud of word for top 100 words in the tweets**
![image](https://user-images.githubusercontent.com/103111784/179453114-f4649f4a-1b61-4699-b92e-530862b2247d.png)

# **plot a cloud of word for all the words in the tweets**
![image](https://user-images.githubusercontent.com/103111784/179453159-c7aa41f2-3724-46f5-97ee-f537e910c412.png)

# **Plot a cloud of words for positive tweets**
![image](https://user-images.githubusercontent.com/103111784/179453235-58182848-33a6-42d7-a8fc-fc2d9ca7e160.png)

  
# **Plot a cloud of words for negative tweets**
![image](https://user-images.githubusercontent.com/103111784/179453303-aa38f922-7fed-41cd-8453-7cdb8db47f58.png)

# **Transforming Dataset using TF-IDF Vectorizer**
Function For Model Evaluation


# **Model Building **
# **Model1:Bernoulli Naive Bayes model**
![image](https://user-images.githubusercontent.com/103111784/179453579-be2e935e-f722-494c-84b8-ad3fdb795fc8.png)


**Plot the ROC-AUC Curve for model-1**
![image](https://user-images.githubusercontent.com/103111784/179453639-a1990814-33eb-454c-9c52-b5decc38a701.png)


# **Model2: SVM (Support Vector Machine) model**

![image](https://user-images.githubusercontent.com/103111784/179453799-83f5308b-d5c9-4f59-9d6c-efadd2346668.png)



**Plot the ROC-AUC Curve for model-2**

![image](https://user-images.githubusercontent.com/103111784/179453843-0b264574-f1b1-454e-8e7d-55af0ea714e6.png)



# **Model-3 Logistic Regression model**
![image](https://user-images.githubusercontent.com/103111784/179453898-3a40f4a8-48ed-4414-80e6-331fc64c1f74.png)


**Plot the ROC-AUC Curve for model-3**
![image](https://user-images.githubusercontent.com/103111784/179453930-5f8e149a-21d1-4101-bc97-ce46fe6b1b73.png)

      
# **conclusion:**
Upon evaluating all the models we can conclude the following details i.e.
**Accuracy:**As far as the accuracy of the model is concerned Logistic Regression performs better than SVM which in turn performs better than Bernoulli Naive Bayes. 
**F1-score:** The F1 Scores for class 0 and class 1 are :
(a) For class 0: Logistic Regression (accuracy = 0.98)< Bernoulli Naive Bayes(accuracy = 0.98) < SVM (accuracy =1.00)

(b) For class 1: Logistic Regression (accuracy = 0.65)< Bernoulli Naive Bayes (accuracy = 0.69) < SVM (accuracy = 0.98)
**AUC Score:** All three models have the same ROC-AUC score. 
We, therefore, conclude that the SVM is the best model for the above-given dataset.
In our problem statement, **SVM Support-vector machine**is following the principle of**Occam’s Razor**which defines that for a particular problem statement if the data has no assumption, then the simplest model works the best. Since our dataset does not have any assumptions and SVM is a simple model, therefore the concept holds true for the above-mentioned dataset




# NLP-basics
Covered the basics of NLP(text classification)

# The Dataset
The "Large Movie Review Dataset" is used in this project. The dataset consists of 50,000 reviews from IMDB on the condition there are no more than 30 reviews per movie. The dataset is balanced, it has 25k positive, 25k negative sentiments. It is also evenly divided into training, testing sets(25k each)

# Preprocessing
The data is cleaned by removing HTML tags, non alphanumeric characters, extra whitespaces. Stopwords are also removed. The words are also lemmatized(pos = v). Sentiments are converted from positive to numeric(1,0). Train,test sets are seperated as well. 

# Environment
Language : Python

Library : TensorFlow

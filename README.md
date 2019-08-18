# Yelp review sentiment

## Purpose

My goal was to try to classify Yelp Review sentiment based on the content of the review. The number of stars the review
gave will be the label in determining the sentiment of the review. This project was divided into two parts:

1) Determing if the VADER Sentiment method add enough predictive power to the classification without taking too much
computer time.

2) Seeing the type of performance improvements we can achieve from PySpark versus Spark with Scala.

## Tools/Dependencies

- Python 3.7.4
- Scala 2.11.12
- Spark 2.4.3
- Scrapy
- AWS EC2, EMR, and S3

## Dataset

Dataset came from [Kaggle](https://www.kaggle.com/yelp-dataset/yelp-dataset). I used the Review Dataset because I was
only interested in the reviews. In this dataset, there was **6,685,900** rows with columns:

- Review Id
- Business Id
- User Id
- Review
- Stars
- Cool
- Funny
- Useful
- Date

However, we will only be looking at Review Id, Review, and Stars.

## Vader vs No Vader

VADER is a lexicon and rule-based sentiment tool that is specifically made for social media. You can see
[here](https://github.com/cjhutto/vaderSentiment) from the inital repository how it is all calculated. One of the
biggest draws from VADER is how quick it can be caluculated. Since it is rule based, we do not have to train a model
to find the score. My **hypothesis** is that we will be able to apply the score, it will provide a big enough lift to
justify taking an extra step to apply the score to our dataset.

All of my testing was done at my school's [cluster](arc.insight.gsu.edu). This was so that we can maintain consistency
for this particular instance of the project. I created two PySpark programs that are essentially the same, except one
will add in VADER. I will be utilizing Spark's ML library and use the Pipeline method to perform Tokenization, Tf-IDF,
stop word removals, and applying the Logistic Regression model.

It turns out that by adding VADER to the process, we increased our time for the program to rund by 50%. Without VADER,
the cluster was able to run my program in ~40 minutes. With VADER, it increased to ~60 minutes. The gains we achieved
from adding VADER to the model was also miniscule, 90.30% to 90.45% in accuracy. In some cases, that type of lift could
be worth the investment, but for my purposes I decided to procede without adding the VADER step.

## Streaming

I was also curious in how Streaming data would work in a Pipeline containing Spark, so I decided to try to see how my
model could be used in practice. I saved the model from the sentiment classification model that did not contain VADER.
I created a spider with Scrapy to find reviews on Yelp and saved it as a CSV file. PySpark was able to score the model
live.

## Python vs Scala

When learning about PySpark, I kept [hearing](https://www.kdnuggets.com/2018/05/apache-spark-python-scala.html) that
using Scala and Spark provided a superior performance. I was curious in how much of a performance increase there would
be with my particular project.

To keep things simple, I was just looking to refactor my final code from the Non Vader version. However, I couldn't
keep everything consistent. I was not able utilze my school's cluster due to maintenance over the summer, so I decided
to use Amazon Web Service's Elastic Map Reduce tool to perform all of my code. I had 3 instances of m5.xlarge from EC2
running, with one being the Master and the other 2 being the Slave nodes. This required me to rerun the model on this
particular cluster to have an apple to apple comparisons.

I first tested on a sample of 100 of the rows. The Scala code was able to run everything in 36 seconds and the PySpark
ran in 45 seconds. This is far from the promised "10 times faster", but I thought maybe the whole dataset will provide
the bigger picture. The strangest thing is that the PySpark code ran in **631** seconds, whereas the Scala code ran in
**794** seconds. More research will be needed to determine why this is the case. But as of right now, the conclusion of
this particular project is that PySpark runs just as well as Scala and Spark.

## Next Steps

There are some things that I wish to expand in the future:

- For streaming, I wanted to impement a dynamic model that will adjust based on new data that I stream in.
- Find out which part of the Scala code that performed the worst compared to the PySpark counterpart.
  - Also to get better with Scala. After this project, I realize there is so much more to learn from this language.

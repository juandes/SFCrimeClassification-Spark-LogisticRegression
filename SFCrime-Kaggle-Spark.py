from datetime import datetime
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SQLContext

conf = SparkConf().setMaster("spark://Juande.local:7077").setAppName(
    "SFCrime-Kaggle"). \
    set("spark.executor.memory", "2g")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# Import both the train and test dataset and register them as tables
train = sqlContext.read.format('com.databricks.spark.csv').options(
    header='true') \
    .load('train.csv')
train.registerTempTable('train')

test = sqlContext.read.format('com.databricks.spark.csv').options(
    header='true') \
    .load('test.csv')
test.registerTempTable('test')

# Get all the unique categories and add them to a dictionary
crimeCategories = sqlContext.sql(
    'SELECT DISTINCT Category FROM train').collect()
crimeCategories.sort()

categories = {}
for category in crimeCategories:
    categories[category.Category] = float(len(categories))

# HashingTF transforms a string into a numerical value	
htf = HashingTF(5000)

# Create LabeledPoint object
trainingData = train.map(lambda x: LabeledPoint(categories[x.Category],
                                                htf.transform(
                                                    [x.DayOfWeek, x.PdDistrict,
                                                     datetime.strptime(x.Dates,
                                                                       '%Y-%m-%d %H:%M:%S').hour])))
# Train the model
logisticRegressionModel = LogisticRegressionWithLBFGS.train(trainingData,
                                                            iterations=100,
                                                            numClasses=39)

# Prepare the testting dataset
testingSet = test.map(lambda x: htf.transform([x.DayOfWeek, x.PdDistrict,
                                               datetime.strptime(x.Dates,
                                                                 '%Y-%m-%d %H:%M:%S').hour]))

# Predict using the day of the week and the police district and hour of crime
predictions = logisticRegressionModel.predict(testingSet)
predictions.saveAsTextFile('predictionsSpark')
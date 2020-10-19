#RDD is Resilient Distributed Dataset
#Transformation is a Spark Operation that produces an RDD
#ACtion is Spark Operation that produces a local object
#Spark Job is a sequence of transformations on data with a final action

from pyspark import SparkContext
sc=SparkContext()
textFileRdd = sc.textFile('notes.txt')

words = textFileRdd.map(lambda x:x.split())#This is an RDD operation and it does not execute itself unless an action is performed.
print(words)
print(words.collect())#This is an RDD action and the RDD operation in previous step gets executed only when this action is performed.

flatWords = textFileRdd.flatMap(lambda x:x.split()).collect() #here we are merging 2 steps i.e., transformation is first performed on textFileRdd
#and then the action. Both are done in a single step.

print(flatWords)

#Let us now see 'Pair RDDs'. Lets use services.txt file for this. Lets grab fields. Lets get total sales per state in this txt file
textFileRdd2 = sc.textFile('servives.txt')
#print(textFileRdd2.collect())
#words2=textFileRdd2.map(lambda line:line.split()).take(2)
#print(words2)
#textFileRdd2.map(lambda line: line[1:] if line[0] == '#' else line).collect()


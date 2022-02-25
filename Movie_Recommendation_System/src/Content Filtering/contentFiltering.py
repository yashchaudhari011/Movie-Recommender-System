import pandas as pd
from math import sqrt
import numpy as np
import random
from itertools import combinations
from collections import defaultdict
from pyspark.sql import SparkSession
import sys

if len(sys.argv) > 1:  
        print("Usage: spark-submit contentFilerting.py.py", file=sys.stderr)
        sys.exit(-1)

spark = (SparkSession.builder.appName("Content Filtering").getOrCreate()) # Establishes the Spark Session
sc = spark.sparkContext
sc.setLogLevel('WARN')

# ratings_rdd = sc.textFile("finalproject/input/u.data").map(lambda x : x.split())
movies_rdd = sc.textFile("finalproject/input/u.item").map(lambda x : x.split('|'))
genre_data = sc.textFile("finalproject/input/u.genre").map(lambda x : x.split('|'))

# (movie,[genre list])
movies_genre_list = movies_rdd.map(lambda x : (x[0],x[5:]))

# building a user profile
# profile consist of a list of movieID, movieName and the ratings given by the user
userInput = [('1','Toy Story', '3.5'),('2','Jumanji', '2'),('296',"Pulp Fiction", '5'),('1274','Akira', '4.5'),('1682','Scream of Stone', '5')]
userInput_rdd = sc.parallelize(userInput)
movieIds = userInput_rdd.map(lambda x : x[0]).collect()

# we extract the data from the above step so that we can only the take the movies which are taken by the users
up_movies_genre_list = movies_genre_list.filter(lambda x : x[0] in movieIds)

genre_list = np.array(up_movies_genre_list.map(lambda x : x[1]).collect())
genre_list = genre_list.astype(np.float)

rating_list = np.array(userInput_rdd.map(lambda x : [x[2]]).collect())
rating_list = rating_list.astype(np.float)

userProfile = np.dot(genre_list.transpose(),rating_list)

# calculate weighted average of each movie in accordance with with the user profile
def predicted_rating(movie_genres, userProfile):
    score = 0.0
    userProfileSum = 0.0
    for i in range(0,len(movie_genres)):
        score += float(movie_genres[i]) * float(userProfile[i][0])
        userProfileSum += float(userProfile[i][0])
    
    return score/userProfileSum if userProfileSum else 0.0

recommended_movie_rating_rdd =  movies_genre_list.map(lambda x : (x[0], predicted_rating(x[1],userProfile)))

# set the recommendations in ascending=False
sortedMovieList = recommended_movie_rating_rdd.sortBy(lambda x: x[1],ascending=False).collect()
final_movie_list = movies_rdd.filter(lambda x: x[0] in (a[0] for a in sortedMovieList[:20])).map(lambda x : (x[0],x[1],list(x[5:]))).collect()

genre_data = genre_data.map(lambda x : x).collect()

# final recommendation 
print("The recommended list is as follows: ")
for (movieId,title,genre_list) in final_movie_list:
    genreArray=[]
    for i in range(0,len(genre_list)):
        if genre_list[i] == '1':
            genreArray.append(genre_data[i][0])
    genreArray = ','.join(genreArray)
    print("Movid ID: "+movieId+"\tTitle: "+title+"\tGenres: "+genreArray+"\n")

sc.stop()
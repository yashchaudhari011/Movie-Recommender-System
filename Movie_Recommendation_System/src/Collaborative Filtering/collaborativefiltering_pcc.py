import random
import numpy as np
from itertools import combinations
from collections import defaultdict
from pyspark.sql import SparkSession
import sys

if len(sys.argv) == 1:  
        print("Usage: spark-submit collabarative_filtering.py <USER ID>", file=sys.stderr)
        sys.exit(-1)

spark = (SparkSession.builder.appName("Colab Filtering").getOrCreate()) #Establishes the Spark Session
sc = spark.sparkContext
sc.setLogLevel('WARN')
userId = sys.argv[1]

lines = sc.textFile("finalproject/input/u.data")
#(movie,[(user,rating)]) --> contains list of user,rating pairs
movie_user_rdd = lines.map(lambda x: (x.split()[1],(x.split()[0],float(x.split()[2])))).groupByKey()

#(movie,[(user,rating),(user,rating)])) --> contains list of user,rating pairs 
movie_user_rdd = movie_user_rdd.map(lambda x : (x[0], list(set(x[1]))))#.filter( lambda p: len(p[1]) > 1)

#(movie,[(user,rating),(user,rating)]) where [(user,rating)] > 1
user_pair_rdd = movie_user_rdd.filter( lambda p: len(p[1]) > 1)

def findUserPair(user_Rating):
    for usr1,usr2 in combinations(user_Rating,2):
        return (usr1[0],usr2[0]),(usr1[1],usr2[1])

# returns (usr1,usr2),((usr1_rating1, usr2_rating2),(usr1_rating2, usr2_rating2))
user_pair_rdd = user_pair_rdd.map(lambda  p: findUserPair(p[1])).groupByKey()

def pcc(user_pair, rating_pairs):
    
    sumX = sumY = avgX = avgY = sumXsq = sumYsq = 0.0
    x = 0
    for ratings in rating_pairs:
        avgX += np.float(ratings[0]) 
        avgY += np.float(ratings[1])

    avgX /= len(rating_pairs)
    avgY /= len(rating_pairs)

    for ratings in rating_pairs:
        sumX += np.float(ratings[0]) - avgX
        sumY += np.float(ratings[1]) - avgY
        sumXsq += (np.float(ratings[0]) - avgX)**2
        sumYsq += (np.float(ratings[1]) - avgY)**2

    numerator = sumX * sumY
    denominator = np.sqrt(sumXsq * sumYsq)

    similarity =  numerator / float(denominator) if denominator else 0.0
    return user_pair, (similarity,x)

# returns cosine similarity for those 2 users ((usr1,usr2),(cosineSimi,count))
user_sim = user_pair_rdd.map(lambda p: pcc(p[0], p[1]))

def FirstUsrKey1(user_pair, similar_movie_data):
    '''
    For each user-user pair, make the first user's id key
    '''
    (usr1_id,usr2_id) = user_pair
    return usr1_id,(usr2_id,similar_movie_data)
# returns (usr1,[(usr2,(cosineSimi,count)),(user3,(cosineSimi,count))])
user_similarity_1=user_sim.map(lambda p: FirstUsrKey1(p[0], p[1])).groupByKey()

def FirstUsrKey2(user_pair, similar_movie_data):
    '''
    For each user-user pair, make the first user's id key
    '''
    (usr1_id,usr2_id) = user_pair
    return usr2_id,(usr1_id,similar_movie_data)
user_sim2=user_sim.map(lambda p: FirstUsrKey2(p[0], p[1])).groupByKey()

user_sim = user_similarity_1.union(user_sim2).filter(lambda x: x[0] == userId)

# returns top 3 users with highest consineSimi usr1,[(usr2,(cosineSimi,count)),(user3,(cosineSimi,count))]
def nearestNeighbour(user, user_similarity, n):
    user_similarity.sort(key=lambda x: x[1][0],reverse=True)
    return user, user_similarity[:n]

user_sim = user_sim.map(lambda x : (x[0], list(x[1]))).map(lambda p: nearestNeighbour(p[0], p[1], 100))

# user,[(movie,ratings),(movie,ratings)]
def getUserDetails(line):
    line = line.split()
    return line[0],(line[1],float(line[2]))
usr_movie_prev_history = lines.map(getUserDetails).groupByKey().collect()

user_dict = {}
for (user,movie) in usr_movie_prev_history:
    user_dict[user] = movie

u = sc.broadcast(user_dict)

def recommendMovie(user_ID, user_Similarity, user_Rating, n):
    t = defaultdict(int)
    sim_s = defaultdict(int)

    usermovieratings = user_Rating.get(user_ID,None)
    userAvgRating = 0.0
    if usermovieratings:
        for (movie,rating) in usermovieratings:
            userAvgRating += rating
        userAvgRating /= len(usermovieratings)

    for (neigh,(sim,count)) in user_Similarity:

        # lookup the movie predictions for this similar neighbours
        unscored_movies = user_Rating.get(neigh,None)
        avg_rating = 0.0
        if unscored_movies:
            for (movie,rating) in unscored_movies:
                avg_rating += rating
            avg_rating/=len(unscored_movies)
            for (movie,rating) in unscored_movies:
                #if neigh != movie:

                    # update totals and sim_s with the rating data
                t[movie] += sim * (rating - avg_rating)
                sim_s[movie] += sim
                

    # create the normalized list of scored movies
    scored_items = [(userAvgRating + (total/abs(sim_s[movie])) if abs(sim_s[movie]) else 0.0 ,movie) for movie,total in t.items()]

    # sort the scored movies in ascending order
    scored_items.sort(reverse=True)

    # take out the movie score
    ranked_items = [x[1] for x in scored_items]

    return user_ID,ranked_items[:n]

user_movie_recs = user_sim.map(lambda p: recommendMovie(p[0], p[1], u.value, 20)).collect()

movieNames= {} 
with open("u.item",encoding='ascii', errors='ignore') as f:
    for line in f:
        fields= line.split('|')
        movieNames[int(fields[0])] = fields[1]
'''
Display the movie recommendation
'''
result= user_movie_recs
movieList = list()
for r in result:
    (user, pair) = r
    for p in pair:
        movieList.append(movieNames[int(p)])

    print("User ",user, "Recomended movies")
    print("\n")
    for x in movieList:
        print(x,"\n")
    del movieList[:]

spark.stop()
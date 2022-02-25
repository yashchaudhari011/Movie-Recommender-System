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
movie_user_pairs = lines.map(lambda x: (x.split()[1],(x.split()[0],float(x.split()[2])))).groupByKey()

#(movie,[(user,rating),(user,rating)])) --> contains list of user,rating pairs 
movie_user_pairs = movie_user_pairs.map(lambda x : (x[0], list(set(x[1]))))#.filter( lambda p: len(p[1]) > 1)

#(movie,[(user,rating),(user,rating)]) where [(user,rating)] > 1
paired_users = movie_user_pairs.filter( lambda p: len(p[1]) > 1)

def findingUserPairs(users_with_rating):
    for user1,user2 in combinations(users_with_rating,2):
        return (user1[0],user2[0]),(user1[1],user2[1])

# returns (user1,user2),((user1_rating1, user2_rating2),(user1_rating2, user2_rating2))
paired_users = paired_users.map(lambda  p: findingUserPairs(p[1])).groupByKey()

def cosineSimilarity(user_pair, rating_pairs):
    sum_x, sum_xy, sum_y = (0.0, 0.0, 0.0)
    x = 0
    for rating_pair in rating_pairs:
        sum_x += np.float(rating_pair[0])**2
        sum_y += np.float(rating_pair[1])**2
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])

    numerator = sum_xy
    denominator = np.sqrt(sum_x) * np.sqrt(sum_y)

    cosine_similarity =  numerator / float(denominator) if denominator else 0.0
    return user_pair, (cosine_similarity,x)

# returns cosine similarity for those 2 users ((user1,user2),(cosine similarity,count))
user_sim = paired_users.map(lambda p: cosineSimilarity(p[0], p[1]))

def FirstUsrKey1(user_pair, movie_sim_data):
    '''
    For each user-user pair, make the first user's id key
    '''
    (user1_id,user2_id) = user_pair
    return user1_id,(user2_id,movie_sim_data)

# returns (user1,[(user2,(cosine similarity,count)),(user3,(cosine similarity,count))])
user_sim1=user_sim.map(lambda p: FirstUsrKey1(p[0], p[1])).groupByKey()

def FirstUsrKey2(user_pair, movie_sim_data):
    '''
    For each user-user pair, make the first user's id key
    '''
    (user1_id,user2_id) = user_pair
    return user2_id,(user1_id,movie_sim_data)
user_sim2=user_sim.map(lambda p: FirstUsrKey2(p[0], p[1])).groupByKey()

user_sim = user_sim1.union(user_sim2).filter(lambda x: x[0] == userId)

# returns top 3 users with highest consineSimi user1,[(user2,(cosine similarity,count)),(user3,(cosine similarity,count))]
def nearestNeighbour(user, users_and_sims, n):
    users_and_sims.sort(key=lambda x: x[1][0],reverse=True)
    return user, users_and_sims[:n]

user_sim=user_sim.map(lambda x : (x[0], list(x[1]))).map(lambda p: nearestNeighbour(p[0], p[1], 100))

# user,[(movie,ratings),(movie,ratings)]
def getUserDetails(line):
    line = line.split()
    return line[0],(line[1],float(line[2]))
user_movie_history = lines.map(getUserDetails).groupByKey().collect()

user_dict = {}
for (user,movie) in user_movie_history:
    user_dict[user] = movie

u = sc.broadcast(user_dict)

def recommendMovie(user_id, user_sims, users_with_rating, n):
    t = defaultdict(int)
    sim_s = defaultdict(int)

    for (neigh,(sim,count)) in user_sims:

        # lookup the movie predictions for this similar neighbours
        unscored_movies = users_with_rating.get(neigh,None)

        if unscored_movies:
            for (movie,rating) in unscored_movies:
                #if neigh != movie:

                    # update totals and sim_s with the rating data
                t[movie] += sim * rating
                sim_s[movie] += sim

    # create the normalized list of scored movies
    scored_items = [(total/abs(sim_s[movie]),movie) for movie,total in t.items()]

    # sort the scored movies in ascending order
    scored_items.sort(reverse=True)

    # take out the movie score
    ranked_items = [x[1] for x in scored_items]

    return user_id,ranked_items[:n]

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

    # print("User", user, "Recomended movies", movieList)
    # del movieList[:]
    
    print("User ",user, "Recomended movies")
    print("\n")
    for x in movieList:
        print(x,"\n")
    del movieList[:]

spark.stop()
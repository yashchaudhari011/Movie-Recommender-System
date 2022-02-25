import os
import sys
import time
import math
from collections import defaultdict
from pyspark.sql import SparkSession

spark = (SparkSession

           .builder

           .appName("Content Based Filtering")

           .getOrCreate())


sc = spark.sparkContext
sc.setLogLevel('WARN')  

#User based recommendation
#two dicts are used, one for similarity sum, one for weighted rating sum
#for every neighbor of a user, get his rated items which hasn't been rated by current user
#then for each movie, sum the weighted rating in the whole neighborhood 
#and sum the similarity of users who rated the movie iterate and sort

#user: id of a user asking for recommendation
#neighbors: [(user_sim, similarity, number of common ratings)]
#usermovHistDict: (user, ([movie], [rating]))
#topK: the number of neighbors to use
#nRec: the number of recommendation

def recommend(user, neighbors, user_movie_history_dict, topK = 200, nRec = 30):
    simSumDict = defaultdict(float)
    weightedSumDict = defaultdict(float)
    movIDUserRated = user_movie_history_dict.get(user, [])
    for (neighbor, simScore, numCommonRating) in neighbors[:topK]:
        mrlistpair = user_movie_history_dict.get(neighbor)
        if mrlistpair:
            for index in range(0, len(mrlistpair[0])):
                movID = mrlistpair[0][index]
                simSumDict[movID] += simScore
                weightedSumDict[movID] += simScore * mrlistpair[1][index]
    candidates = [(mID, 1.0 * wsum / simSumDict[mID]) for (mID, wsum) in weightedSumDict.items()]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return (user, candidates[:nRec])

def eval_user_movie_hist(userRatingGroup):
    userID = userRatingGroup[0]
    movieList = [item[0] for item in userRatingGroup[1]]
    ratingList = [float(item[1]) for item in userRatingGroup[1]]
    return (userID, (movieList, ratingList))

# calculate the average rating of a user
def meanuserRating(userRatingGroup):
    userID = userRatingGroup[0]
    ratingSum = 0.0
    ratingCnt = len(userRatingGroup[1])
    if ratingCnt == 0:
        return (userID, 0.0)
    for item in userRatingGroup[1]:
        ratingSum += float(item[1])
    return (userID, 1.0 * ratingSum / ratingCnt)

def userRatingAverage_broadcast(uRRDDTrain):
    userRatingAvgList = uRRDDTrain.map(lambda x: meanuserRating(x)).collect()
    userRatingAvgDict = {}
    for (user, avgscore) in userRatingAvgList:
        userRatingAvgDict[user] = avgscore
    uRatingAvgBC = sc.broadcast(userRatingAvgDict)
    return uRatingAvgBC

def userMovieHistory_broadcast(uRRDDTrain):
    userMovieHistList = uRRDDTrain.map(lambda x: eval_user_movie_hist(x)).collect()
    userMovieHistDict = {}
    for (user, mrlistTuple) in userMovieHistList:
        userMovieHistDict[user] = mrlistTuple
    uMHistBC = sc.broadcast(userMovieHistDict)
    return uMHistBC

def evalCommonRating(tup1, tup2):
    user1, user2 = tup1[0], tup2[0]
    mrlist1 = sorted(tup1[1])
    mrlist2 = sorted(tup2[1])
    ratepair = []
    index1, index2 = 0, 0
    while index1 < len(mrlist1) and index2 < len(mrlist2):
        if mrlist1[index1][0] < mrlist2[index2][0]:
            index1 += 1
        elif mrlist1[index1][0] == mrlist2[index2][0]:
            ratepair.append((mrlist1[index1][1], mrlist2[index2][1]))
            index1 += 1
            index2 += 1

        else:
            index2 += 1
    return ((user1, user2), ratepair)

# calculate cosine similarity, takes in ((user1, user2), [(rating1, rating2)])
# returns ((user1, user2), (similarity, common ratings))
def cosineSimilarity(tup):
    dotproduct = 0.0
    sqsum1, sqsum2, cnt = 0.0, 0.0, 0
    for rpair in tup[1]:
        dotproduct += rpair[0] * rpair[1]
        sqsum1 += (rpair[0]) ** 2
        sqsum2 += (rpair[1]) ** 2
        cnt += 1
    denominator = math.sqrt(sqsum1) * math.sqrt(sqsum2)
    similarity = (dotproduct / denominator) if denominator else 0.0
    return (tup[0], (similarity, cnt))

def userKey(record):
    return [(record[0][0], (record[0][1], record[1][0], record[1][1])), 
            (record[0][1], (record[0][0], record[1][0], record[1][1]))]

# get the similar users
def retrieveSimilar_user(user, records, numK = 200):
    llist = sorted(records, key=lambda x: x[1], reverse=True)
    llist = [x for x in llist if x[2] > 9]
    return (user, llist[:numK])

# user_neighbour_dict_bc is broadcasted
def neighbourDict_broadcast(uNeighborRDD):
    userNeighborList = uNeighborRDD.collect()
    userNeighborDict = {}
    for user, simrecords in userNeighborList:
        userNeighborDict[user] = simrecords
    uNeighborBC = sc.broadcast(userNeighborDict)
    return uNeighborBC

# movie_name_dict_bc is broadcasted
def movieNameDict_broadcast(movRDD):
    movieNameList = movRDD.map(lambda x : (x[0],x[1])).collect()
    movieNameDict = {}
    for (movID, movName) in movieNameList:
        movieNameDict[movID] = movName
    mNameDictBC = sc.broadcast(movieNameDict)
    return mNameDictBC

# generate movie recommendation name
def movieRecommendationName(user, records, movNameDict):
    nlist = []
    for record in records:
        nlist.append(movNameDict[record[0]])
    return (user, nlist)

rawRatings = sc.textFile('finalproject/input/u.data').map(lambda x : x.split()).cache()
rawMovies = sc.textFile('finalproject/input/u.item').map(lambda x : x.split('|')).cache()
userId = sys.argv[1]

ratingsCount = rawRatings.count()
moviesCount = rawMovies.count()

userRatingRDD = rawRatings.map(lambda x: (x[0], (x[1], float(x[2])))).groupByKey()
userRatingAvgBC = userRatingAverage_broadcast(userRatingRDD)

userMovieHistBC = userMovieHistory_broadcast(userRatingRDD)

cartesianRDD = userRatingRDD.cartesian(userRatingRDD)
userPairRawRDD = cartesianRDD.filter(lambda x : x[0][0] < x[1][0])

userPairRDD = userPairRawRDD.map(lambda x: evalCommonRating(x[0], x[1]))

userSimilarityRDD = userPairRDD.map(lambda x: cosineSimilarity(x))
userSimGroupRDD = userSimilarityRDD.flatMap(lambda x: userKey(x)).groupByKey()

userNeighborRDD = userSimGroupRDD.map(lambda x: retrieveSimilar_user(x[0], x[1], 200))
userNeighborBC = neighbourDict_broadcast(userNeighborRDD)
userRecomMovIDsRDD = userNeighborRDD.map(lambda x: recommend(x[0], x[1], userMovieHistBC.value))

movieNameDictBC = movieNameDict_broadcast(rawMovies)
userRecomMovNamesRDD = userRecomMovIDsRDD.map(lambda x: movieRecommendationName(x[0], x[1], movieNameDictBC.value))

print (userRecomMovNamesRDD.filter(lambda x: x[0] == userId).collect())






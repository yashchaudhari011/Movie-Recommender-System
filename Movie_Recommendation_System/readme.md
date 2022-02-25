Algorithms implemented:

1. Content based filtering using weighted average technique
2. Collaborative filtering using cosine similarity technique
3. Collaborative filtering using pearson correlation coefficient
4. Collaborative filtering using nearest neighbour

Directory structure for final submission:
	src-->
		Content Filtering
			-->contentfiltering.py

		Collaborative Filtering 
			-->collaborativefiltering_pcc.py
			-->collaborativefiltering_cosine.py
			-->collaborativefiltering_nearest_neigh.py
	data-->
		u.data
		u.item
		u.genre

src file location on hadoop cluster:
current directory from where the spark job gets submitted

data file location on the hdfs:
	/user/<user_id>/finalproject/input/u.data
	/user/<user_id>/finalproject/input/u.item
	/user/<user_id>/finalproject/input/u.genre
	
commands to run the programs:
	$spark-submit contentfiltering.py
	$spark-submit collaborativefiltering_pcc.py <user_id>
	$spark-submit collaborativefiltering_cosine.py <user_id>
	$spark-submit collaborativefiltering_nearest_neigh.py <user_id>


	

	
	


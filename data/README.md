File Descriptions: 

movielens_movie.parquet: Movie table. Schema: movie_id, genres, titile
movielens_movie_s_8192.parquet: Movie Table (stored with row_group_size=8192)

movielens_user.parquet: User table. Schema: user_id, gender, occupation, age, zipcode
movielens_user_s_8192.parquet: User Table (stored with row_group_size=8192)

movielens_rating.parquet: User-movie rating table. Schema: movie_id, user_id, rating, timestamp
movielens_rating_s_8192.parquet: User-movie rating table (stored with row_group_size=8192)

movielens_user_rating: Users' average rating table. Schema: user_id, user_average_rating
movielens_movie_rating: Movies' average rating table. Schema: movie_id, movie_average_rating

gen_data.py: Generate query table with given number of samples. Schema: q_user_id, q_movie_id. This file will be invoked in velox to generate data at run-time.
preview_data.py: Print-out the metadata of generated query data.
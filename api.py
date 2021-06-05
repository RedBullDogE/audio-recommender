import pandas as pd
from flask import Flask, request
from audio_recommender import AudioRecommenderSystem

app = Flask(__name__)

# Song metadata [song_id, title, release, artist_name, year]
_song_metadata = pd.read_csv("audio_dataset/song_data.csv")

# Data about user listens [user_id, song_id, listen_count]
_user_listens = pd.read_csv("audio_dataset/10000.csv")

# Data only about songs with user listens
# [song_id, title, artist_name, release, year, listen_count]
song_listens = pd.merge(
    _user_listens,
    _song_metadata.drop_duplicates(["song_id"]),
    on="song_id",
    how="outer",
)

# Extended song data with genres field
_spotify_artists = pd.read_csv("audio_dataset/artists.csv")
_spotify_data = pd.read_csv("audio_dataset/data_o.csv")
_grouped_song_df = (
    song_listens.groupby(["song_id", "title", "artist_name", "release", "year"])
    .agg({"listen_count": "count"})
    .reset_index()
)
song_df = pd.merge(
    _grouped_song_df,
    _spotify_artists,
    left_on="artist_name",
    right_on="name",
    how="inner",
).drop(["id", "year", "popularity", "followers", "name"], axis=1)

song_df.drop_duplicates(subset="title", keep="last", inplace=True)

song_df["genres"] = song_df["genres"].apply(eval).apply(tuple)
_spotify_data["artists"] = _spotify_data["artists"].apply(lambda x: eval(x)[0])

song_features = (
    pd.merge(
        song_df,
        _spotify_data,
        how="left",
        left_on=["title", "artist_name"],
        right_on=["name", "artists"],
    )
    .dropna(subset=["artists"])
    .drop_duplicates(["title", "artist_name"])
)

# Artist data
_artist_features = pd.read_csv("audio_dataset/data_by_artist_o.csv")
artist_df = (
    song_df.groupby(["artist_name", "genres"])
    .agg({"listen_count": "sum"})
    .reset_index()
)

artist_df = pd.merge(
    artist_df.set_index("artist_name"),
    _artist_features.set_index("artists"),
    left_index=True,
    right_index=True,
)

artist_df = artist_df[
    ~artist_df.index.duplicated(keep="last")
]

rs = AudioRecommenderSystem(
    song_df, song_listens, artist_df, song_features
)


@app.route("/top-artists", methods=["GET"])
def top_artists():
    """
    Endpoint with top artists: general and by genres
    """
    genre = request.args.get("genre")
    count = int(request.args.get("count"))

    res = rs.get_top_artists(genre, count)

    return res.to_json(orient="records")


@app.route("/top-songs", methods=["GET"])
def top_songs():
    """
    Endpoint with top songs: general and by genres
    """
    genre = request.args.get("genre")
    count = int(request.args.get("count"))

    res = rs.get_top_songs(genre, count)

    return res.to_json(orient="records")


@app.route("/user-prefered-genres/<user_id>", methods=["GET"])
def prefered_genres(user_id):
    """
    Endpoint for getting user's most preferred genres
    """
    count = request.args.get("count")

    res = rs.get_user_prefered_genres(user_id, count)

    return res.to_json()


@app.route("/similar-songs-by-rating/<song_id>", methods=["GET"])
def similar_songs_cf(song_id):
    """
    Endpoint for getting similar songs with collaborative filtering
    """
    simcoef = request.args.get("simcoef")
    count = request.args.get("count")

    res = rs.get_similar_songs_cf(song_id, simcoef, count)

    return res.to_json(orient="records")


@app.route("/similar-songs-by-features/<song_id>", methods=["GET"])
def similar_songs_cbf(song_id):
    """
    Endpoint for getting similar songs with content-based filtering
    """
    simcoef = request.args.get("simcoef")
    count = request.args.get("count")

    res = rs.get_similar_songs_cbf(song_id, simcoef, count)

    return res.to_json(orient="records")


@app.route("/init-recommendations-by-genres", methods=["GET"])
def init_recommendations_by_genres():
    """
    Get initial recommendation by specified genres
    """
    genres = request.args.get("genres").split(",")
    if len(genres) == 0:
        return app.response_class(
            response="Request should contain genres query param", status=404
        )
    res = rs.get_initial_recommendations_by_genres(genres)

    return res.to_json(orient="records")


@app.route("/init-recommendations-by-artists", methods=["GET"])
def init_recommendations_by_artists():
    """
    Get initial recommendation by specified artists
    """
    artists = request.args.get("artists").split(",")
    if len(artists) == 0:
        return app.response_class(
            response="Request should contain artists query param",
            status=404,
        )

    res = rs.get_initial_recommendations_by_artists(artists)

    return res.to_json(orient="records")


if __name__ == "__main__":
    app.run(port=5000, debug=True, threaded=True)

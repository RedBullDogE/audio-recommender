import pandas as pd
from utils import normalize, to_1D


class AudioRecommenderSystem:
    """
    Main class of audio recommender system
    """

    data_fields = ("song_df", "song_listens", "artist_df", "song_features")

    def __init__(self, song_df, song_listens, artist_df, song_features):
        self.song_df = song_df
        self.song_listens = song_listens
        self.artist_df = artist_df
        self.song_features = song_features

    def reload_data(self, **kwargs):
        """
        Method for updating data
        """
        for field, data in kwargs.items():
            if field in self.data_fields:
                self.__dict__[field] = data
            else:
                raise AttributeError(
                    f"Only attributes can be updated: {', '.join(self.data_fields)}"
                )

    def get_initial_recommendations_by_genres(self, genres: list):
        """
        Get initial recommendations by specified list of genres
        """
        res_count = 25
        genres_count = len(genres)
        songs_by_genre = res_count // genres_count

        res = pd.concat(
            [self.get_top_songs(genre, 100).sample(songs_by_genre) for genre in genres]
        ).sample(frac=1)

        return res

    def get_initial_recommendations_by_artists(self, artists: list):
        """
        Get initial recommendations by specified list of artists
        """
        res_count = 30
        artist_count = len(artists) * 2
        songs_by_artist = res_count // artist_count

        similar_artists = pd.concat(
            [
                self.get_similar_artists(artist, 0.9, n=10).sample(2)
                for artist in artists
            ]
        ).index.tolist()

        res = pd.concat(
            [
                self.get_artist_songs(artist, 20)
                .sample(songs_by_artist, replace=True)
                .drop_duplicates("song_id")
                for artist in similar_artists
            ]
        ).sample(frac=1)

        return res

    def get_user_prefered_genres(self, user_id: str, n=10):
        """
        Genres with most common genres in user's playlist
        """
        if n is None:
            n = 10

        user_song_data = pd.merge(
            self.song_listens[self.song_listens["user_id"] == user_id],
            self.song_df,
            on="song_id",
        )

        counted_genres = to_1D(user_song_data["genres"]).value_counts()
        return counted_genres.head(n)

    def get_top_songs(self, genre=None, n=25):
        """
        Get top songs: general or by specified genre
        """
        if n is None:
            n = 25
        songs = self.song_df

        if genre:
            songs = self.song_df[self.song_df["genres"].apply(lambda x: genre in x)]

        top_songs = songs.sort_values("listen_count", ascending=False)

        return top_songs.head(n)

    def get_top_artists(self, genre=None, n=10):
        """
        Get top artists: general or by specified genre
        """
        if n is None:
            n = 10

        artists = self.artist_df

        if genre:
            artists = self.artist_df[
                self.artist_df["genres"].apply(lambda x: genre in x)
            ]

        top_artists = artists.sort_values("listen_count", ascending=False).reset_index()

        return top_artists.head(n)

    def get_artist_songs(self, artist: str, n=10):
        """
        Get artist's songs
        """
        return self.song_df[self.song_df["artist_name"] == artist].head(n)

    def get_similar_songs_cf(self, song_id, min_sim_coef=0.9, n=25):
        """
        Using collaborative filtering to find similar songs
        """
        if min_sim_coef is None:
            min_sim_coef = 0.9

        if n is None:
            n = 25

        song_rating_matrix = pd.pivot_table(
            self.song_listens[["user_id", "listen_count", "song_id"]],
            values="listen_count",
            index="user_id",
            columns="song_id",
        )

        try:
            target_song_row = song_rating_matrix[song_id]
        except KeyError:
            return pd.DataFrame()

        song_similarity = (
            pd.DataFrame(
                song_rating_matrix.corrwith(target_song_row).drop(song_id),
                columns=["similarity"],
            )
            .reset_index()
            .dropna()
        )

        result = song_similarity.merge(self.song_df).sort_values(
            by="similarity", ascending=False
        )
        result = result[
            (min_sim_coef < result["similarity"]) & (result["similarity"] < 0.9999)
        ]

        return result.head(n)

    def get_similar_songs_cbf(self, song_id, min_sim_coef=0.5, n=25):
        """
        Using content-based filtering to find similar songs
        """
        if min_sim_coef is None:
            min_sim_coef = 0.9

        if n is None:
            n = 25
        cols = [
            "valence",
            "acousticness",
            "danceability",
            "duration_ms",
            "energy",
            "instrumentalness",
            "liveness",
            "loudness",
            "speechiness",
            "tempo",
        ]
        normalized_song_features = normalize(self.song_features, cols)
        song = normalized_song_features[normalized_song_features["song_id"] == song_id]
        corr_vector = normalized_song_features[cols].corrwith(
            song[cols].squeeze(), axis=1
        )

        result = pd.merge(
            self.song_features,
            corr_vector.rename("similarity"),
            left_index=True,
            right_index=True,
        ).sort_values("similarity", ascending=False)[
            ["song_id", "title", "artist_name", "similarity"]
        ]

        result = result[
            (min_sim_coef < result["similarity"]) & (result["similarity"] < 0.9999)
        ]

        return result.head(n)

    def get_similar_artists(self, artist_name, min_sim_coef=0.5, n=10):
        """
        Using content-based filtering to find similar artists
        """

        cols = [
            "valence",
            "acousticness",
            "danceability",
            "duration_ms",
            "energy",
            "instrumentalness",
            "liveness",
            "loudness",
            "speechiness",
            "tempo",
        ]

        normalized_artist_features = normalize(self.artist_df, cols)
        normalized_artist_features = normalized_artist_features[cols].astype(float)
        artist = normalized_artist_features.loc[artist_name]

        corr_vector = normalized_artist_features[cols].corrwith(artist[cols], axis=1)

        result = (
            pd.merge(
                self.artist_df,
                corr_vector.rename("similarity"),
                left_index=True,
                right_index=True,
            )
            .drop(cols, axis=1)
            .sort_values("similarity", ascending=False)
        )

        result = result[
            (min_sim_coef < result["similarity"]) & (result["similarity"] < 0.9999)
        ]

        return result

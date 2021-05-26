import pandas as pd
from utils import to_1D


class AudioRecommenderSystem:
    """
    Основний клас рекомендаційної системи
    """

    def __init__(self, songdata_path, userdata_path, artistdata_path):
        print("Initializing RS...")
        print("Getting general song data")
        # Метадані для пісень [song_id, title, release, artist_name, year]
        song_metadata = pd.read_csv(songdata_path)

        print("Getting user data")
        # Дані про кількість прослуховувань пісень користувачами [user_id, song_id, listen_count]
        self.user_listens = pd.read_fwf(userdata_path)
        self.user_listens.columns = ["user_id", "song_id", "listen_count"]

        print("Merging song and user data")
        # Дані лише з тими піснями, що мають прослуховування
        # [song_id, title, artist_name, release, year, listen_count]
        self.song_df_listens = pd.merge(
            self.user_listens,
            song_metadata.drop_duplicates(["song_id"]),
            on="song_id",
            how="left",
        )

        print("Getting artist data")
        # Розширення даних про пісні додатковим полем з жанрами
        spotify_artists = pd.read_csv(artistdata_path)

        print("Making extended song data")
        _grouped_song_df = (
            self.song_df_listens.groupby(
                ["song_id", "title", "artist_name", "release", "year"]
            )
            .agg({"listen_count": "count"})
            .reset_index()
        )

        self.extended_song_df = pd.merge(
            _grouped_song_df,
            spotify_artists,
            left_on="artist_name",
            right_on="name",
            how="left",
        )[
            [
                "song_id",
                "title",
                "artist_name",
                "release",
                "year",
                "listen_count",
                "genres",
            ]
        ]

        self.extended_song_df = self.extended_song_df.dropna(subset=["genres"])
        self.extended_song_df["genres"] = (
            self.extended_song_df["genres"].apply(eval).apply(tuple)
        )

        print("Making extended artist data")
        # Датафрейм з рейтингами виконавців
        self.extended_artist_df = (
            self.extended_song_df.groupby(["artist_name", "genres"])
            .agg({"listen_count": "sum"})
            .reset_index()
        )

        print("RS successfully initialized")

    def get_user_prefered_genres(self, user_id, n=15):
        """
        Жанри, що найчастіше зустрічаються у плейлисті користувача
        """
        user_song_data = pd.merge(
            self.user_listens[self.user_listens["user_id"] == user_id],
            self.extended_song_df,
            on="song_id",
        )

        counted_genres = to_1D(user_song_data["genres"]).value_counts()
        return counted_genres.head(n)

    def get_top_songs_by_genre(self, genre, n=50):
        """
        Топ пісень за вказаним жанром
        """
        songs_of_genre = self.extended_song_df[
            self.extended_song_df["genres"].apply(lambda x: genre in x)
        ]
        top_songs = songs_of_genre.sort_values("listen_count", ascending=False)

        return top_songs.head(n)

    def get_top_artists_by_genre(self, genre):
        """
        Топ артистів за вказаним жанром
        """
        artist_in_genre = self.extended_artist_df[
            self.extended_artist_df["genres"].apply(lambda x: genre in x)
        ]

        return artist_in_genre.sort_values(
            "listen_count", ascending=False
        ).reset_index()

    def get_similar_songs_cf(self, song_id, min_sim_coef):
        """
        Колаборативна фільтрація за схожістю сутностей (пісень)
        """
        song_rating_matrix = pd.pivot_table(
            self.song_df_listens,
            values="listen_count",
            index="user_id",
            columns="song_id",
        )
        target_song_row = song_rating_matrix[song_id]

        song_similarity = (
            pd.DataFrame(
                song_rating_matrix.corrwith(target_song_row).drop(song_id),
                columns=["similarity"],
            )
            .reset_index()
            .dropna()
        )

        result = song_similarity.merge(self.extended_song_df).sort_values(
            by="similarity", ascending=False
        )
        result = result[
            (min_sim_coef < result["similarity"]) & (result["similarity"] < 0.9999)
        ]

        return result.reset_index()

    def get_similar_songs_cbf(self, song_id):
        """
        Фільтрація за змістом, враховує схожість пісень за їх жанрами та параметрами
        """
        pass


if __name__ == "__main__":
    print("Initializing RS...")
    print("Getting general song data")
    # Метадані для пісень [song_id, title, release, artist_name, year]
    song_metadata = pd.read_csv(songdata_path)

    print("Getting user data")
    # Дані про кількість прослуховувань пісень користувачами [user_id, song_id, listen_count]
    user_listens = pd.read_fwf(userdata_path)
    user_listens.columns = ["user_id", "song_id", "listen_count"]

    print("Merging song and user data")
    # Дані лише з тими піснями, що мають прослуховування
    # [song_id, title, artist_name, release, year, listen_count]
    song_df_listens = pd.merge(
        user_listens,
        song_metadata.drop_duplicates(["song_id"]),
        on="song_id",
        how="left",
    )

    print("Getting artist data")
    # Розширення даних про пісні додатковим полем з жанрами
    spotify_artists = pd.read_csv(artistdata_path)

    print("Making extended song data")
    _grouped_song_df = (
        song_df_listens.groupby(
            ["song_id", "title", "artist_name", "release", "year"]
        )
        .agg({"listen_count": "count"})
        .reset_index()
    )

    extended_song_df = pd.merge(
        _grouped_song_df,
        spotify_artists,
        left_on="artist_name",
        right_on="name",
        how="left",
    )[
        [
            "song_id",
            "title",
            "artist_name",
            "release",
            "year",
            "listen_count",
            "genres",
        ]
    ]

    extended_song_df = extended_song_df.dropna(subset=["genres"])
    extended_song_df["genres"] = (
        extended_song_df["genres"].apply(eval).apply(tuple)
    )

    print("Making extended artist data")
    # Датафрейм з рейтингами виконавців
    extended_artist_df = (
        extended_song_df.groupby(["artist_name", "genres"])
        .agg({"listen_count": "sum"})
        .reset_index()
    )

    print("RS successfully initialized")
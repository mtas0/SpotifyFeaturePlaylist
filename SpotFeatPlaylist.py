""" 
Music Playlist Generation Script According to Audio Features!

This script analyzes Spotify stream history metadata, splits data into most and least popular tracks based on stream count,
retrieves additional track information using the Spotify API, performs exploratory data analysis (EDA) on top tracks and features,
creates and runs an ensemble neural network model with popularity as the target variable and audio features as input,
and uses the model to generate a playlist of songs with similar features. You can apply this on your own metadata to discover
new tracks similar to your music taste.

Author: Merve H. Tas Bangert
Date: January 30, 2024.

"""
import time
import pickle
import requests
from typing import Any, Optional
import random
import numpy as np
import pandas as pd
from pandas import json_normalize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials


class SpotifyAudioAnalysis:
    def __init__(self, cid, secret):
        """
        Initialiazes the SpotifyAudioAnalysis class.

        Args:
            cid (str): Spotify API client_id.
            secret (str): Spotify API client_secret.
        """
        self.sp = self.spotify_login(cid, secret)
    
    def spotify_login(self, cid, secret):
        """
        Accesses Spotify.

        Args:
            cid (str): Spotify API client_id.
            secret (str): Spotify API client_secret.

        Returns:
            Spotify: Spotify object.
        """
 
        client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
        return Spotify(client_credentials_manager=client_credentials_manager)
    
    
    def to_pandasdf(self, data, years=False, save=False):
        """
        Converts data to Pandas DataFrame.

        Args:
            data (json): list of metadata json files. 
            years (bool, optional):if True, include year information.
            save (bool, optional): if True, save the DataFrame to CSV.
        Returns:
            pandas DataFrame: List of or concatenated dataframes.
        """

        dfs_list = []
        for i in data:
            with open(i, encoding="utf8") as f:
                # read the json files and drop any NaN's from track name.
                df = pd.read_json(f)
                df = df.dropna(subset=["master_metadata_track_name"])
                
                # add year column to show in which year the track was listened to.
                if years:
                    df['ts'] = pd.to_datetime(df['ts'])
                    df['year'] = df['ts'].dt.year
                    grouped_by_year = df.groupby('year')
                    # create separate dataframes for each year.
                    for year, group_df in grouped_by_year:
                        dfs_list.append(group_df[['master_metadata_track_name', 'master_metadata_album_artist_name', 'spotify_track_uri']])
                        if save:
                            group_df.to_csv(f"./df_{year}.csv", index=False)
                else:
                    dfs_list.append(df[['master_metadata_track_name', 'master_metadata_album_artist_name', 'spotify_track_uri']])
                    if save:
                        dfs_list.to_csv('./df.csv', index=False)
        if not years:
            dfs_list = pd.concat(dfs_list)

        return dfs_list

    def create_popularity(self, dfs, topn=10):
        """
        Creates a popularity column in the DataFrame based on track listen count.

        Args:
            dfs (DataFrame): DataFrame or list of DataFrames.
            topn (int, optional): Number of top and bottom tracks to consider. Defaults to 10.

        Returns:
            Pandas DataFrame: DataFrame with added 'popularity' column.
        """
        dfs_selected_list = []

        if isinstance(dfs, list):
            files = dfs
        else:
            files = [dfs]
        
        # create track name - artist name column so unique value counts can be acquire for track listen count.
        for df in files:
            df['track_artist'] = df['master_metadata_track_name'] + ' - ' + df['master_metadata_album_artist_name']
            df['track_listen_count'] = df['track_artist'].map(df['track_artist'].value_counts())

            # get top and bottom tracks.
            top_tracks = df['track_artist'].value_counts().nlargest(topn).index
            bottom_tracks = df['track_artist'].value_counts().nsmallest(topn).index

            # create a new dataframe with unique track_artist combinations.
            df_unique = df.drop_duplicates(subset=['track_artist']).reset_index(drop=True)

            # filter the dataframe based on the top and bottom tracks.
            selected_tracks_df = df_unique[df_unique['track_artist'].isin(top_tracks) | df_unique['track_artist'].isin(bottom_tracks)].reset_index(drop=True)

            # add the the popularity column based on top and bottom tracks
            selected_tracks_df['popularity'] = selected_tracks_df['track_artist'].isin(top_tracks).astype(int)
            dfs_selected_list.append(selected_tracks_df)

        # concatenate the selected dataframes.
        result_df = pd.concat(dfs_selected_list)

        return result_df
    
    def get_genre(self, artist_name):
        """
        Gets genre for a given artist.

        Args:
            artist_name (str): Artist name.

        Returns:
            str: Genre or None.
        """
        try:
            artist_search = self.sp.search(q='artist:' + artist_name, type='artist')
            # search artist genre in Spotify API and get the first result
            if "artists" in artist_search and "items" in artist_search["artists"] and artist_search["artists"]["items"]:
                genres = artist_search["artists"]["items"][0]["genres"][0]
                return genres    
        except IndexError:
            pass
        else:
            return None
    
    def create_genre(self, df): 
        """
        Creates a genre column in the DataFrame.

        Args:
            df (DataFrame): DataFrame where the genre column will be added

        Returns:
            Pandas DataFrame: DataFrame with added 'genre' columns
        """
        genre_dict = {}
        for artist in df["master_metadata_album_artist_name"].unique():
            genre_dict[artist] = self.get_genre(artist)

        # create a genre column and add the genre acquired from Spotify API.
        df['genre'] = df['master_metadata_album_artist_name'].map(genre_dict)
        return df     

    
    def get_audio_features(self, df, save_df=False):
        """
        Extracts audio features and analyses and adds them to the DataFrame.

        Args:
            df (DataFrame): DataFrame where the audio features will be added to.
            save_df (bool, optional): if True, save the DataFrame with audio features to CSV. Defaults to False.

        Returns:
            Pandas DataFrame: DataFrame with added audio features.
        """
        try:
            # extract unique track URIs.
            if 'track_uri' not in df.columns:
                df['track_uri'] = df['spotify_track_uri'].str.split(':').str[-1]

            unique_track_uris = df['track_uri'].unique()
            
            # get audio features for each unique track URI.
            audio_features_list = []
            for track_uri in unique_track_uris:
                try:
                    audio_features = self.sp.audio_features(track_uri)[0]
                    audio_analysis = self.sp.audio_analysis(track_uri)
                    mean_bar_duration = np.mean([entry['duration'] for entry in audio_analysis["bars"]])
                    mean_pitches = np.mean([entry["pitches"] for entry in audio_analysis["segments"]])
                    mean_max_loudness = np.mean([entry["loudness_max"] for entry in audio_analysis["segments"]])
                    mean_timbre = np.mean([entry["timbre"] for entry in audio_analysis["segments"]])
                    mean_tatums = np.mean([entry["duration"] for entry in audio_analysis["tatums"]])
                    audio_features_list.append({'track_uri': track_uri, 'audio_features': audio_features, 'mean_bar_duration':mean_bar_duration,
                                                'mean_pitches': mean_pitches, 'mean_max_loudness': mean_max_loudness, 'mean_timbre': mean_timbre, 'mean_tatums': mean_tatums})
                except Exception as audio_error:
                    print(f"Error processing track {track_uri}: {audio_error}")
                
            # create a df from the list of audio features.
            audio_features_df = pd.DataFrame(audio_features_list)

            # merge audio features back into the original df based on the track URI.
            df = pd.merge(df, audio_features_df, on='track_uri', how='left')
            result_df = pd.concat([df, json_normalize(df['audio_features'])], axis=1)
            result_df = result_df.drop(['audio_features'], axis=1)

            if save_df:
                result_df.to_csv('./df_with_audio_features.csv', index=False)

            return result_df
        except Exception as e:
            print("Error:", e)
            return None

class CreatePlots():
    def __init__(self, df):
        """
        Initializes the CreatePlots class.

        Args:
            df (DataFrane): DataFrame.
        """
        self.df = df

    def barplot_streamcount(self, xlabel="artist", topn=10, rotation=45, year=None):
        """
        Creates a bar count plot for the top streamed variables (e.g. artist, track, or genre).

        Args:
            xlabel (str, optional): Value count variable. Defaults to "artist".
            topn (int, optional): Number of top variable to display. Defaults to 10.
            rotation (int, optional): Rotation of the x tick labels. Defaults to 45.
            year (int, optional): The specific year that the item was listened to. Defaults to None.
        """
        sns.set_palette("viridis")
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))

        if year is not None:
            plt_title = f'Top {topn} Most Streamed {xlabel.title()}s {year}'
            df_tracks = self.df[self.df["year"] == year]
        else:
            plt_title = f'Top {topn} Most Streamed {xlabel.title()}s'
            df_tracks = self.df

        if xlabel == "track":        
            df_top_tracks = df_tracks.nlargest(topn+2, 'track_listen_count')
            
            sns.barplot(x='track_listen_count', y='master_metadata_track_name', data=df_top_tracks)
        elif xlabel == "artist":
            top_artists = (
                df_tracks.groupby(f'master_metadata_album_{xlabel}_name')['track_listen_count']
                .sum()
                .nlargest(topn)
                .reset_index(name='total_listen_count'))
            
            sns.barplot(x='total_listen_count', y=f'master_metadata_album_{xlabel}_name', data=top_artists)
        else:
            top_x = (
                df_tracks.groupby(xlabel)['track_listen_count']
                .sum()
                .nlargest(topn)
                .reset_index(name='total_listen_count'))
            
            sns.barplot(x='total_listen_count', y=xlabel, data=top_x)


        plt.xlabel('Stream Count')
        plt.ylabel(xlabel.title())
        plt.title(plt_title, fontsize=14, fontweight='bold') 
        plt.tick_params(axis='y', rotation=rotation, labelsize=8)

        plt.xticks(rotation=rotation, fontsize=8)
        
        plt.grid(axis='x', linestyle='--', alpha=0.7)  
        plt.rc('axes', axisbelow=True)  

        plt.tight_layout() 
        plt.show()     

    def plot_features(self, feature_list=None, groupby=None, normalize=True):
        """
        Plots the mean and standard deviation of audio featues.

        Args:
            feature_list (list, optional):List of audio features to plot. Defaults to None.
            groupby (str, optional): Column to group by. Defaults to None.
            normalize (bool, optional): If True, normalize the feature values. Defaults to True.
        """
        default_features = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "time_signature", "mean_bar_duration",
                            "mean_pitches", "mean_max_loudness", "mean_timbre", "mean_tatums"]

        if feature_list is None:
            feature_list = default_features
        
        if normalize:
            self.df[feature_list] = (self.df[feature_list] - self.df[feature_list].min()) / (self.df[feature_list].max() - self.df[feature_list].min())

        if groupby is not None and groupby in self.df.columns:
        # group by the specified column and calculate mean and std.
            grouped_df = self.df.groupby(groupby)[feature_list].agg(['mean', 'std']).transpose()

            plt.figure(figsize=(18, 6))
            width = 0.35
            x = np.arange(len(feature_list))  

            for idx, (group, values) in enumerate(grouped_df.items()):
                mean_values = values.xs('mean', level=1)
                std_values = values.xs('std', level=1)

                plt.bar(x + width * idx, mean_values.values, width=width, yerr=std_values.values, label=f'{group} Mean', capsize=5)
            plt.xticks(x, feature_list, rotation=45)
            plt.legend()
            plt.title('Mean and Standard Deviation of Audio Features Grouped by ' + groupby.capitalize())
        else:
            # overall barplot
            df = self.df[feature_list] 
            mean_values = df.mean()
            std_values = df.std()    

            plt.figure(figsize=(18, 6))

            sns.barplot(x=mean_values.index, y=mean_values, color='skyblue', label='Mean')
            plt.errorbar(x=mean_values.index, y=mean_values, yerr=std_values, fmt='none', color='black', capsize=5, label='Std Dev')

            plt.title('Mean and Standard Deviation of Audio Features')
            plt.xticks(rotation=45, ha='right')

        plt.xlabel('Audio Features')
        plt.ylabel('Values')        
        plt.legend()
        
        plt.show()


class RunClassifier():
    def __init__(self, df):
        """
        Initializes the RunClassifier class.

        Args:
            df (DataFrame): DataFrame.
        """
        self.df = df

    def create_model(self, train_size=0.82, max_iter=400, random_state=42, audio_features=None, target_variable=None):
        """
        Creates and evaluates an ensembed model using RandomForestClassifier and MLPClassifier.

        Args:
            train_size (float, optional): Proportion of the dataset to include in the training split. Defaults to 0.82.
            max_iter (int, optional): Maximum number of iterations for MLPClassifier. Defaults to 400.
            random_state (int, optional): Maximum number of iterations for MLPClassifier. Defaults to 42.
            audio_features (DataFrame col. list, optional):Features used for training the model. Defaults to None.
            target_variable (DataFrame col., optional): Target variable used for training the model. Defaults to None.

        Returns:
            obj.: Trained ensemble model (VotingClassifier).
        """
        default_features = self.df[["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "mean_bar_duration",
                                    "mean_pitches", "mean_max_loudness", "mean_timbre", "mean_tatums"]]
        default_target = self.df['popularity']

        if audio_features is None:
            audio_features = default_features
        if target_variable is None:
            target_variable = default_target

        # train and test split the data
        X_train, X_test, y_train, y_test = train_test_split(audio_features, target_variable, train_size=train_size, random_state=random_state)
         
        # scale input variables before running classifier.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # instantiate the classifiers.
        rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", solver="sgd", shuffle=False, max_iter=max_iter, random_state=random_state)

        # add it to ensemble classifier.
        voting_classifier = VotingClassifier(estimators=[
            ('rf', rf_classifier),
            ('mlp', mlp_classifier)
        ], voting='soft')


        # fit the ensemble model.
        voting_classifier.fit(X_train_scaled, y_train)

        # make .
        voting_predictions = voting_classifier.predict(X_test_scaled)

        # evaluate the ensemble model.
        print("Voting Classifier:")
        print("Accuracy:", accuracy_score(y_test, voting_predictions))
        print("Classification Report:\n", classification_report(y_test, voting_predictions))
                
        cm = confusion_matrix(y_test, voting_predictions)

        sns.heatmap(cm ,annot = True)
        plt.xlabel("Predicted")
        plt.ylabel("Truth")

        return voting_classifier
    
    def save_model(self, cls, name=None):
        """
        Saves the trained model to a file.

        Args:
            cls: Trained model.
            name (str, optional): Name of the file to save the model. Defaults to None.
        """
        if name is not None:
            with open(f'{name}.pk1', 'wb') as model_file:
                pickle.dump(cls, model_file)
                print("Model saved successfully.")
        else:
            with open('spotify-audiopref-model.pk1', 'wb') as model_file:
                pickle.dump(cls, model_file)
                print("Model saved successfully.")


class CreatePlaylist():
    def __init__(self, spotify_analyzer, classifier_model, num_songs=10):
        """
        Initializes the CreatePlaylist class.

        Args:
            spotify_analyzer (obj): An instance of SpotifyAudioAnalysis class.
            classifier_model (obj): Trained classifier model.
            num_songs (int, optional): Number of songs to include in the playlist. Defaults to 10.
        """
        self.spotify_analyzer = spotify_analyzer
        self.classifier_model = classifier_model
        self.num_songs = num_songs

    def get_wildcard(self):
        """
        Generates a random wildcard string

        Returns:
            str: Randomly generated wildcard string.
        """
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        wildcard = random.choice(alphabet)

        rand = random.randint(0, 2)
        wildcard = f"%{wildcard}%" if rand == 2 else f"%{wildcard}" if rand == 0 else f"{wildcard}%"

        return wildcard

    def get_random(self, spotify,
               limit= 10,
               offset_min= 0,
               offset_max= 20,
               type= "track",
               market=None,
               album=None,
               artist=None,
               track=None,
               year=None,
               upc=None,
               tag_hipster=None,
               tag_new=None,
               isrc=None,
               genre=None,
               ):
        """
        Gets a random element (track, album, artist) from Spotify.

        Args:
            spotify (Spotify): Spotify API client.
            limit (int, optional): Maximum number of items to retrieve. Defaults to 10.
            offset_min (int, optional): Minimum offset for randomization. Defaults to 0.
            offset_max (int, optional): Maximum offset for randomization. Defaults to 20.
            type (str, optional): Type of item to search for (track, album, artist). Defaults to "track".
            market (Any, optional): Spotify market. Defaults to None.
            album (str, optional): Album name. Defaults to None.
            artist (str, optional): Artist name. Defaults to None.
            track (str, optional):Track name. Defaults to None.
            year (str, optional): Release year. Defaults to None.
            upc (str, optional): Universal Product Code. Defaults to None.
            tag_hipster (bool, optional): Hipster tag. Defaults to None.
            tag_new (bool, optional):New tag_. Defaults to None.
            isrc (str, optional): International Standard Recording Code. Defaults to None.
            genre (str, optional): Music genre. Defaults to None.

        Returns:
            dict: Random element dictionary.
        """
        if offset_max > 1000:
            raise ValueError("The maximum allowed offset is 1000.")

        random_type = random.choice(type.split(','))

        offset = random.randint(offset_min, offset_max)

        # get random wildcard to search in Spotify.
        wildcard = self.get_wildcard()
        q = f"{wildcard}"

        filters = [
            ("artist", artist),
            ("album", album),
            ("track", track),
            ("year", year),
            ("genre", genre),
            ("isrc", isrc),
            ("upc", upc),
            ("tag:hipster", tag_hipster),
            ("tag:new", tag_new),
        ]
        # search using filters if desired.
        for filter_name, filter_value in filters:
            filter_supported = not any(filter_name in t for t in ["playlist", "show", "episode"])
            if filter_supported and filter_value is not None:
                q += f" {filter_name}:{filter_value}"

        result = spotify.search(q, limit, offset, type, market)
        random_type_result_key = f"{random_type}s"

        if result is None:
            raise ValueError("No result was returned.")

        try:
            element = result[random_type_result_key]["items"][0]
        except KeyError:
            raise KeyError("The result could not be parsed.")

        return element

    
    def create_playlist(self, genre=None, tag_hipster=None, tag_new=None, market=None):
        """
        Creates a playlist based on user preferences and classifier predictions.

        Args:
            genre (str, optional): Preferred music genre. Defaults to None.
            tag_hipster (bool, optional): Hipster tag preference. Defaults to None.
            tag_new (bool, optional): New tag preference. Defaults to None.
            market (Any, optional): Spotify market. Defaults to None.

        Returns:
            pandas DataFrame: DataFrame containing the created playlist.
        """
        # use a set to store unique track URIs
        playlist_data = set()  

        max_retries = 3
        retry_delay = 2

        #while len(playlist_data) < self.num_songs and max_retries > 0:
        try:
            # get a random track from Spotify
            rd_track = self.get_random(self.spotify_analyzer.sp, genre=genre, tag_hipster=tag_hipster, tag_new=tag_new, market=market)
            
            uri = rd_track.get('uri').split(':')[-1]

            # check if the track is already in the playlist
            # if uri in playlist_data:
            #     continue  

            artist = rd_track.get('artists', [])[0].get('name', None)
            track = rd_track.get('name')

            # extract audio features
            audio_features = self.spotify_analyzer.sp.audio_features(uri)[0]
            audio_analysis = self.spotify_analyzer.sp.audio_analysis(uri)
            mean_bar_duration = np.mean([entry['duration'] for entry in audio_analysis["bars"]])            
            mean_pitches = np.mean([entry["pitches"] for entry in audio_analysis["segments"]])
            mean_max_loudness = np.mean([entry["loudness_max"] for entry in audio_analysis["segments"]])
            mean_timbre = np.mean([entry["timbre"] for entry in audio_analysis["segments"]])
            mean_tatums = np.mean([entry["duration"] for entry in audio_analysis["tatums"]])

            scaler = StandardScaler()

            # scale the features
            scaled_features = scaler.fit_transform([[audio_features["danceability"], audio_features["energy"], audio_features["acousticness"],
                                                    audio_features["speechiness"], audio_features["instrumentalness"],
                                                    audio_features["liveness"], audio_features["valence"], mean_bar_duration, 
                                                    mean_pitches, mean_max_loudness, mean_timbre, mean_tatums]])

            # make a prediction using the model
            prediction = self.classifier_model.predict(scaled_features)

            # if the model predicts that you like the song, add it to the playlist_data set
            if prediction == 1:
                playlist_data.add((track, artist, uri))
        except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', 1))
                    print(f"Rate limited. Retrying after {retry_after} seconds.")
                    time.sleep(retry_after + 1)
                    #continue
                else:
                    raise
        except Exception as e:
            print(f"An error occured: {e}")

        # create the playlist df
        playlist = pd.DataFrame(playlist_data, columns=['track', 'artist', 'uri'])

        return playlist

# This is a flask app to deploy the model on the web
from flask import Flask, render_template, request, redirect
from utils import extract_feature
import numpy as np
import pickle
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pyaudio
import wave
import pickle
from sys import byteorder
from array import array
from struct import pack
from sklearn.neural_network import MLPClassifier
import os
from utils import extract_feature
from scipy.io import wavfile
import time
from datetime import timedelta as td
import pandas as pd
import requests
import webbrowser

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30


def is_silent(snd_data):
    "True if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Normalizes the volume"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trims blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Adds silence to the start and end"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Records using the microphone
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()



# Initialize the flask app
app = Flask(__name__)


# Spotify credentials
cid = '628e916c2f3246f594bc5f6100ed2423'
secret = '9295dcd50cf14aef8fc1ed70a1b0b21c'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Spotify URLS
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE_URL = "https://api.spotify.com"
API_VERSION = "v1"
SPOTIFY_API_URL = "{}/{}".format(SPOTIFY_API_BASE_URL, API_VERSION)

# make a request to the spotify API
def get_spotify_token():
    # POST
    response = requests.post(SPOTIFY_TOKEN_URL, {
        'grant_type': 'client_credentials',
        'client_id': cid,
        'client_secret': secret,
    })
    token_json = response.json()
    access_token = token_json['access_token']
    return access_token

# get the token
token = get_spotify_token()

# set the header
headers = {
    'Authorization': 'Bearer {token}'.format(token=token)
}

# return a json of list of songs in a playlist given the playlist id
def get_playlist_songs(playlist_id):
    # GET
    playlist_endpoint = "{}/playlists/{}/tracks".format(SPOTIFY_API_URL, playlist_id)
    response = requests.get(playlist_endpoint, headers=headers)
    playlist_json = response.json()
    return playlist_json

# return a json of the song given the song id
def get_song(song_id):
    # GET
    song_endpoint = "{}/tracks/{}".format(SPOTIFY_API_URL, song_id)
    response = requests.get(song_endpoint, headers=headers)
    song_json = response.json()
    return song_json


def spotify_playlist_emotion(playlist_id, emotion):
    playlist_id = playlist_id
    playlist_json = get_playlist_songs(playlist_id)
    songs_json = playlist_json['items']
    for song_json in songs_json:
        song_id = song_json['track']['id']
        song_json = get_song(song_id)
    print("Song for {} emotion".format(emotion))
    print("Song name: {}".format(song_json['name']))
    print("Song url: {}".format(song_json['external_urls']['spotify']))
    print("Singer: {}".format(song_json['artists'][0]['name']))
    # play the playlist
    webbrowser.open(song_json['external_urls']['spotify'])



# This is the route where we will display the form to the user
@app.route("/")

def index():
 # load the saved model (after training)

    model = pickle.load(open("result/speech_sentiment2.model", "rb"))
    print("Please talk - wait 1 sec and start speaking")
    filename = "test.wav" # file name
    # record the file (start talking)
    record_to_file(filename)
    # extract features and reshape it
    features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # predict
    result = model.predict(features)[0]
    if result == 'happy':
        # pass a playlist id of a happy playlist
        spotify_playlist_emotion('37i9dQZF1DXcBWIGoYBM5M', result)
    elif result == 'sad':
        # pass a playlist id of a sad playlist
        spotify_playlist_emotion('37i9dQZF1DWSqBruwoIXkA' , result)
    elif result == 'angry':
        # pass a playlist id of an angry playlist
        spotify_playlist_emotion('37i9dQZF1DX6VdMW310YC7' , result)
    elif result == 'calm':
        # pass a playlist id of a neutral playlist
        spotify_playlist_emotion('37i9dQZF1DX59NCqCqJtoH' , result)

    return render_template("index.html", data=result)

# run the app
if __name__ == "__main__":
    app.run(debug=True, port=9000, host="127.0.0.1")
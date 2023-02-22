from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import google.auth

credentials, project = google.auth.default()

from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    '/Users/AIUDD75/Downloads/bcr-technology-hackathon-47cf62696e22.json')

scoped_credentials = credentials.with_scopes(
    ['https://www.googleapis.com/auth/cloud-platform'])

# Create your views here.
from django.http import HttpResponse
from django.template import loader
from pytube import YouTube
from google.cloud import storage
import os
import subprocess
from yt_dlp import YoutubeDL
from google.cloud import videointelligence
import whisper
whisper_model = whisper.load_model("base")

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained('t5-3b')
model = AutoModelWithLMHead.from_pretrained('t5-3b', return_dict=True)

from google.cloud import language_v1
import six



# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")
def index(request):
    # template = loader.get_template('index.html')
    context = {'sayan':1}
    # return HttpResponse(template.render(context, request))
    return render(request, 'index.html', context)

@csrf_exempt 
def show_video_info(request):
    if request.method == 'POST':
        vids = []
        words = request.POST['kw']
        vc = request.POST['vc']
        print("KeyWords: ", words, vc)
        a = subprocess.run(f"yt-dlp ytsearch10:{words} --get-id --min-views {vc}", stdout=subprocess.PIPE, shell = True )
        print("videos: ", a)
        metadata = []
        for i in a.stdout.splitlines():
            with YoutubeDL() as ydl: 
                info_dict = ydl.extract_info(f'https://www.youtube.com/watch?v={i.decode("utf-8")}', download=False)
            metadata.append([info_dict.get("original_url"), \
                     info_dict.get("fulltitle"), \
                     info_dict.get("channel"),\
                     info_dict.get("duration_string"), 
                     info_dict.get("channel_follower_count"),\
                     info_dict.get("like_count"),\
                     info_dict.get("comment_count"),\
                     i.decode("utf-8"),\
                     info_dict.get("view_count"),\
                     info_dict.get("uploader"),\
                     
                     ])
        
        print(metadata)
        context = {'metadata':metadata}
    return render(request, 'show_video_info.html', context)

@csrf_exempt 
def video_details(request):
    
    if request.method == 'POST':
        vid = request.POST['id']
    print("VID ", vid)
    # outfile = download_video(vid.split(',')[0])
    outfile = download_video(f'https://www.youtube.com/watch?v={vid}')
    os.rename(outfile, f'{vid}.mp4')
    print("FIle Saves as ", outfile)
    print (os.getcwd())
    os.chdir('..')
    bucket_name = 'erpv1'
    blob_name = f'videos/{vid}.mp4'
    gcs_url = upload_to_bucket(blob_name, f"videos/{vid}.mp4", bucket_name)
    print("GCS URL ", gcs_url) 
    # speech = transcribe("gs://erpv1/videos/amway-vitB.mp4")
    speech = my_custom_transcribe('videos/'+vid+'.mp4', vid)

    # with open('audio.txt', 'r') as file:
    #     speech = file.read().replace('\n', '')
    
       
    summary = get_summary(speech)
    # with open('summary.txt', 'r') as file:
    #     summary = file.read().replace('\n', '')
    senti, mg = sample_analyze_sentiment(summary)
    # print("SENTI ", senti, mg)
    # print(summary) 
    context = {'full_speech': speech, 'summary': summary, 'senti': senti, 'mag': mg}
        # create a form instance and populate it with data from the request:

    return render(request, 'video_details.html', context)

def download_video(url):
    # YouTube(url).streams.first().download()
    os.chdir("videos/")
    yt = YouTube(url)
    outfile = yt.streams\
    .filter(progressive=True, file_extension='mp4')\
    .order_by('resolution')\
    .desc()\
    .first()\
    .download()
    
    return outfile


#pip install --upgrade google-cloud-storage. 

def upload_to_bucket(blob_name, path_to_file, bucket_name):
    """ Upload data to a bucket"""
     
    # Explicitly use service account credentials by specifying the private key
    # file.
    # storage_client = storage.Client.from_service_account_json('creds.json')
    storage_client = storage.Client(project="bcr-technology-hackathon")
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path_to_file)
    
    #returns a public url
    return blob.public_url


def transcribe(path):
    """Transcribe speech from a video stored on GCS."""
    speech = []

    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.SPEECH_TRANSCRIPTION]

    config = videointelligence.SpeechTranscriptionConfig(
        language_code="en-US", enable_automatic_punctuation=True
    )
    video_context = videointelligence.VideoContext(speech_transcription_config=config)

    path = path  

    operation = video_client.annotate_video(
        request={
            "features": features,
            "input_uri": path,
            "video_context": video_context,
        }
    )
    print("\nProcessing video for speech transcription.")

    result = operation.result(timeout=600)

    # There is only one annotation_result since only
    # one video is processed.
    annotation_results = result.annotation_results[0]

    for speech_transcription in annotation_results.speech_transcriptions:
        for alternative in speech_transcription.alternatives:
            # print("Transcript: {}".format(alternative.transcript))
            speech.append(alternative.transcript)
    return speech


def my_custom_transcribe(v_path, vid):
    command = f'ffmpeg -y -i {v_path} -ab 160k -ar 44100 -vn audios/{vid}.wav'
    subprocess.call(command, shell=True)
    
    result = whisper_model.transcribe('audios/'+vid+'.wav')
    file = open("transcribes/audio.txt","w")
    file.writelines(result["text"])
    file.close()
    return result["text"]

def get_summary(speech):
    inputs = tokenizer.encode("summarize: " + speech, \
                          return_tensors='pt', \
                          max_length=512, \
                          truncation=True)
    summary_ids = model.generate(inputs, max_length=512, min_length=60)
    summary = tokenizer.decode(summary_ids[0])
    file = open("summary.txt","w")
    file.writelines(summary)
    file.close()
    return summary


def sample_analyze_sentiment(content):

    client = language_v1.LanguageServiceClient()

    # content = 'Your text to analyze, e.g. Hello, world!'

    if isinstance(content, six.binary_type):
        content = content.decode("utf-8")

    type_ = language_v1.Document.Type.PLAIN_TEXT
    document = {"type_": type_, "content": content}

    response = client.analyze_sentiment(request={"document": document})
    sentiment = response.document_sentiment
    print("Score: {}".format(sentiment.score))
    print("Magnitude: {}".format(sentiment.magnitude))

    return sentiment.score, sentiment.magnitude
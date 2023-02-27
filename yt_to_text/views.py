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

from sentence_transformers import SentenceTransformer, util
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

from google.cloud import language_v1
import six

from google.cloud import bigquery
bq_client = bigquery.Client()

import pickle
import numpy as np
import pandas as pd
import json
from datetime import datetime

import numpy as np  
import pandas as pd 

pd.set_option('Display.max_columns',None,'Display.max_rows',None) 






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
        metadata = []
        print("here1")
        words = request.POST.get('kw', 'xyz')
        print("here2")
        vc = request.POST.get('vc','50')
        print("KeyWords: ", words, vc)
        if words != 'xyz':
            a = subprocess.run(f"yt-dlp ytsearch5:'{words}' --get-id --min-views {vc}", stdout=subprocess.PIPE, shell = True )
            print("videos: ", a)
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
                        info_dict.get("upload_date"),\
                        info_dict.get("language")\
                        ])
            
        url = request.POST.get('url', 'xyz')
        print("here3", url)
        if url !='xyz':
            with YoutubeDL() as ydl: 
                info_dict = ydl.extract_info(url, download=False)
                metadata.append([info_dict.get("original_url"), \
                     info_dict.get("fulltitle"), \
                     info_dict.get("channel"),\
                     info_dict.get("duration_string"), 
                     info_dict.get("channel_follower_count"),\
                     info_dict.get("like_count"),\
                     info_dict.get("comment_count"),\
                     url[-11:],\
                     info_dict.get("view_count"),\
                     info_dict.get("uploader"),\
                     info_dict.get("upload_date"),\
                     info_dict.get("language")\
                     ])
        context = {'metadata':metadata}
        
    return render(request, 'show_video_info.html', context)

@csrf_exempt 
def video_details(request):
    
    if request.method == 'POST':
        vid = request.POST['id']
    print("VID ", vid)
    original_speech = ''
    lang = "es-US"
    outfile = download_video(f'https://www.youtube.com/watch?v={vid}')
    os.rename(outfile, f'{vid}.mp4')
    print("FIle Saves as ", outfile)
    print (os.getcwd())
    os.chdir('..')
    bucket_name = 'erpv1'
    #Upload Video
    blob_name = f'videos/{vid}/{vid}.mp4'
    path_to_file = f"videos/{vid}.mp4"
    gcs_url = upload_to_bucket(blob_name, path_to_file, bucket_name)
    
    print("GCS URL ", gcs_url) 
    speech = my_custom_transcribe('videos/'+vid+'.mp4', vid)
    # print(speech)
    

    # with open('transcribes/audio.txt', 'r') as file:
    #     speech = file.read().replace('\n', '')
    
       
    summary = get_summary(speech)
    # with open('summary.txt', 'r') as file:
    #     summary = file.read().replace('\n', '')
    senti, mg = sample_analyze_sentiment(summary)
    print("SENTI ", senti, mg)
    print(summary) 
    keyword_df = keyword_similarity(speech, vid)
    overall_score = sum(keyword_df.score)/len(keyword_df.texts)
    
    # gcs_url = upload_to_bucket(blob_name, path_to_file, bucket_name)
    # print(keyword_df)
    json_records = keyword_df.reset_index().to_json(orient ='records')
    kdf = []
    kdf = json.loads(json_records)
    # context ={'sayan':'here'}
    #Upload Audio
    blob_name = f'videos/{vid}/{vid}.wav'
    path_to_file = f"audios/{vid}.wav"
    gcs_url = upload_to_bucket(blob_name, path_to_file, bucket_name)
    #Upload Transcript
    blob_name = f'videos/{vid}/{vid}.txt'
    path_to_file = f"transcribes/audio.txt"
    gcs_url = upload_to_bucket(blob_name, path_to_file, bucket_name)

    context = {'full_speech': speech, 'summary': summary, 'senti': senti, 'mag': mg, 'kdf': kdf, 'oscore':overall_score}
        # create a form instance and populate it with data from the request:
    metadata = []
    with YoutubeDL() as ydl: 
            info_dict = ydl.extract_info(f'https://www.youtube.com/watch?v={vid}', download=False)
            metadata.append([info_dict.get("original_url"), \
                        info_dict.get("fulltitle"), \
                        info_dict.get("channel"),\
                        info_dict.get("duration_string"), 
                        info_dict.get("channel_follower_count"),\
                        info_dict.get("like_count"),\
                        info_dict.get("comment_count"),\
                        vid,\
                        info_dict.get("view_count"),\
                        info_dict.get("uploader"),\
                        info_dict.get("upload_date"),\
                        original_speech,
                        info_dict.get("description"),\
                        summary,
                        senti,
                        speech,
                        overall_score,
                        len(keyword_df.texts)
                                                
                        ])
    save_to_bq(metadata)
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


def transcribe(path, lang):
    """Transcribe speech from a video stored on GCS."""
    speech = []
    os.chdir("videos/")
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.SPEECH_TRANSCRIPTION]

    config = videointelligence.SpeechTranscriptionConfig(
        language_code=lang, enable_automatic_punctuation=True
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
    print("SPEECH", speech)
    file = open("transcribes/audio.txt","w")
    file.writelines(speech)
    file.close()
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



def save_to_bq(metadata):
    table = bq_client.get_table("{}.{}.{}".format('bcr-technology-hackathon', 'Popkin_Team', 'yt_crawl_data_hack'))
    for i in metadata:
        print("My Data ", i)
        row_to_insert = [{u"id": i[7], u"orig_uri": i[0], u"title": i[1],  u"channel": i[2],  u"duration": i[3], \
                            u"channel_follower_count": i[4],  u"like_count": i[5],  u"comment_count": i[6],\
                             u"view_count": i[8],  u"uploader": i[9],  u"upload_date": i[10], \
                                u"language": i[11],u"description": i[12], u"full_text": i[15], u"summary": i[13], \
                                u"impact_score": '.2',\
                                u"confidence_score": '0.5',\
                                 u"sentiment_score": str(i[14]), 
                                 u"exact_keyword_count": '2',\
                                u"semantic_keyword_count":str(i[17]), 
                                u"semantic_score": str(i[16]),\
                                u"ts": datetime.now().timestamp()
                           }]
        errors = bq_client.insert_rows_json(table, row_to_insert)
        if errors == []:
            print("success")
        else: print(errors)



def keyword_similarity(text, vid):
    sentences2 = text.split('.')
    embeddings2 = sentence_model.encode(sentences2, convert_to_tensor=True)
    result = {}
    with open('ana_keywords.pickle', 'rb') as handle:
        d = pickle.load(handle)
    # print(d['Covid Keywords']['Market Language (English)'])
    for i in d['Covid Keywords']['Market Language (English)']:
        # print(i)
        sentences1 = i
        embeddings1 = sentence_model.encode(sentences1, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        idx = list(np.where(cosine_scores>.3)[1])
        scores = cosine_scores[0][idx].tolist()
        if len(idx) > 0:
            result[i] = [idx, scores]
        # print(result)
        data = []
        for key, values in result.items():
            if key == "#StayProtected":
                print("StayProtected: ",values)
            lines = [sentences2[i] for i in values[0]]
            score = np.mean(values[1])
            data.append([key, lines, score, vid])
        df = pd.DataFrame(data)
        df.rename(columns={0: 'key_words', 1: 'texts', 2: 'score', 3: 'id'}, inplace=True)
    # print("STAY ",df[df['key_words'] == "#StayProtected"]) 
    #Upload semantic score
    df.to_csv(f"transcribes/{vid}.csv", encoding='utf-8', index=False)
    blob_name = f'videos/{vid}/{vid}.csv'
    path_to_file = f"transcribes/{vid}.csv"
    bucket_name = 'erpv1'
    gcs_url = upload_to_bucket(blob_name, path_to_file, bucket_name)   
    return df


def translate(path, lang):
    client = speech.SpeechClient()
    audio =  "gs://erpv1/videos/-8UxTeSOz6E/-8UxTeSOz6E.flac"
    audio = speech.RecognitionAudio(uri=audio)
    config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
                sample_rate_hertz=44100,
                language_code= "es-US", #"en-US",
                audio_channel_count=2,
            )

    operation = client.long_running_recognize(config=config, audio=audio)

    response = operation.result()
    #Convert Spanish audio to English text
    project_id = environ.get("PROJECT_ID", "bcr-technology-hackathon")
    parent = f"projects/{project_id}"
    client = translate.TranslationServiceClient()

    target_language_code = "en"

    out_text = []
    orig_lang=[]
    for result in response.results:
        it = result.alternatives[0].transcript
        orig_lang.append(it)
        tt = client.translate_text(
            contents=[it],
            target_language_code=target_language_code,
            parent=parent,
        ) 
        ot = MessageToDict(tt._pb)['translations'][0]['translatedText']
        out_text.append(ot)  
    return orig_lang, out_text


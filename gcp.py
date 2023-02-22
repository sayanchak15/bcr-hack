
import google.auth

credentials, project = google.auth.default()

from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    '/Users/AIUDD75/Downloads/bcr-technology-hackathon-47cf62696e22.json')

scoped_credentials = credentials.with_scopes(
    ['https://www.googleapis.com/auth/cloud-platform'])


from google.cloud import storage

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

bucket_name = 'erpv1'
blob_name = 'videos/test-video-5.mp4'

upload_to_bucket(blob_name, 'audio.wav', bucket_name)
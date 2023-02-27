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

vid = 'XZeL3n-7Yvg'
bucket_name = 'erpv1'
blob_name = f'videos/{vid}/{vid}.mp4'
path_to_file = "videos/XZeL3n-7Yvg.mp4"
upload_to_bucket(blob_name, path_to_file, bucket_name)
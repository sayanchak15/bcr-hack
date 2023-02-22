

from google.cloud import language_v1
import six

import google.auth

credentials, project = google.auth.default()

from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    '/Users/AIUDD75/Downloads/bcr-technology-hackathon-47cf62696e22.json')

scoped_credentials = credentials.with_scopes(
    ['https://www.googleapis.com/auth/cloud-platform'])


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

print(sample_analyze_sentiment("I'm very happy"))
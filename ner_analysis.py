from google.cloud import language_v1
import pandas as pd

def analyze_text(text_content):
    client = language_v1.LanguageServiceClient()

    type_ = language_v1.types.Document.Type.PLAIN_TEXT

    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}

    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_entity_sentiment(
        request={"document": document, "encoding_type": encoding_type}
    )
    
    return response

def analyze_text_as_dataframe(text_content):
    
    response = analyze_text(text_content)
    
    entities = [(i.name,
                 i.type_.name,
                 i.type_.value,
                 i.salience,
                 i.sentiment.score,
                 i.sentiment.magnitude) for i in response.entities]
    
    columns = ['name',
               'type',
               'type_value',
               'salience',
               'sentiment_score',
               'sentiment_magnitude']
    
    result = pd.DataFrame(entities, columns=columns)
    
    return result
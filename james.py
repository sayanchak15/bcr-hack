import pandas as pd

import ner_analysis


def ana_keyword_parser(excel_file, sheet_name):
    """
    NOTE: requires openpyxl to run
    NOTE: Expects the string 'Sub-category' in cell B3 for function to run
    """
    
    ana_opp_misreps = pd.read_excel(excel_file, 
                                    sheet_name=sheet_name,
                                    header=2)
    
    inp = ana_opp_misreps["Sub-category"].dropna().reset_index()

    inp_dict = dict(zip(inp['index'], inp["Sub-category"]))

    inp_dict = {k:v for k,v in inp_dict.items() if "Keywords:" not in v}
    
    # define the columns
    columns = ana_opp_misreps.columns[3:].to_list()
    
    columns.pop(columns.index('OPERATOR'))

    ana_keyword_dict = dict()

    for i in range(0, len(list(inp_dict.items())) -1):
        # print(i)
        start = list(inp_dict.items())[i][0]+2
        stop = list(inp_dict.items())[i+1][0]-1

        topic = list(inp_dict.items())[i][1]

        keywords = ana_opp_misreps[columns][start:stop]

        ana_keyword_dict[topic] = keywords

    # grab the final key and value
    start = list(inp_dict.items())[-1][0]

    stop = ana_opp_misreps.shape[0]

    topic = list(inp_dict.items())[-1][1]

    keywords = ana_opp_misreps[columns][start:stop]

    ana_keyword_dict[topic] = keywords
    
    return ana_keyword_dict

def ner_keyword_labeler(text, key_word_dict):
    
    result = ner_analysis.analyze_text_as_dataframe(text)
    
    for i in range(len(list(key_word_dict.items()))):

        flag_type = list(key_word_dict.items())[i][0]

        # print(flag_type)

        key_words = list(key_word_dict.items())[i][1]

        # print(key_words)

        flag_series = [key_words.apply(lambda x: x == i).any().any() for i in result.name]

        result[flag_type] = flag_series
        
    return result


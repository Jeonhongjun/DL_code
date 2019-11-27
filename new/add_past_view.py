import pandas as pd
import numpy as np
import json

def past_view():
    date_list = pd.read_csv('./data/date_list.csv', sep='\t', header=0)
    date = 0
    while date != (len(date_list)+1):
        contents = pd.read_csv('./Past_data/' + str(date_list["date"][date]) +'.csv', header=0, index_col=0)
        df = pd.read_csv('./data/hashmap.tsv', sep='\t', header=None)

        for i in range(contents.count()[0]):
            if np.any(df[df[0] == contents['contents'][i]][1].values != None):
                    contents['contents'][i] = df[df[0] == contents['contents'][i]][1].values[0]
            else :
                continue
        contents.to_csv('./Past_data/'+ str(date_list["date"][date]) + '.csv', encoding='utf-8')

        contents = pd.read_csv('./Past_data/' + str(date_list["date"][date]) +'.csv', header=0, index_col=0)
        contents2 = pd.read_csv('./data/' + str(date_list["date"][date+1]) +'.csv', header=0, index_col=0)

        past_hc = []
        for i in range(len(contents)):
            past_hc.append(contents["ViewCount"][i]/contents["Episode"][i])

        past_hc = pd.DataFrame(past_hc, columns = ["Past_count"])

        reconst = pd.concat([contents, past_hc], axis =1)

        reconst.to_csv('Past_data/'+ str(date_list["date"][date]) +'.csv', encoding='utf-8')

        past_view = []
        for i in contents2.Program.tolist():
            past_view.append(contents[contents["contents"] == i]["Past_count"].values)

        past_view = pd.DataFrame(past_view, columns = ["past_view"])

        past_view = past_view.fillna(0)

        contents = pd.concat([contents2, past_view], axis = 1)

        contents.to_csv('data/'+ str(date_list["date"][date]) +'.csv', encoding='utf-8')

        date = date + 1

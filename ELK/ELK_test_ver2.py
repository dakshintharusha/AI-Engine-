import time
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import tensorflow as tf
from joblib import load
import warnings

warnings.filterwarnings("ignore")

#load scaler model
sc= load('C:\scaler.bin')

#load ANN
ANN = tf.keras.models.load_model('C:\ANN.h5')

elastic_client = Elasticsearch()

# Retrive Elasticsearch documents to get with API call
def get_elk_docs():
    response = elastic_client.search(
        index='test',
        body={}
    )
    # declare a new list for the Elasticsearch documents
    # nested inside the API response object
    elastic_docs = response["hits"]["hits"]
    fields = {}
    for num, doc in enumerate(elastic_docs):
            pass

            source_data = doc["_source"]

            for key, val in source_data.items():   
                if(key == "src_ip"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "dst_ip"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])
                        
            for key, val in source_data.items():   
                if(key == "dst_port"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "flow_duration"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "tot_fwd_pkts"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "tot_bwd_pkts"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "totlen_fwd_pkts"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "totlen_bwd_pkts"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_pkt_len_max"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_pkt_len_min"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_pkt_len_mean"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_pkt_len_std"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_pkt_len_max"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_pkt_len_min"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_pkt_len_mean"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_pkt_len_std"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "flow_byts_s"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "flow_pkts_s"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "flow_iat_mean"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "flow_iat_std"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "flow_iat_max"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "flow_iat_min"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_iat_tot"):                                   
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_iat_mean"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_iat_std"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_iat_max"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_iat_min"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_iat_tot"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_iat_mean"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_iat_std"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_iat_max"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_iat_min"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_psh_flags"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_psh_flags"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_urg_flags"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_urg_flags"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_header_len"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_header_len"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_pkts_s"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_pkts_s"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "pkt_len_min"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "pkt_len_max"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "pkt_len_mean"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "pkt_len_std"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "pkt_len_var"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fin_flag_cnt"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "syn_flag_cnt"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "rst_flag_cnt"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "psh_flag_cnt"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "ack_flag_cnt"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "urg_flag_cnt"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "cwe_flag_count"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "ece_flag_cnt"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "down_up_ratio"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "pkt_size_avg"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_seg_size_avg"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_seg_size_avg"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_byts_b_avg"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_pkts_b_avg"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_blk_rate_avg"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_byts_b_avg"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_pkts_b_avg"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "bwd_blk_rate_avg"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "subflow_fwd_pkts"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "subflow_fwd_byts"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "subflow_bwd_pkts"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "subflow_bwd_byts"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "init_fwd_win_byts"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "init_bwd_win_byts"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_act_data_pkts"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "fwd_seg_size_min"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "active_mean"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "active_std"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "active_max"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "active_min"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "idle_mean"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "idle_std"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "idle_max"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])

            for key, val in source_data.items():   
                if(key == "idle_min"):
                    try:
                        fields[key] = np.append(fields[key], val)
                    except KeyError:
                        fields[key] = np.array([val])
    return fields

def ann_predicts(elastic_df):
    F_df = elastic_df
    f_df = F_df.drop(['src_ip', 'dst_ip'], axis = 1)
    features = sc.transform(f_df)
    preds = ANN.predict(features)
    predictions = np.argmax(preds, axis=1)
    labels = predictions.astype(str)

    for i in range(labels.shape[0]):
        if(labels[i] == '0'):
           labels[i] = 'benign'
        elif(labels[i] == '1'):
           labels[i] = 'bot'
        elif(labels[i] == '2'):
           labels[i] = 'bruteforce'
        elif(labels[i] == '3'):
           labels[i] = 'ddos'
        elif(labels[i] == '4'):
           labels[i] = 'ftppatator'
        elif(labels[i] == '5'):
           labels[i] = 'goldeneye'
        elif(labels[i] == '6'):
           labels[i] = 'hulk'
        elif(labels[i] == '7'):
           labels[i] = 'portscan'
        elif(labels[i] == '8'):
           labels[i] = 'slowhttptest'
        elif(labels[i] == '9'):
           labels[i] = 'slowloris'
        elif(labels[i] == '10'):
           labels[i] = 'sql'
        elif(labels[i] == '11'):
           labels[i] = 'sshpatator'
        elif(labels[i] == '12'):
           labels[i] = 'sql'
        elif(labels[i] == '13'):
           labels[i] = 'synflood'  
    Labels = pd.DataFrame(labels, columns = ['Labels'])
    #Appending Labels Column to dataframe
    elastic_df['Labels'] = Labels['Labels'].values
    return elastic_df

use_these_keys = [ "ID", "src_ip", "dst_ip", "dst_port", "flow_duration", "tot_fwd_pkts", "tot_bwd_pkts", "totlen_fwd_pkts", "totlen_bwd_pkts", "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_mean", "fwd_pkt_len_std", "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean", "bwd_pkt_len_std", "flow_byts_s", "flow_pkts_s", "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min", "fwd_iat_tot", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max", "fwd_iat_min", "bwd_iat_tot", "bwd_iat_mean", "bwd_iat_std", "bwd_iat_max", "bwd_iat_min", "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags", "fwd_header_len", "bwd_header_len", "fwd_pkts_s", "bwd_pkts_s", "pkt_len_min", "pkt_len_max", "pkt_len_mean", "pkt_len_std", "pkt_len_var", "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt", "psh_flag_cnt", "ack_flag_cnt", "urg_flag_cnt", "cwe_flag_count", "ece_flag_cnt", "down_up_ratio", "pkt_size_avg", "fwd_seg_size_avg", "bwd_seg_size_avg", "fwd_byts_b_avg", "fwd_pkts_b_avg", "fwd_blk_rate_avg", "bwd_byts_b_avg", "bwd_pkts_b_avg", "bwd_blk_rate_avg", "subflow_fwd_pkts", "subflow_fwd_byts", "subflow_bwd_pkts", "subflow_bwd_byts", "init_fwd_win_byts", "init_bwd_win_byts", "fwd_act_data_pkts", "fwd_seg_size_min", "active_mean", "active_std", "active_max", "active_min", "idle_mean", "idle_std", "idle_max", "idle_min", "Labels" ]

def filterKeys(document):
    return {key: document[key] for key in use_these_keys }

def ann_doc_generator(df):
    df_iter = df.iterrows()
    for index, document in df_iter:
        yield {
                "_index": 'elk_test',
                "_type": "_doc",
                "_id" : f"{document['ID']}",
                "_source": filterKeys(document),
            }
es_client = Elasticsearch(http_compress=True)

def main():

    count = 1
    while True:
        print('Records :', count)

        #retrive dictionary
        Docs = get_elk_docs()
        
        # create a Pandas DataFrame array from the fields dict
        elastic_df = pd.DataFrame(Docs)
        
        #predict usning ANN
        Elastic_df = ann_predicts(elastic_df)

        # Exporting Pandas Data to Elasticsearch
        Elastic_df.insert(0, 'ID', range(count, count + len(Elastic_df)))        
        df_iter = Elastic_df.iterrows()
        index, document = next(df_iter)
        helpers.bulk(es_client, ann_doc_generator(Elastic_df))
        
        print('Records :', count , 'exported to ELK \n')
        print('************************************')
        count = count + len(Elastic_df)
        # get new records in every 5 seconds
        time.sleep(5)

if __name__ == '__main__':
    main()

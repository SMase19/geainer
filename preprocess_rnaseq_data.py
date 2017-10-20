import os
import argparse
import json

import pandas as pd
import glob

from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--cancer_id',
                    help='cancer data from TCGA',
                    default='LUNG')
args = parser.parse_args()

c_folder = args.cancer_id

columnlist = ['cancer_id', 'gender', 'age_at_diagnosis', 'days_to_death', 'tumor_stage', 'vital']

# Input File
files = glob.glob('./data/' + c_folder + '/RAW/*/*.FPKM.txt.gz')

j_file = open('./data/' + c_folder + '/RAW/metadata.cart.2017-10-19T07-05-02.956240.json')
json_dict = json.load(j_file)

# Output File
rna_out_raw = os.path.join('data', c_folder, c_folder + '_raw_rnaseq.tsv.gz')
rna_out_file = os.path.join('data', c_folder, c_folder + '_scaled_rnaseq.tsv.gz')
rna_out_zeroone_file = os.path.join('data', c_folder, c_folder + '_scaled_zeroone_rnaseq.tsv.gz')
clinical_out_file = os.path.join('data', c_folder, c_folder + '_clinical.tsv.gz')

rnaseq_trans_df = pd.DataFrame()
clinical_df = pd.DataFrame()
print(len(files))
for i in range(len(files)):
    su_id = 'None'
    for id in json_dict:
        if id['file_id'] == files[i][16:52]:
            su_id = id['cases'][0]['submitter_id']
            c_id = id['cases'][0]['project']['project_id'].replace('TCGA-', '')
            gender = id['cases'][0]['demographic']['gender']
            age = id['cases'][0]['diagnoses'][0]['age_at_diagnosis']
            to_death = id['cases'][0]['diagnoses'][0]['days_to_death']
            stage = id['cases'][0]['diagnoses'][0]['tumor_stage']
            vital = id['cases'][0]['diagnoses'][0]['vital_status']
            in_df = pd.DataFrame([[c_id, gender, age, to_death, stage, vital]], index=[su_id], columns=columnlist)
            clinical_df = clinical_df.append(in_df)
            break
    assert su_id != 'None', 'invalid rnaseq id :{0}'.format(files[i][16:52])

    # 欠損値に0埋めしたい場合、.fillna(0)
    rnaseq_df = pd.read_table(files[i], index_col=0, names=('sample', su_id)).T
    if i == 0:
        rnaseq_trans_df = rnaseq_df
    else:
        rnaseq_trans_df = rnaseq_trans_df.append(rnaseq_df.loc[su_id])

rnaseq_trans_df.to_csv(rna_out_raw, sep='\t', compression='gzip')
clinical_df.to_csv(clinical_out_file, sep='\t', compression='gzip')

# Select upper 5000 mean absolute deviation values
num_mad_genes = 5000
mad_gene = rnaseq_trans_df.mad(axis=0).sort_values(ascending=False)
top_mad_genes = mad_gene.iloc[0:num_mad_genes, ].index
rnaseq_subset_df = rnaseq_trans_df.loc[:, top_mad_genes]

# Scale RNAseq data using z-scores
rnaseq_scaled_df = preprocessing.StandardScaler().fit_transform(rnaseq_subset_df)
rnaseq_scaled_df = pd.DataFrame(rnaseq_scaled_df,
                                columns=rnaseq_subset_df.columns,
                                index=rnaseq_subset_df.index)
rnaseq_scaled_df.to_csv(rna_out_file, sep='\t', compression='gzip')
print(rnaseq_scaled_df.iloc[0:5, 0:2])
print(rnaseq_scaled_df.shape)

# Scale RNAseq data using zero-one normalization
rnaseq_scaled_zeroone_df = preprocessing.MinMaxScaler().fit_transform(rnaseq_subset_df)
rnaseq_scaled_zeroone_df = pd.DataFrame(rnaseq_scaled_zeroone_df,
                                        columns=rnaseq_subset_df.columns,
                                        index=rnaseq_subset_df.index)
rnaseq_scaled_zeroone_df.to_csv(rna_out_zeroone_file, sep='\t', compression='gzip')
print(rnaseq_scaled_zeroone_df.iloc[0:5, 0:2])
print(rnaseq_scaled_zeroone_df.shape)

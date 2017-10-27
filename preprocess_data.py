import os
import argparse

import pandas as pd
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--cancer_id',
                    help='cancer data from TCGA',
                    default='lung')
args = parser.parse_args()

c_folder = args.cancer_id

rna_file = os.path.join('data', c_folder, 'raw', 'HiSeqV2.gz')
#clinical_file = os.path.join('data', 'lung', 'raw', 'LUNG_clinicalMatrix.gz')

rna_out_raw = os.path.join('data', c_folder, c_folder + '_raw_rnaseq.tsv.gz')
rna_out_file = os.path.join('data', c_folder, c_folder + '_scaled_rnaseq.tsv.gz')
rna_out_zeroone_file = os.path.join('data', c_folder, c_folder + '_scaled_zeroone_rnaseq.tsv.gz')

rnaseq_df = pd.read_table(rna_file, index_col=0)
#clinical_df = pd.read_table(clinical_file, index_col=0)

rnaseq_df.index = rnaseq_df.index.map(lambda x: x.split('|')[0])
rnaseq_df.columns = rnaseq_df.columns.str.slice(start=0, stop=15)
rnaseq_df = rnaseq_df.drop('?').fillna(0).sort_index(axis=1)

rnaseq_df.drop('SLC35E2', axis=0, inplace=True)
rnaseq_df = rnaseq_df.T

rnaseq_df.to_csv(rna_out_raw, sep='\t', compression='gzip')
print(rnaseq_df.iloc[0:5, 0:2])
print(rnaseq_df.shape)

num_mad_genes = 5000
mad_genes = rnaseq_df.mad(axis=0).sort_values(ascending=False)
top_mad_genes = mad_genes.iloc[0:num_mad_genes, ].index
rnaseq_subset_df = rnaseq_df.loc[:, top_mad_genes]

rnaseq_scaled_df = preprocessing.StandardScaler().fit_transform(rnaseq_subset_df)
rnaseq_scaled_df = pd.DataFrame(rnaseq_scaled_df,
                                columns=rnaseq_subset_df.columns,
                                index=rnaseq_subset_df.index)
rnaseq_scaled_df.to_csv(rna_out_file, sep='\t', compression='gzip')
print(rnaseq_scaled_df.iloc[0:5, 0:2])
print(rnaseq_scaled_df.shape)

rnaseq_scaled_zeroone_df = preprocessing.MinMaxScaler().fit_transform(rnaseq_subset_df)
rnaseq_scaled_zeroone_df = pd.DataFrame(rnaseq_scaled_zeroone_df,
                                        columns=rnaseq_subset_df.columns,
                                        index=rnaseq_subset_df.index)
rnaseq_scaled_zeroone_df.to_csv(rna_out_zeroone_file, sep='\t', compression='gzip')
print(rnaseq_scaled_zeroone_df.iloc[0:5, 0:2])
print(rnaseq_scaled_zeroone_df.shape)
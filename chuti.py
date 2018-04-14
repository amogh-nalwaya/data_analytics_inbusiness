import sys
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import csv
import math
import operator
import os
from sklearn.preprocessing import StandardScaler

sys.path.append("/home/miller/Documents/GT/Biz Anal/Projy/Code/data_analytics_inbusiness/")

from feat_eng import *
from modeling import *
from mlp_bayes_opt_legit import *
from create_pred_set import *

#abs_path = os.path.abspath(__file__)
#file_dir = os.path.dirname(abs_path)
#sys.path.append(file_dir)


os.chdir("/home/miller/Documents/GT/Biz Anal/Projy/Data/CSV/")

def group_basket_stats(product_list, df_transactions, df_demographic):

    print("Grouping Baskets...")
    df_grouped_basket = get_grouped_basket(df_transactions)

    print("getting product counts for each basket")
    df_grouped_basket_count = get_grouped_basket_count(df_grouped_basket)

    print("getting summed quantities for each basket id...")
    df_grouped_basket_sum = get_grouped_basket_sum(df_grouped_basket)

    print("Applying label...")
    df_grouped_basket = apply_label_grouped_basket(df_grouped_basket)

    print("merging count, sum and labels...")
    df_grouped_basket_merge = merging_sum_count_labels(df_grouped_basket, df_grouped_basket_count, df_grouped_basket_sum)

    print("merging with demmographic data....")
    df_grouped_basket_merge = df_grouped_basket_merge.merge(df_demographic, on="household_key", how="left").reset_index(drop=True)

    print("First ten rows of the dataset...")
    print(df_grouped_basket_merge.head(10)) # Sanity check

    return df_grouped_basket_merge

def get_grouped_basket(df_transactions):
    return df_transactions.groupby(['household_key', 'BASKET_ID', 'DAY'])

def get_grouped_basket_count(df_grouped_basket):
    df_grouped_basket_count = df_grouped_basket.size().reset_index()
    df_grouped_basket_count = df_grouped_basket_count.rename(columns={0: 'PROD_PURCHASE_COUNT'})
    return df_grouped_basket_count

def apply_label_grouped_basket(df_grouped_basket):
    df_grouped_basket = df_grouped_basket.apply(
            lambda x : 1 if len(set(x.PRODUCT_ID.tolist()) & set(product_list)) > 0 else 0
        ).reset_index().rename(columns={0:"label"})
    return df_grouped_basket

def get_grouped_basket_sum(df_grouped_basket):
    df_grouped_basket_sum = df_grouped_basket.sum().reset_index()
    df_grouped_basket_sum.drop(['RETAIL_DISC', 'TRANS_TIME', 'COUPON_MATCH_DISC', 'START_DAY', 'END_DAY'], axis=1, inplace=True)
    return df_grouped_basket_sum

def merging_sum_count_labels(df_grouped_basket, df_grouped_basket_count, df_grouped_basket_sum):
    df_grouped_basket_merge = df_grouped_basket_sum.merge(df_grouped_basket, on=["household_key", "BASKET_ID"]).reset_index(drop=True)
    del df_grouped_basket
    del df_grouped_basket_sum

    df_grouped_basket_merge = df_grouped_basket_merge.merge(df_grouped_basket_count, on=["household_key", "BASKET_ID"]).reset_index(drop=True)
    del df_grouped_basket_count

    df_grouped_basket_merge = df_grouped_basket_merge.drop(['DAY_x', 'DAY_y'], axis=1)

    return df_grouped_basket_merge

def get_products_for_coupon(coupon_Id, df_coupon):
    subset = df_coupon[df_coupon['COUPON_UPC'] == coupon_Id]
    return subset['PRODUCT_ID'].unique()

def get_campaigns_for_coupon(coupon_Id, df_coupon):
    subset = df_coupon[df_coupon['COUPON_UPC'] == coupon_Id]
    return subset['CAMPAIGN'].unique()

def get_households_for_campaigns(campaigns, df_campaign_table, df_campaign_desc):
    #get subset from campaign table to get the households for the campaign
    subset = df_campaign_table[df_campaign_table['CAMPAIGN'].isin(campaigns)]
    hh_start_dates = subset.merge(df_campaign_desc, on='CAMPAIGN', how='left')
    hh_start_dates = hh_start_dates.sort_values(['household_key', 'START_DAY'])
    return hh_start_dates.drop_duplicates(['household_key'], keep="first")

def get_transactions_for_hh(df_transactions, hh_start_dates):
    trans_merge = df_transactions.merge(hh_start_dates, on='household_key', how='left')
    trans_merge['START_DAY'].fillna(10000, inplace=True)
    return trans_merge[trans_merge['DAY'].astype(float) < trans_merge['START_DAY']]

def get_transactions_for_hh_within(df_transactions, hh_start_dates, product_list):
    trans_merge = df_transactions.merge(hh_start_dates, on='household_key', how='left')
    trans_merge['START_DAY'].fillna(10000, inplace=True)
    trans_merge['END_DAY'].fillna(0, inplace=True)
    trans_filtered = trans_merge[(trans_merge['DAY'].astype(float) >= trans_merge['START_DAY']) & (
                trans_merge['DAY'].astype(float) <= trans_merge['END_DAY'])]
    trans_filtered['label'] = 0
    trans_filtered['label'] = trans_filtered.apply(lambda row: 1 if row['PRODUCT_ID'] in product_list else 0,
                                                   axis=1)
    return trans_filtered[['household_key', 'PRODUCT_ID', 'CAMPAIGN']][trans_filtered['label'] == 1], list(trans_filtered['household_key'].unique())

if __name__ == "__main__":
    
    coupon_Id = "10000085362"

    print("Coupon ID: " + coupon_Id)

    print("Reading coupon data...")
    df_coupon = pd.read_csv('coupon.csv', dtype={'COUPON_UPC': str, 'CAMPAIGN': str, 'PRODUCT_ID': str})
    campaigns = get_campaigns_for_coupon(coupon_Id, df_coupon)
    print("Campaigns associated with the coupon: " + str(len(campaigns)))

    product_list = get_products_for_coupon(coupon_Id, df_coupon)
    del df_coupon
    print("Products associated with the coupon: "+ str(len(product_list)))

    print("Reading in campaign_table and campaign_desc...")
    df_campaign_table = pd.read_csv('campaign_table.csv', dtype={'household_key': str, 'CAMPAIGN': str})
    df_campaign_desc = pd.read_csv('campaign_desc.csv', dtype={'CAMPAIGN': str})

    hh_start_dates = get_households_for_campaigns(campaigns, df_campaign_table, df_campaign_desc)
    del df_campaign_table
    hh_start_dates.drop(['DESCRIPTION_x', 'DESCRIPTION_y'], axis=1, inplace=True)
    print("Households associated with the campaign: "+str(len(hh_start_dates)))

    print("Reading in transactions... it's huge")
    df_transactions = pd.read_csv('transaction_data.csv', dtype={'BASKET_ID': str, 'PRODUCT_ID': str, 'household_key': str, 'DAY': str})
    print("lenght of all transactions: "+str(len(df_transactions)))

    print("filtering transactions for households ")
    df_transactions = get_transactions_for_hh(df_transactions, hh_start_dates)
    df_transactions['CUSTOMER_PAID'] = df_transactions['SALES_VALUE'] + df_transactions['COUPON_DISC']
    print("filtered transactions length: "+str(len(df_transactions)))

    df_demographic = pd.read_csv('hh_demographic.csv', dtype={'household_key': str})
    df_grouped_basket = group_basket_stats(product_list, df_transactions, df_demographic)

    df_grouped_basket['demo_missing'] = np.where(df_grouped_basket['INCOME_DESC'].isnull(), 1, 0) # Creating one-hot for where demographic data is missing
    
    print("Feature engineering...")
    exp_stats = ['label', 'PROD_PURCHASE_COUNT', 'QUANTITY']

    df_eng_feats_train = feat_eng(df_grouped_basket, exp_stats, exp_stats)

    df_eng_feats_train = prep_train_set(df_eng_feats_train)

    #Optional code to write output to a file
#    df_eng_feat_train.to_csv("train_set_feat_eng_{}.csv".format(coupon_Id), index=False)
    print("length of feat eng: "+str(len(df_eng_feats_train)))

    X, y = split_feats_label(df_eng_feats_train)
        
    scaler = StandardScaler()
    features_std = scaler.fit_transform(X) # Normaliizing features
    
#    del X

    #train the model
    print("Training the model...")
    
    trained_mlp = train_mlp(features_std[:100], y[:100], 2, 1, 10, 10)
    
    print("\nGenerating prediction set")

    df_eng_feats_pred = gen_pred_set(coupon_Id)
    
#    set(df_eng_feats_train.columns) - set(df_eng_feats_pred.columns)
    
    X,y = split_feats_label(df_eng_feats_pred)
    
    pred_features_std = scaler.transform(pred_features)
    
    pred_set_predictions = trained_mlp.predict(pred_features_std)

#df_eng_feats_train["AGE_DESC"]



### HOW TO FIND MOST COMMON PRODUCT ###

#most_common_prod = 1082185
#
#x = df_transactions.groupby("PRODUCT_ID").size().reset_index()
#
#x.sort_values(0, inplace=True, ascending = False)
#
#x.head()
#
#df_coupon[df_coupon.PRODUCT_ID.astype(float) == most_common_prod]

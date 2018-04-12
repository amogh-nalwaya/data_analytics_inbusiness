import sys
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import csv
import math
import operator

def get_grouped_basket(product_list, trans_merge):
    print(len(trans_merge[trans_merge['BASKET_ID'] == "27601281299"]))

    df_grouped_basket = trans_merge.groupby(['household_key', 'BASKET_ID', 'DAY']).size().reset_index()
    # print("=============")
    # print(df_grouped_basket.head())
    # print("=============")
    df_grouped_basket.columns = ['household_key', 'BASKET_ID', 'DAY', 'COUNT']
    df_grouped_basket_2 = trans_merge.groupby(['household_key', 'BASKET_ID', 'STORE_ID']).sum().reset_index()
    # print("~~~~~~~~~~~~~")
    # print(df_grouped_basket_2.head())
    # print("~~~~~~~~~~~~~~~~")
    df_grouped_basket_2.drop(['RETAIL_DISC', 'TRANS_TIME', 'COUPON_MATCH_DISC', 'START_DAY', 'END_DAY', 'WEEK_NO_x', 'WEEK_NO_y'], axis=1, inplace=True)

    df_grouped_basket_merge = df_grouped_basket_2.merge(df_grouped_basket, on=["household_key", "BASKET_ID"]).reset_index()
    #print(df_grouped_basket_merge.head())
    trans_merge_label = trans_merge.apply(lambda x : 1 if len(set(x.PRODUCT_ID.tolist()) & set(product_list)) > 0 else 0).reset_index().rename(columns={0:"label"})
    #print(trans_merge_label.head())
    return df_grouped_basket_merge

def get_products_for_coupon(coupon_Id, df_coupon):
    subset = df_coupon[df_coupon['COUPON_UPC'] == coupon_Id]
    return subset['PRODUCT_ID'].unique()

def get_campaigns_for_coupon(coupon_Id, df_coupon):
    subset = df_coupon[df_coupon['COUPON_UPC'] == coupon_Id]
    return subset['CAMPAIGN'].unique()

def get_households_for_campaigns(campaigns, df_campaign_table, df_campaign_desc):
    #get subset from campiagn table to get the households for the campaign
    subset = df_campaign_table[df_campaign_table['CAMPAIGN'].isin(campaigns)]
    hh_start_dates = subset.merge(df_campaign_desc, on='CAMPAIGN', how='left')
    hh_start_dates = hh_start_dates.sort_values(['household_key', 'START_DAY'])
    return hh_start_dates.drop_duplicates(['household_key'], keep="first")

def get_transactions_for_hh(df_transactions, hh_start_dates):
    trans_merge = df_transactions.merge(hh_start_dates, on='household_key', how='left')
    return trans_merge[trans_merge['DAY'].astype(float) < trans_merge['START_DAY']]

def add_week_to_transactions(df_transactions):
    df_transactions['WEEK_NO'] = df_transactions.apply(lambda row: math.ceil(float(row['DAY'])/ 7), axis=1)
    return df_transactions

def merge_with_causal(causal_data, df_transactions):
    df_trans_merge = df_transactions.merge(causal_data, on=['STORE_ID', 'PRODUCT_ID'], how="left")
    return df_trans_merge

if __name__ == "__main__":
    coupon_Id = "51800000050"

    df_coupon = pd.read_csv('coupon.csv', dtype={'COUPON_UPC': str, 'CAMPAIGN': str, 'PRODUCT_ID': str})
    campaigns = get_campaigns_for_coupon(coupon_Id, df_coupon)

    df_campaign_table = pd.read_csv('campaign_table.csv', dtype={'household_key': str, 'CAMPAIGN': str})
    df_campaign_desc = pd.read_csv('campaign_desc.csv', dtype={'CAMPAIGN': str})

    hh_start_dates = get_households_for_campaigns(campaigns, df_campaign_table, df_campaign_desc)
    hh_start_dates.drop(columns=['DESCRIPTION_x', 'DESCRIPTION_y'], inplace=True)

    df_transactions = pd.read_csv('transaction_data.csv', dtype={'BASKET_ID': str, 'PRODUCT_ID': str, 'household_key': str, 'DAY': str})
    df_causal = pd.read_csv('causal_data.csv', dtype={'PRODUCT_ID': str, 'STORE_ID': int, 'WEEK_NO': float, 'display': str})

    print("length::::::::::::::::::")

    print(len(df_transactions[df_transactions["BASKET_ID"] == "27601281299"]))

    trans_merge = get_transactions_for_hh(df_transactions, hh_start_dates)

    df_transactions = add_week_to_transactions(trans_merge)

    print("length::::::::::::::::::")

    print(len(df_transactions[df_transactions["BASKET_ID"] == "27601281299"]))

    df_transactions = merge_with_causal(df_causal, df_transactions)
    df_transactions['CUSTOMER_PAID'] = df_transactions['SALES_VALUE'] + df_transactions['COUPON_DISC']
    df_transactions['WEEKS_TO_MAILER'] = df_transactions['WEEK_NO_x'] - df_transactions['WEEK_NO_y']
    df_transactions['WEEKS_TO_MAILER'].fillna(1000, inplace=True)
    df_transactions.sort_values(['household_key', 'BASKET_ID', 'PRODUCT_ID', 'WEEKS_TO_MAILER'], inplace=True)
    print(df_transactions[df_transactions["BASKET_ID"] == "27601281299"])
    df_transactions = df_transactions.drop_duplicates(['household_key', 'BASKET_ID', 'PRODUCT_ID'], keep="first")
    df_transactions.loc[df_transactions['WEEKS_TO_MAILER'] < 0, 'WEEKS_TO_MAILER'] = 1000
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~===========")
    print(len(df_transactions[df_transactions['WEEKS_TO_MAILER'] > 0]))
    print(df_transactions[df_transactions["BASKET_ID"] == "27601281299"])


    product_list = get_products_for_coupon(coupon_Id, df_coupon)
    df_grouped_basket = get_grouped_basket(product_list, df_transactions)



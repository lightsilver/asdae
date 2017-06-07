# coding: utf-8

import pandas as pd

app_info = pd.read_csv("/home/mjoys/user_profile2/t_appinfo.csv")

app_info.sort_values(by="apptype")

print app_info

from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
import numpy as np
import re
import pandas as pd
import requests
import json
import os
from pathlib import Path

def getdetails(x):
    df=pd.DataFrame()
    for item in x.find_all(class_="card px-0 px-md-4"):
        item_name = (json.loads(item.find(class_='itemInfo').input['value'])['name'])
        item_price = (json.loads(item.find(class_='itemInfo').input['value'])['price'])
        item_cat = (json.loads(item.find(class_='itemInfo').input['value'])['category'])
        item_url = (item.find(class_='productImg').img['src'])
        prod_url = "https://www.ikea.com.hk"+item.find(class_='card-header').a['href']

        df = df.append({"item_name":item_name, "item_price":item_price, "item_cat":item_cat,"item_url":item_url,"prod_url":prod_url},ignore_index=True) 
        return df

def ikeascrape(productlist):
    ikeadf = pd.DataFrame()
    driver = webdriver.Chrome(executable_path='/Applications/chromedriver')
    for product in productlist:
        URL = "https://www.ikea.com.hk/en/products/"+product
        driver.get(URL)
        subhtml = driver.page_source
        soup = BeautifulSoup(subhtml, "html.parser")

        # try:
        while True:
            itemdf = getdetails(soup)
            ikeadf = pd.concat([ikeadf,itemdf])
            WebDriverWait(driver, 30)

            nextlink = soup.find(class_='page-item next')

            if nextlink:
                newurl = nextlink.find('a',{'class':"page-link"})['data-sitemap-url']
                driver.get(newurl)
                newhtml = driver.page_source
                soup = BeautifulSoup(newhtml, "html.parser")
            else:
                break
    return ikeadf
    
def cleansing(df):
    #clean unwanted category
    excludeli = ['0126 Footstools','0917 Baby highchairs',"0951 Children's beds (8-14)","1233 Chairpads","0211 Living room storage"]
    dfclean = df[~df["item_cat"].isin(excludeli)]

    #drop duplicated images
    dfclean.drop_duplicates(subset ="item_url",keep=False, inplace = True) 

    dfclean['item_cat'].replace(
    {'0113 Sofa beds': 'sofas', 
    '0111 Sofas': 'sofas',
    '0125 Armchairs': 'chairs',
    '0521 Bed frames..': 'beds',
    '0423 Wardrobes': 'dressers',
     '0212 Living room cabinets':'dressers',
    '0811 Dining tables': 'tables',
    '0822 Dining stools': 'chairs',
    '0821 Dining chairs and folding chairs': 'chairs',
    '0823 Bar stools': 'chairs',
    '1012 Table lamps': 'lamps',
    '1011 Floor lamps': 'lamps',
    '1016 Wall lamps and wall spotlights': 'lamps'},inplace=True
    )
    
    return dfclean.reset_index()

def savecleandf(df):
    df.to_csv("ikeadata2/"+'ikea_scrape.csv',index=False)

def getscrapeimage(newdf):
    for index, row in newdf.iterrows():
        try:
            os.makedirs(Path("ikeadata2/"+str(row['item_cat'])))
        except FileExistsError:
            # directory already exists
            pass

        with open("ikeadata2/"+str(row['item_cat'])+'/'+str(index)+'.jpg','wb') as f:
            image = requests.get(row['item_url'])
            f.write(image.content)

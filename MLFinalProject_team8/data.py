#!/usr/bin/env python
# coding: utf-8

# In[24]:


import urllib.request
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import csv


for num in range(1,95): #1 94
    print(num)
    path='https://www.homeaway.com/results/keywords:New%20York%2C%20NY%2C%20USA/page:'+str(num)+'?petIncluded=false&ssr=true&fbclid=IwAR3EaCuDnt45GpjJewg-btkPFYkBYSAppZUjtPquJGjBSpnoxvYEmGqoons'
    r= requests.get(path)    
    index=0
    if r.status_code == requests.codes.ok:
        soup = BeautifulSoup(r.text, 'html.parser')
        #print(soup.prettify())
        price=[]
        kind=[]
        squ=[]
        bed=[]
        bath=[]
        loc=[]
        rate=[]

        internet=np.zeros(3060)
        aircon=np.zeros(3060)
        tv=np.zeros(3060)
        cable=np.zeros(3060)
        heater=np.zeros(3060)
        wash=np.zeros(3060)
        elevator=np.zeros(3060)
        kit=np.zeros(3060)
        ht=np.zeros(3060)
        fit=np.zeros(3060)
        con=np.zeros(3060)
        park=np.zeros(3060)

        t1 = soup.find_all('a',attrs={'class':'a--plain-link Hit__infoLink'})#找出所有飯店
        yet=[]
        index=0
        for i in t1:
            t2 = i.get('href')
            path='https://www.homeaway.com'+str(t2)
            r2= requests.get(path)
            if r2.status_code == requests.codes.ok:
                soup1 = BeautifulSoup(r2.text, 'html.parser')

                p=soup1.find('meta',property='og:price:amount')
                if p == None:
                    continue
                #tmp=p['content']
                z=soup1.find('meta',property='og:rating')
                if z == None:
                    continue
                #index=index+1
                t=soup1.find_all('span', class_='listing-bullets__span')
                kinds=pd.Series(['Apartment', 'House', 'Cabin', 'Studio', 'Condo', 'Hotel Suites', 'Cottage', 'Resort', 'Hotel'])
                flag1=0
                flag2=0
                flag3=0
                flag4=0
                for i in t:
                    if np.any(kinds.str.contains(i.text)) and flag1==0:
                        kind.append(i.text)
                        flag1=1
                    elif i.text.find('sq. ft')!=-1 and flag2==0:
                        s=i.text.split(' ')
                        if int(s[0])<300:
                            s[0]=0
                        elif int(s[0])<600:
                            s[0]=1
                        elif int(s[0])<900:
                            s[0]=2
                        elif int(s[0])<1200:
                            s[0]=3
                        elif int(s[0])<1500:
                            s[0]=4
                        elif int(s[0])<1800:
                            s[0]=5
                        elif int(s[0])<2100:
                            s[0]=6
                        elif int(s[0])<2400:
                            s[0]=7
                        elif int(s[0])<2700:
                            s[0]=8
                        elif int(s[0])<3000:
                            s[0]=9
                        elif int(s[0])<3300:
                            s[0]=10
                        elif int(s[0])<3600:
                            s[0]=11
                        elif int(s[0])<3900:
                            s[0]=12
                        elif int(s[0])<4200:
                            s[0]=13
                        elif int(s[0])<4500:
                            s[0]=14
                        elif int(s[0])<4800:
                            s[0]=15
                        elif int(s[0])<5100:
                            s[0]=16
                        elif int(s[0])<5400:
                            s[0]=17
                        squ.append(float(s[0])/float(18))
                        flag2=1
                    elif i.text.find('Bedroom')!=-1 and flag3==0:
                        s=i.text.split(' ')
                        bed.append(s[1])
                        flag3=1
                    elif i.text.find('Bathroom')!=-1 and flag4==0:
                        s=i.text.split(' ')
                        bath.append(s[1])
                        flag4=1
                        
                if flag1==0:
                    kind.append(np.NaN)
                if flag2==0:
                    squ.append(np.NaN)
                if flag3==0:
                    bed.append(np.NaN)
                if flag4==0:
                    bath.append(np.NaN)

                price.append(p['content'])
                while True:
                    try:
                        rate.append(float(z['content'])/float(5))
                        break
                    except ValueError:
                        rate.append(np.NaN)

                p2=soup1.find_all(attrs={'class':'amenity-single__label'})
                for k in p2:
                    tmp=k.text
                    if tmp=='Internet': 
                        internet[index]=1
                    elif tmp=='Free Wifi':
                        internet[index]=1
                    elif tmp=='Air Conditioning':
                        aircon[index]=1
                    elif tmp=='TV':
                        tv[index]=1
                    elif tmp=='Television':
                        tv[index]=1
                    elif tmp=='Satellite or Cable':
                        cable[index]=1
                    elif tmp=='Satellite / Cable':
                        cable[index]=1
                    elif tmp=='Heater':
                        heater[index]=1
                    elif tmp=='Heating':
                        heater[index]=1
                    elif tmp=='Washing Machine':
                        wash[index]=1
                    elif tmp=='Elevator':
                        elevator[index]=1
                    elif tmp=='Kitchen':
                        kit[index]=1
                    elif tmp=='Hot Tub':
                        ht[index]=1
                    elif tmp=='Fitness Room / Equipment':
                        fit[index]=1
                    elif tmp=='Concierge':
                        con[index]=1
                    elif tmp=='Parking':
                        park[index]=1
                    else:
                        if tmp not in yet:
                            yet.append(tmp)            
                index=index+1 
    #print(len(price))
    #print(internet)
    internet=np.resize(internet,len(price))
    aircon=np.resize(aircon,len(price))
    tv=np.resize(tv,len(price))
    cable=np.resize(cable,len(price))
    heater=np.resize(heater,len(price))
    wash=np.resize(wash,len(price))
    elevator=np.resize(elevator,len(price))
    kit=np.resize(kit,len(price))
    ht=np.resize(ht,len(price))
    fit=np.resize(fit,len(price))
    con=np.resize(con,len(price))
    park=np.resize(park,len(price))  

    #squ=(squ - np.amin(squ, axis=0))/(np.amax(squ, axis=0)-np.amin(squ, axis=0)) 
    #print(squ)
    
    df = pd.DataFrame({'price':price,'kind':kind,'sqft':squ, 'bed':bed,'bath':bath,'rating':rate,'internet':internet,'air-conditioning':aircon,'television':tv,'Satellite/Cable':cable,'heater':heater,'washing machine':wash,'elevator':elevator,'kitchen':kit,'hot tub':ht,'fitness room':fit,'concierge':con,'parking':park})
    df['price'] = df['price'].astype(str).str.replace('$', '')
    #df['bed'] = df['bed'].str.replace('Bedrooms: ', '')
    #df['bath'] = df['bath'].str.replace('Bathrooms: ', '')
    df['kind'] = df['kind'].astype(str).str.replace('Apartment', '0')
    df['kind'] = df['kind'].astype(str).str.replace('House', '1')
    df['kind'] = df['kind'].astype(str).str.replace('Cabin', '2')
    df['kind'] = df['kind'].astype(str).str.replace('Studio', '3')
    df['kind'] = df['kind'].astype(str).str.replace('Condo', '4')
    df['kind'] = df['kind'].astype(str).str.replace('Hotel Suites', '5')
    df['kind'] = df['kind'].astype(str).str.replace('Cottage', '6')
    df['kind'] = df['kind'].astype(str).str.replace('Resort', '7')
    df['kind'] = df['kind'].astype(str).str.replace('Hotel', '8')
   
    #print (df)
    df['bed'] = df['bed'].replace(to_replace=r'^Bathrooms: .$', value=np.NaN, regex=True)
    df['bed'] = df['bed'].replace(to_replace='Studio', value=np.NaN, regex=True)
    df['price'] = df['price'].replace(to_replace='', value=np.NaN, regex=True)
    #df=df.dropna()
    df['price'] = pd.to_numeric(df['price'])
    df['kind'] = pd.to_numeric(df['kind'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'])
    df['bed'] = pd.to_numeric(df['bed'], errors='coerce')
    df['bath'] = pd.to_numeric(df['bed'], errors='coerce')
    df=df.dropna()
    
    if num==1:
        df_total=df
    else:
        frames = [df_total,df]
        df_total = pd.concat(frames, ignore_index=True)

df_total.to_csv('data.csv')
#df_total.to_csv('data.csv', header=False, mode='a')


# In[32]:


df=pd.read_csv('data.csv')
df=df.drop(df.columns[0], axis=1)
#print(df)
df.to_csv('data.csv', index=False)


# In[ ]:





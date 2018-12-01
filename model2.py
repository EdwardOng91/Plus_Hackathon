import pandas as pd
import numpy as np
import sklearn
from datetime import datetime
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import random

table = pd.read_csv('table.csv')
table['reward'] = round({table['fare'],0})
table['reward'] = table['reward'].astype(int)

def conv_datetime(x):
    a = datetime.strptime(x, '%m/%d/%y %H:%M')
    return a

def time_diff(avr, dept):
    if avr < dept:
        return avr+24
    else:
        return avr*1
		
table['dept_date_time'] = table['dept_date_time'].apply(lambda x : conv_datetime(x))
table['arrv_date_time'] = table['arrv_date_time'].apply(lambda x : conv_datetime(x))

df = table
df['total_duration'] = df['norm_duration'] + df['rnr_duration']
df = df.set_index('cust_id')
df['depart_hr'] = df['dept_date_time'].apply(lambda x: x.hour)
df['arv_hr'] = df['arrv_date_time'].apply(lambda x: x.hour)
df = df.drop(['norm_duration','rnr_duration','dept_date_time','arrv_date_time'],axis=1)
df['arv_hr'] = df.apply(lambda x: time_diff(x['arv_hr'], x['depart_hr']),1)
df['hr_duration'] = df['arv_hr']-df['depart_hr']
df['next_day'] = df['arv_hr'].apply(lambda x : 1 if x>24 else 0)

#Input
from_where = 'tapah'
where_to = 'taiping'
when = '9:00'

def format_check(location):
    corr_format = []
    for i in location.split(' '):
        corr_format.append(i.capitalize())

    if len(corr_format)>1:
        return ' '.join(corr_format)
    else:
        return corr_format[0]

#Correct format
def output_detail(dest_from, dest_to, time):
    global vehicles
    global model_data
    global pred
    global location_from
    global location_to
    global pred
    global duration
    global min_dept
    global max_arv
    location_from = format_check(dest_from)
    location_to = format_check(dest_to)
#     print('From {} to {}'.format(location_from, location_to))

    travel_from_no = df[df['travel_from'] == location_from]['travel_from_no'][0]
    travel_to_no = df[df['travel_to'] == location_to]['travel_to_no'][0]
#     print('From {} to {}'.format(travel_from_no, travel_to_no))

    dept_hr = int(time.split(':')[0])
#     print('Departing at {}:00'.format(dept_hr))

    model_data = df[(df['travel_from_no']>=travel_to_no)&
                    (df['travel_to_no']<=travel_from_no)]
#     print(len(model_data))
    
    min_dept = dept_hr
    max_arv = min_dept + int(model_data[(model_data['travel_from_no']==travel_from_no)&
               (model_data['travel_to_no']==travel_to_no)]['hr_duration'].max())
    if max_arv > 24:
        max_arv = max_arv-24
        next_day = 1
    else:
        next_day = 0
        
    if next_day == 1:
        duration = max_arv+24-min_dept
    else:
        duration = max_arv-min_dept
    
    #number of vehicles
    model_data = df[(df['depart_hr']<max_arv)&
                    (df['arv_hr']>min_dept)&
                    (df['next_day']==next_day)]
#     print(len(model_data))

    #congestion amount
    past_data2 = pd.read_csv(r'C:\Users\edwar\Desktop\Hackathon\past2.csv')
    y=past_data2.delay
    x=past_data2[['cars','road_work','bad_weather']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    #Prediction
    vehicles = len(model_data)
    weather = random.randint(0,1)
    road_work = random.randint(0,1)
    user_input = [[len(model_data),weather,road_work]]
    pred = regr.predict(user_input)
    
output_detail(from_where,where_to,when)

fare = []
timeslot = []
selected_timeslot = []
travel_duration = []
eta = []


if pred/60 > duration/1.5:
    congestion = 'yes'

    for i in range(min_dept-3,min_dept+4):
        when = str(i)+":00"
        output_detail(from_where,where_to,when)
        fare.append(df[(df['travel_from']==location_from)&(df['travel_to']==location_to)]['fare'][0])
        timeslot.append(when)
        selected_timeslot.append(len(model_data))
        travel_duration.append(str(int(str((duration + pred[0]/60)).split('.')[0])) +
                       ":"+
                       str(int(round({((duration + pred[0]/60) - int(str((duration + pred[0]/60)).split('.')[0]))*60,0}))))
        eta.append(str(int(str((max_arv + pred[0]/60)).split('.')[0])) +
                       ":"+
                       str(int(round({((max_arv + pred[0]/60) - int(str((max_arv + pred[0]/60)).split('.')[0]))*60,0}))))
        
else:
    congestion = 'no'
    output_detail(from_where,where_to,when)
    fare.append(df[(df['travel_from']==location_from)&(df['travel_to']==location_to)]['fare'][0])
    timeslot.append(when)
    selected_timeslot.append(len(model_data))
    travel_duration.append(str(int(str((duration + pred[0]/60)).split('.')[0])) +
                   ":"+
                   str(int(round({((duration + pred[0]/60) - int(str((duration + pred[0]/60)).split('.')[0]))*60,0}))))
    eta.append(str(int(str((max_arv + pred[0]/60)).split('.')[0])) +
                   ":"+
                   str(int(round({((max_arv + pred[0]/60) - int(str((max_arv + pred[0]/60)).split('.')[0]))*60,0}))))
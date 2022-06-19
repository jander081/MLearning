import pandas as pd
import numpy as np
from datetime import datetime



#---------------------------------------------------------------------   
def j_factors(col, dframe, tol):
        n =  len(dframe[col].value_counts())
        print('Number of factors: ', n, '\n')
        
        
        while True:
            try:
                if n >= tol:
                    ques = ' '
                    while not (ques.startswith('y') or ques.startswith('n')):

                        ques = input('The factors exceed your tolerance, do you wish to see a truncated version?').lower()
                    if ques.startswith('y'):
                        print(dframe[col].value_counts().head(10))
                        break
                    else:
                        print('cool')
                        break
                else:
                    print(dframe[col].value_counts())
                    break

            except ValueError:
                print('yes or no')

                
                
#---------------------------------------------------------------------                
def delta(string):
    
    '''this takes in the joined date and target'''
    
    if string.split()[1] == 'Non-Renew':
        start_date = string.split()[0]
        end_date = datetime(2018, 6, 1)
        start_date = datetime.strptime(start_date,'%Y%m') 
        delta = end_date - start_date
        days = delta.days
        if days >= 365 and days < 730:
            days = 365
        elif days >= 730:
            days = 730
            
        else:
            days = delta.days
        
        return days
    
    else:
        start_date = string.split()[0]
        start_date = datetime.strptime(start_date,'%Y%m') 
        delta = datetime.now() - start_date
        days = delta.days
        return days

    
    
#---------------------------------------------------------------------     
    
      

def csf(state):
    
    '''maps for competitive state fund'''
    
    csf = ['AZ', 'CA', 'CO', 
       'HI', 'ID', 'KY', 'LA', 'ME',
       'MD', 'MN', 'MO', 'MT', 'NM', 
       'NY', 'OK', 'OR', 'PA', 'RI', 'TX', 'UT'] 

    if state in csf:
        state = '1'
    else:
        state = '0'
    return(state)    

#---------------------------------------------------------------------   
   
'''Region dictionary for mapping'''

region = {'AK': 'western', 'AZ': 'western', 'CO': 'western', 'ID': 'western', 'MT': 'western',
          'NM': 'western','NV': 'western', 'OR': 'western', 'UT': 'western', 'WA': 'western', 
          'WY': 'western', 
          
          'IL': 'central', 'IN': 'central', 'IA': 'central', 'KY': 'central', 
          'MI': 'central', 'MN': 'central', 'NE': 'central', 'ND': 'central', 'OH': 'central', 
          'SD': 'central', 'WI': 'central',
          
          'AL': 'southern', 'FL': 'southern', 'GA': 'southern', 
          'MS': 'southern', 'NC': 'southern', 'SC': 'southern', 'TN': 'southern',
          
          'CT': 'northeast', 'MA': 'northeast',  'ME': 'northeast',  'NH': 'northeast',  
          'NY': 'northeast',  'RI': 'northeast',  'VT': 'northeast', 
         
          'DC': 'mid atlantic', 'DE': 'mid atlantic', 'MD': 'mid atlantic', 'NJ': 'mid atlantic', 
          'PA': 'mid atlantic', 'VA': 'mid atlantic', 'WV': 'mid atlantic',
         
          'AR': 'south central', 'KS': 'south central', 'LA': 'south central', 'MO': 'south central', 
          'OK': 'south central', 'TX': 'south central'}
    
#---------------------------------------------------------------------       
    
def market_sh(state):
    
    '''market share for states with competitve state funds'''
    
    high = ['CO', 'ID', 'MT', 'RI', 'ME', 'OR', 'UT'] #  > 50%
    med = ['HI', 'KY', 'NM', 'NY', 'OK', 'TX']  # > 25%
    low = ['CA', 'MN', 'PA', 'AZ', 'LA', 'MD', 'MO']  # > 0
    
    if state in high:
        state = 'high'
    elif state in med:
        state = 'med'
    elif state in low:
        state = 'low'
    else:
        state = 'none'
    return(state)

#---------------------------------------------------------------------   


def div(state):
    
    '''Div buckets for states with competitive state funds'''
    
    high = ['LA', 'MT', 'OR', 'TX'] #  > 18%
    med = ['CO', 'ME', 'NY', 'UT']  # > 5%
    low = ['AZ', 'CA', 'HI', 'ID', 'MD', 'MN', 'MO', 'RI']  # > 0
    
    if state in high:
        state = 'high'
    elif state in med:
        state = 'med'
    elif state in low:
        state = 'low'
    else:
        state = 'none'
    return(state)

#---------------------------------------------------------------------   

def converter(string):
    if string.split()[1] == 'Non-Renew':
        #print('yes')
        days = int(string.split()[0])
        if days >= 365 and days < 730:
            days = 365
        elif days >= 730:
            days = 730
        return days
        
    else:
        days = int(string.split()[0])
        return days
    
#---------------------------------------------------------------------           


def sub_df(dframe):
    
    '''returns a dataframe of only numerical values'''
    num_df = []
    
    for col in dframe.columns:
        x = dframe[col].iloc[0]
        if isinstance(x, np.float64) or isinstance(x, np.int64):
            num_df.append(col)
    
    number_df = dframe[num_df]
    return number_df

# -*- coding: utf-8 -*-

from collections import defaultdict
import pandas as pd

from datetime import datetime

#------------------------------------------------------------

'''dict mapping for ordinal feature'''

UW2rank = {'Mulloy,Spencer': 23,       
'Despain,Royd': 22,         
'Rodlin,Debbie': 21,      
'Armstrong,Ryan': 20,       
'Moralez,Randa': 19,        
'Dunn,Jonathan': 18,        
'Carmicheal,Debra': 17,     
'Chaudhuri,Debbie': 16,     
'Fuss,Amanda': 15,        
'Reeves,Heather': 14,      
'Harding,Kristine': 13,      
'Harris,Norm': 12,          
'Ruggeberg,Mike': 11,      
'Moore,David': 10,         
'Hilst,Scott': 9,            
'Furness,Nick': 8,          
'White,Becky': 7,           
'Dole,Kylie': 6,           
'Denney,Melanie': 5,         
'Eastwood,Jennifer': 4,      
'Costello,Cathleen': 3,      
'Sebolt,Catie': 2,           
'Rybeck,Chris': 1,           
}


#------------------------------------------------------------

def make_dict(col, dframe):
    
    '''returns a dict for freqs. This can then be mapped to 
    any col to create freq feature. Must be run prior to freq_group'''
    
    
    df = pd.DataFrame(dframe[col].value_counts())
    df.reset_index(level=0, inplace=True)
    df.rename(columns={'index': 'key', col: 'value'}, inplace=True)
    df_dict = defaultdict(list)
    for k, v in zip(df.key, df.value):
        df_dict[k] = (int(v))
    return df_dict


#------------------------------------------------------------

def freq_group(freq, _dict, rare, infrequent, less_common):
    
    '''run as lambda on col'''
    
    rev_dict = {v:k for k, v in _dict.items()}
    
    if freq <= rare:
        string = 'rare'
    elif freq > rare and freq <= infrequent:
        string = 'infrequent'
    elif freq > infrequent and freq <= less_common:
        string = 'less common'
    else:
        string = rev_dict[freq]
    return(string)

#------------------------------------------------------------



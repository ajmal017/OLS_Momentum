# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:23:12 2017

@author: yji
"""
import pandas as pd
import datetime




#Download from Google Finance
def googledata(symbollist,field):        
    
    
    dataset=pd.DataFrame()

    for ticker in symbollist:
    
        bars = web.DataReader(ticker, 'google',datetime.datetime(2016,1,1), datetime.datetime(2017,8,8))
        bars['ticker'] = ticker
        print ticker
        dataset=dataset.append(bars)

    output = dataset.pivot(columns = 'ticker', values =field)
    return output

    
def bdh_ohlc(symbol, start, end= datetime.date.today(), addfield = None, addfieldname = None):
    
    """
    download single security data from bloomberg.
    by default downlodas OHLC and volumes, can add additional fields 
    """
    from tia.bbg import LocalTerminal
    import pandas as pd
    import sys
    
    fields=['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST','EQY_WEIGHTED_AVG_PX']
    fieldnames=['Open','High','Low','Close','VWAP']

    if(addfield == None):
        if(addfieldname == None):
            pass
        else:
            sys.exit("additional fields and additoinal field names don't match!")
    else:
        field2=addfield
        fieldname2 = addfieldname
        fields = fields + field2
        fieldnames = fieldnames + fieldname2
    
    
    
    
    data=LocalTerminal.get_historical(symbol, fields, start, end)
    data=data.as_frame()
    data.columns=fieldnames
    #fill the nan

    
    return data
    
    
def bdhs(symbol, start, addfield, addfieldname, end= datetime.date.today(),):
    
    """
    Download any field without any default fields
    """
    import tia.bbg.datamgr as dm
    
    fields=addfield
    fieldnames=addfieldname
    
    mgr=dm.BbgDataManager()
    security=mgr[symbol]
    data=pd.DataFrame(security.get_historical(fields, start, end))
    data.columns=fieldnames
   
    return data
    
def bdps(symbol,field):
    
    """
    download current value for securities
    """
    
    
    from tia.bbg import LocalTerminal
    import pandas as pd
    
    data=LocalTerminal.get_reference_data(symbol,field)
    data=data.as_frame()
    
    return data
    
    
    
    
    
### Momentum Strategy


#Data Doownload

start = datetime.datetime(2002,1,1)
end = datetime.datetime(2017,10,12)


sx7e=bdh_ohlc('CA1 Index', start, end=end, addfield=['VOLATILITY_30D'], addfieldname=['30D Vol']).fillna(method='ffill').to_csv("sx7e.csv")
spx=bdh_ohlc('ES1 Index', start, end=end, addfield=['VOLATILITY_30D'], addfieldname=['30D Vol']).fillna(method='ffill').to_csv("spx.csv")
nikkei=bdh_ohlc('NO1 Index', start, end=end, addfield=['VOLATILITY_30D'], addfieldname=['30D Vol']).fillna(method='ffill').to_csv("nikkei.csv")
hks=bdh_ohlc('HU1 Index', start, end=end, addfield=['VOLATILITY_30D'], addfieldname=['30D Vol']).fillna(method='ffill').to_csv("hks.csv")

wti=bdh_ohlc('CL1 Comdty', start, end=end, addfield=['VOLATILITY_30D'], addfieldname=['30D Vol']).fillna(method='ffill').to_csv("wti.csv")
sugar=bdh_ohlc('SB1 Comdty', start, end=end, addfield=['VOLATILITY_30D'], addfieldname=['30D Vol']).fillna(method='ffill').to_csv("sugar.csv")

gold=bdh_ohlc('GC1 Comdty', start, end=end, addfield=['VOLATILITY_30D'], addfieldname=['30D Vol']).fillna(method='ffill').to_csv("gold.csv")
silver=bdh_ohlc('SI1 Comdty', start, end=end, addfield=['VOLATILITY_30D'], addfieldname=['30D Vol']).fillna(method='ffill').to_csv("silver.csv")

ty= bdh_ohlc('TY1 Comdty', start, end=end, addfield=['VOLATILITY_30D'], addfieldname=['30D Vol']).fillna(method='ffill').to_csv("ty.csv")                                  
rx= bdh_ohlc('RX1 Comdty', start, end=end, addfield=['VOLATILITY_30D'], addfieldname=['30D Vol']).fillna(method='ffill').to_csv("rx.csv")

eur= bdh_ohlc('EC1 Curncy', start, end=end, addfield=['VOLATILITY_30D'], addfieldname=['30D Vol']).fillna(method='ffill').to_csv("eur.csv")
jpy= bdh_ohlc('JY1 Curncy', start, end=end, addfield=['VOLATILITY_30D'], addfieldname=['30D Vol']).fillna(method='ffill').to_csv("jpy.csv")







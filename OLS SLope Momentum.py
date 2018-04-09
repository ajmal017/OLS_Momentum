# -*- coding: utf-8 -*-
"""


@author: traderji
"""
def OLS_Momentum_orders(securities, fast, slow, stop=100):
    """
    ------Long Short CTA strategy for futures-------
    """
    import pandas as pd
    import numpy as np
    
    fast_str = str(fast) + 'd Slope'
    slow_str = str(slow) + 'd Slope'

    trades = pd.DataFrame({"Price": [], "Regime": [], "Signal": []})
    
    for s in securities:
        
        #serial number
        s[1]["Nr"] = range(0, len(s[1]))
        
        #OLS for fast trend
        modelf=pd.stats.ols.MovingOLS(y=s[1]["Close"], x=s[1]["Nr"], window_type='rolling', window=fast, intercept=True)
        s[1][fast_str] = modelf.beta.x
        
        #OLS for slwo trend
        models=pd.stats.ols.MovingOLS(y=s[1]["Close"], x=s[1]["Nr"], window_type='rolling', window=slow, intercept=True)
        s[1][slow_str] = models.beta.x

       #Creating Regimes for long/short/flat positionings. Open when fast crosses slow, and close when price crosses fast.
        s[1]["Regime1"] = np.where(s[1][slow_str] > 0, 1, 0)
        s[1]["Regime1"] = np.where(s[1][slow_str] < 0, -1, s[1]["Regime1"])
        
        s[1]["Regime2"] = np.where(s[1][fast_str] > 0, 1, 0)
        s[1]["Regime2"] = np.where(s[1][fast_str] < 0, -1, s[1]["Regime2"])
        

        s[1]["Regime"] = (s[1]["Regime1"] + s[1]["Regime2"]) / 2
        
        s[1]["Signal"] = s[1]["Regime"] - s[1]["Regime"].shift(1)

        s[1]["Cost"]=np.where(s[1]["Signal"]<>0,s[1]["Open"],np.nan)
        s[1]["Cost"]=s[1]["Cost"].fillna(method='ffill')
            
        s[1]["PnL"]= (s[1]["Open"] - s[1]["Cost"]) * s[1]["Regime"] / s[1]["Open"] *100
        s[1]["StopLoss"] = np.where(s[1]["PnL"] < -1* stop, 1 , 0)
        
        # Get signals
        signals = pd.concat([
            pd.DataFrame({"Price": s[1].loc[(s[1]["Signal"].shift(1) == 1) & (s[1]["Regime1"].shift(1) == 1), "Open"],
                         "Regime": s[1].loc[(s[1]["Signal"].shift(1) == 1) & (s[1]["Regime1"].shift(1) == 1), "Regime"],
                         "Vol"   : s[1].loc[(s[1]["Signal"].shift(1) == 1) & (s[1]["Regime1"].shift(1) == 1), "30D Vol"],
                         "Symbol": s[0],   
                         "Signal": "BTO"}),
            pd.DataFrame({"Price": s[1].loc[(s[1]["Signal"].shift(1) == -1) & (s[1]["Regime1"].shift(1) == -1), "Open"],
                         "Regime": s[1].loc[(s[1]["Signal"].shift(1) == -1) & (s[1]["Regime1"].shift(1) == -1), "Regime"],
                         "Vol"   : s[1].loc[(s[1]["Signal"].shift(1) == -1) & (s[1]["Regime1"].shift(1) == -1), "30D Vol"],
                         "Symbol": s[0],  
                         "Signal": "STO"}),
            pd.DataFrame({"Price": s[1].loc[(s[1]["Signal"].shift(1) == -1) & (s[1]["Regime1"].shift(1) == 1), "Open"],
                         "Regime": s[1].loc[(s[1]["Signal"].shift(1) == -1) & (s[1]["Regime1"].shift(1) == 1), "Regime"],
                         "Vol"   : s[1].loc[(s[1]["Signal"].shift(1) == -1) & (s[1]["Regime1"].shift(1) == 1), "30D Vol"],
                         "Symbol": s[0],  
                         "Signal": "STC"}),
            pd.DataFrame({"Price": s[1].loc[(s[1]["Signal"].shift(1) == 1) & (s[1]["Regime1"].shift(1) == -1), "Open"],
                         "Regime": s[1].loc[(s[1]["Signal"].shift(1) == 1) & (s[1]["Regime1"].shift(1) == -1), "Regime"],
                         "Vol"   : s[1].loc[(s[1]["Signal"].shift(1) == 1) & (s[1]["Regime1"].shift(1) == -1), "30D Vol"],   
                         "Symbol": s[0],  
                         "Signal": "BTC"}),
            pd.DataFrame({"Price": s[1].loc[(s[1]["Regime"].shift(1) <> 0) & (s[1]["StopLoss"].shift(1) == 1), "Open"],
                         "Regime": s[1].loc[(s[1]["Regime"].shift(1) <> 0) & (s[1]["StopLoss"].shift(1) == 1), "Regime"],
                         "Vol"   : s[1].loc[(s[1]["Regime"].shift(1) <> 0) & (s[1]["StopLoss"].shift(1) == 1), "30D Vol"],   
                         "Symbol": s[0],  
                         "Signal": "STOP"}),              
            pd.DataFrame({"Price": s[1].loc[(s[1]["Signal"].shift(1) == 2) & (s[1]["Regime1"].shift(1) == 1), "Open"],
                         "Regime": s[1].loc[(s[1]["Signal"].shift(1) == 2) & (s[1]["Regime1"].shift(1) == 1), "Regime"],
                         "Vol"   : s[1].loc[(s[1]["Signal"].shift(1) == 2) & (s[1]["Regime1"].shift(1) == 1), "30D Vol"],   
                         "Symbol": s[0],  
                         "Signal": "BTCBTO"}),
            pd.DataFrame({"Price": s[1].loc[(s[1]["Signal"].shift(1) == -2) & (s[1]["Regime1"].shift(1) == -1), "Open"],
                         "Regime": s[1].loc[(s[1]["Signal"].shift(1) == -2) & (s[1]["Regime1"].shift(1) == -1), "Regime"],
                         "Vol"   : s[1].loc[(s[1]["Signal"].shift(1) == -2) & (s[1]["Regime1"].shift(1) == -1), "30D Vol"],   
                         "Symbol": s[0],  
                         "Signal": "STCSTO"}),
        ])

        trades = trades.append(signals)
 
    trades.sort_index(inplace = True)

    
 
    return trades
 
    
    
def backtest_fut(signals, cash, max_leverage=3, batch = 1, trancost=0, slippage=0):
    """
    :param signals: pandas DataFrame containing buy and sell signals with stock prices and symbols, like that returned by ma_crossover_orders
    :param cash: integer for starting cash value
    :param max_leverage: maximum level of leverage for each future to take on, based on margin requirements
    :param batch: Trading batch sizes
 
    :return: pandas DataFrame with backtesting results
 
    This function backtests strategies, with the signals generated by the strategies being passed in the signals DataFrame. A fictitious portfolio is simulated and the returns generated by this portfolio are reported.
    """
    
    import numpy as np
    import pandas as pd
 
    SYMBOL = 1 # Constant for which element in index represents symbol
    portfolio = dict()    # Will contain how many stocks are in the portfolio for a given symbol
    port_prices = dict()  # Tracks old trade prices for determining profits
    # Dataframe that will contain backtesting report
    
    results = pd.DataFrame({"Start Cash": [],
                            "End Cash": [],
                            "Portfolio Value": [],
                            "Type": [],
                            "Symbol": [],
                            "Contracts": [],
                            "future Price": [],
                            "Avg Cost": [],
                            "Trade Value": [],
                            "Profit per Share": [],
                            "PMV" : [],
                            "Total Profit": []})

    for index, row in signals.iterrows():
        # Initiation 
        contracts = portfolio.setdefault(row["Symbol"], 0)
        trade_val = 0
        batches = 0 #numbers of contract to be traded under the leverage limit
       # cash_change = row["Price"] * contracts   # contracts could potentially be a positive or negative number (cash_change will be added in the end; negative indicate a short)
        portfolio[row["Symbol"]] = 0  # For a given symbol, a position is effectively cleared
 
        old_price = port_prices.setdefault(row["Symbol"], row["Price"])
        portfolio_val = 0
        pprofit=0
        # Adding trades:
            
      #  for key, val in portfolio.items():
       #     portfolio_val += val * port_prices[key]
 
        if row["Signal"] == "BTO":  # Entering a long position
            batches = np.floor(cash * max_leverage  / ( batch * row["Psize"] * (1 + slippage) * row["Total Vol"] * row["Price"] * row ["Vol"]))# Maximum number of batches of stocks invested in
            trade_val = batches * batch * row["Psize"] * row["Price"] # How much money is put on the line with each trade
            #cash_change -= trade_val  # We are buying shares so cash will go down
            portfolio[row["Symbol"]] = batches   # Recording how many contracts are currently long
            port_prices[row["Symbol"]] = row["Price"]   # Record price
            old_price = row["Price"]  * (1 + slippage)
            tprofit =  - np.absolute(trancost * batches)

        elif row["Signal"] == "STO": # Entering a short
            batches= - np.floor(cash * max_leverage / ( batch * row["Psize"] * (1 - slippage) * row["Price"] * row["Total Vol"] * row ["Vol"]))
            trade_val = batches * row["Psize"] * batch * row["Price"] # How much money is put on the line with each trade
            portfolio[row["Symbol"]] = batches  # Recording the current open positions
            port_prices[row["Symbol"]] = row["Price"]   # Record price
            old_price = row["Price"] * (1 - slippage)
            tprofit =  - np.absolute(trancost * batches)

        elif row["Signal"] == "STC" or row["Signal"] == "BTC":
            batches= - contracts
            trade_val = batches * batch * row["Psize"] * row["Price"] # How much money is put on the line with each trade
            pprofit = (row["Price"] - old_price)*np.sign(contracts)  # Compute profit per share; old_price is set in such a way that entering a position results in a profit of zero
            tprofit = pprofit * np.absolute(batches) * batch* row["Psize"] - np.absolute(trancost * batches)

        elif row["Signal"] == "BTCBTO":
            batches= - contracts
            pprofit = (row["Price"] - old_price)*np.sign(contracts)  # Compute profit per share; old_price is set in such a way that entering a position results in a profit of zero
            tprofit = pprofit * np.absolute(batches) * batch* row["Psize"] - np.absolute(trancost * batches)
           
            batches = np.floor(cash * max_leverage  / ( batch * row["Psize"] * (1 + slippage) * row["Total Vol"] * row["Price"] * row ["Vol"]))# Maximum number of batches of stocks invested in
            trade_val = batches * batch * row["Psize"] * row["Price"] # How much money is put on the line with each trade
            portfolio[row["Symbol"]] = batches   # Recording how many contracts are currently long
            port_prices[row["Symbol"]] = row["Price"]   # Record price
            old_price = row["Price"]  * (1 + slippage)                
       
            
        elif row["Signal"] == "STCSTO":
            batches= - contracts
            pprofit = (row["Price"] - old_price)*np.sign(contracts)  # Compute profit per share; old_price is set in such a way that entering a position results in a profit of zero
            tprofit = pprofit * np.absolute(batches) * batch* row["Psize"] - np.absolute(trancost * batches)
            
            batches= - np.floor(cash * max_leverage / ( batch * row["Psize"] * (1 - slippage) * row["Price"] * row["Total Vol"] * row ["Vol"]))
            trade_val = batches * row["Psize"] * batch * row["Price"] # How much money is put on the line with each trade
            portfolio[row["Symbol"]] = batches  # Recording the current open positions
            port_prices[row["Symbol"]] = row["Price"]   # Record price
            old_price = row["Price"] * (1 - slippage)
            
        elif row["Signal"] == "STOP":
            batches= - contracts
            pprofit = (row["Price"] - old_price)*np.sign(contracts)  # Compute profit per share; old_price is set in such a way that entering a position results in a profit of zero
            tprofit = pprofit * np.absolute(batches) * batch* row["Psize"] - np.absolute(trancost * batches)
            portfolio[row["Symbol"]] = 0  # Recording the current open positions
             
            
            
        # Update report
        results = results.append(pd.DataFrame({
                "Start Cash": cash,
                "End Cash": cash + tprofit,
                "Portfolio Value": portfolio_val + trade_val,
                "Type": row["Signal"],
                "Symbol": row["Symbol"],
                "Contracts": batches,
                "future Price": row["Price"],
                "Avg Cost": old_price,
                "Trade Value": np.absolute(batches) * batch * row["Psize"] * row["Price"],
                "Profit per Share": pprofit,
                "PMV": 1 / (row ["Vol"] * row["Total Vol"]),
                "Total Profit": tprofit
            }, index = [index]))
        cash += tprofit  # Final change to cash balance
        cash=max(cash,0)
        
            
        
    results.sort_index(inplace = True)

 
    return results  
    
    
    

import datetime
import pandas as pd
import datetime
import numpy as np
import tia.bbg.datamgr as dm
from ma_crossover_orders import ma_crossover_orders_fut
from data_downloader import bdh_ohlc
from data_downloader import bdps
from backtest import backtest_fut


start = datetime.datetime(2017,6,6)
end = datetime.datetime.today()


#Data Doownload
sx7e=bdh_ohlc('GX1 Index', start, end=end, addfield=["Volatility_30D"], addfieldname=["30D Vol"]).fillna(method='ffill')
spx=bdh_ohlc('ES1 Index', start, end=end , addfield=["Volatility_30D"], addfieldname=["30D Vol"]).fillna(method='ffill')
gold=bdh_ohlc('GC1 Comdty', start, end=end , addfield=["Volatility_30D"], addfieldname=["30D Vol"]).fillna(method='ffill')
sugar=bdh_ohlc('SB1 Comdty', start, end=end , addfield=["Volatility_30D"], addfieldname=["30D Vol"]).fillna(method='ffill')
ty= bdh_ohlc('TY1 Comdty', start, end=end , addfield=["Volatility_30D"], addfieldname=["30D Vol"]).fillna(method='ffill')   
rx= bdh_ohlc('RX1 Comdty', start, end=end , addfield=["Volatility_30D"], addfieldname=["30D Vol"]).fillna(method='ffill')
ng= bdh_ohlc('NG1 Comdty', start, end=end , addfield=["Volatility_30D"], addfieldname=["30D Vol"]).fillna(method='ffill')
silver= bdh_ohlc('SI1 Comdty', start, end=end , addfield=["Volatility_30D"], addfieldname=["30D Vol"]).fillna(method='ffill')
                               

cash= 100000
fast1=100
slow1=300
string=str(fast1) + '/' + str(slow1)

#contract sizes
psize=pd.DataFrame([50,1000,50,100,1000,1120,1000,10000,5000], index=["SX7E","WTI","SPX","GOLD", "RX", "SUGAR", "TY", "NG", "SILVER" ])
psize.columns=["FUT_VAL_PT"]


signals = OLS_Momentum_orders([("WTI", wti),
                                   ("SX7E",sx7e),
                                   ("GOLD",gold),
                                   ("NG",ng),
                                   ("SILVER",silver),
                                   ("SPX",spx), 
                                   ("SUGAR",sugar), 
                                   ("RX",rx), 
                                   ("TY", ty)], 
                                   fast = fast1, slow= slow1, stop=30)

signals=signals[10:]


total_vol= 1/wti["30D Vol"] + 1/sx7e["30D Vol"] + 1/spx["30D Vol"] + 1/gold["30D Vol"] + 1/sugar["30D Vol"]  + 1/rx["30D Vol"] + 1/ng["30D Vol"] + 1/silver["30D Vol"] + 1/ty["30D Vol"]
total_vol=pd.DataFrame(total_vol).fillna(method='ffill') 
signals["Psize"]=signals.Symbol.map(psize.FUT_VAL_PT)
signals['Psize']=pd.to_numeric(signals['Psize'])
signals=signals.merge(total_vol,left_index=True, right_index=True, how='left')
signals.columns=signals.columns.str.replace('30D', 'Total')


bk = backtest_fut(signals, cash, max_leverage=20, trancost=3)


bogie = spx

ax_bench = (bogie["Close"] / bogie.ix[0, "Close"]).plot(label = "SPX")
ax_bench = (bk["End Cash"].groupby(level = 0).apply(lambda x: x[-1]) / cash).plot(label = "30 SL - Portfolio %s" % string)
ax_bench.legend(ax_bench.get_lines(), [l.get_label() for l in ax_bench.get_lines()], loc = 'best')
ax_bench.set_yscale('log')
ax_bench


#Analytics:
class portAnalytics(object):
    
    def __init__(self, ret, index):
        self.ret = ret
        self.index = index
    
    def SR(self):
        daily_ret=self.ret["End Cash"].pct_change()
        srnr=np.sqrt(((self.ret.index[-1]-self.ret.index[1]).days)/len(daily_ret))*daily_ret.mean()/daily_ret.std()
        print srnr
        return srnr
        
        
daily_ret=bk["End Cash"].pct_change()
sr=np.sqrt(len(bk))*daily_ret.mean()/daily_ret.std()
print "Sharpe Ratio = %s" %(round(sr,2))


winner=bk.loc[(bk["Profit per Share"]>0),"Profit per Share"].count()
loser=bk.loc[(bk["Profit per Share"]<0),"Profit per Share"].count()
win_rate=winner*100/(winner+loser)
print "win rate= %s percent" %(win_rate) 

winner=(bk.loc[(bk["Profit per Share"]>0),"Total Profit"]/bk.loc[(bk["Profit per Share"]>0),"Start Cash"]).mean()
loser=(bk.loc[(bk["Profit per Share"]<0),"Total Profit"]/bk.loc[(bk["Profit per Share"]<0),"Start Cash"]).mean()
print "averge win size pct %s" %round(winner*100,2) 
print "averge lose size pct %s" %round(loser*100,2) 

cagr=(bk.iloc[-1]["End Cash"]/bk.iloc[0]["End Cash"])**(365.25/(end-start).days)*100-100
print "CAGR= %s" %(round(cagr,2))


def max_dd(ser):
    import pandas as pd
    max2here = pd.expanding_max(ser)
    dd2here = ser - max2here
    return dd2here.min()
      
print "max draw down is:  %s" % (round(max_dd(bk["End Cash"].pct_change())*100,2))


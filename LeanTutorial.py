# Equity
class EQBuyHold(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2017, 6, 1)
        self.SetEndDate(2017, 6, 15)

        #1,2. Select IWM minute resolution data and set it to Raw normalization mode
        self.iwm = self.AddEquity("IWM", Resolution.Minute)
        self.iwm.SetDataNormalizationMode(DataNormalizationMode.Raw)

    def OnData(self, data):

        #3. Place an order for 100 shares of IWM and print the average fill price
        #4. Debug the AveragePrice of IWM
        if not self.Portfolio.Invested:
            self.MarketOrder('IWM', 100)
            self.Debug(str(self.Portfolio['IWM'].AveragePrice))

# Forex
class FXBuyHold(QCAlgorithm):

    def Initialize(self):
        self.SetCash(100000)
        self.SetStartDate(2017, 5, 1)
        self.SetEndDate(2017, 5, 31)
        
        #1. Request the forex data
        self.AddForex("AUDUSD", Resolution.Hour, Market.Oanda)
        
        #2. Set the brokerage model
        self.SetBrokerageModel(BrokerageName.OandaBrokerage)
        
    def OnData(self, data):
       
        #3. Using "Portfolio.Invested" submit 1 order for 2000 AUDUSD:
        if not self.Portfolio.Invested:
            self.MarketOrder("AUDUSD", 2000)

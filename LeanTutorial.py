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

# Buy and Hold with a Trailing Stop

class BootCampTask(QCAlgorithm):
    
    # Order ticket for our stop order, Datetime when stop order was last hit
    stopMarketTicket = None
    stopMarketOrderFillTime = datetime.min
    highestSPYPrice = -1
    
    def Initialize(self):
        self.SetStartDate(2018, 12, 1)
        self.SetEndDate(2018, 12, 10)
        self.SetCash(100000)
        spy = self.AddEquity("SPY", Resolution.Daily)
        spy.SetDataNormalizationMode(DataNormalizationMode.Raw)
        
    def OnData(self, data):
        
        # 1. Plot the current SPY price to "Data Chart" on series "Asset Price"
        self.Plot("Data Chart", "Asset Price", data["SPY"].Close)

        if (self.Time - self.stopMarketOrderFillTime).days < 15:
            return

        if not self.Portfolio.Invested:
            self.MarketOrder("SPY", 500)
            self.stopMarketTicket = self.StopMarketOrder("SPY", -500, 0.9 * self.Securities["SPY"].Close)
        
        else:
            
            #2. Plot the moving stop price on "Data Chart" with "Stop Price" series name
            self.Plot("Data Chart", "Stop Price", self.highestSPYPrice * 0.9)
            
            if self.Securities["SPY"].Close > self.highestSPYPrice:
                
                self.highestSPYPrice = self.Securities["SPY"].Close
                updateFields = UpdateOrderFields()
                updateFields.StopPrice = self.highestSPYPrice * 0.9
                self.stopMarketTicket.Update(updateFields) 
            
    def OnOrderEvent(self, orderEvent):
        
        if orderEvent.Status != OrderStatus.Filled:
            return
        
        if self.stopMarketTicket is not None and self.stopMarketTicket.OrderId == orderEvent.OrderId: 
            self.stopMarketOrderFillTime = self.Time
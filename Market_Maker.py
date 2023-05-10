from typing import Dict, List

import numpy as np
from datamodel import OrderDepth, TradingState, Order, Trade

Time = int
Symbol = str
Product = str
Position = int
UserId = str
Observation = int


class Markov:
    def __init__(self, datax: [np.array], datay: np.array = None):
        if not type(datax) is list:
            datax = [datax]
        self.datax = []
        for i in range(len(datax)):
            try:
                self.datax.append(datax[i].to_numpy())
            except AttributeError:
                self.datax.append(datax[i])
        if datay is None:
            self.datay = self.datax[0][1:]
            for i in range(len(self.datax)):
                self.datax[i] = self.datax[i][:-1]
        else:
            try:
                self.datay = datay.to_numpy()
            except AttributeError:
                self.datay = datay

        # print(self.datax[0], self.datax)
        self.minx = np.zeros(len(datax), dtype="int32")
        self.maxx = np.zeros(len(datax), dtype="int32")
        for i in range(len(self.datax)):
            self.minx[i] = int(np.nanmin(self.datax[i]))
            self.maxx[i] = int(np.nanmax(self.datax[i]))
        self.miny = int(np.nanmin(self.datay))
        self.maxy = int(np.nanmax(self.datay))
        shape = self.markov_shape()
        self.markov = np.zeros(shape)
        self.markov_raw = np.zeros(shape)
        self.datax_list = []
        for i in range(len(self.datax)):
            self.datax_list.append(self.datax[i].tolist())
        self.datay_list = self.datay.tolist()

        # print(self.minx[0])
        self.count()

    def markov_shape(self):
        shape = []
        for i in range(len(self.datax)):
            shape.append(int(self.maxx[i] - self.minx[i] + 2))  # +2 because index max will be the nans
        shape.append(int(self.maxy - self.miny + 1))
        return shape

    def count(self):
        for i in range(len(self.datax)):  # replace nans with maxx + 1
            self.datax[i][np.isnan(self.datax[i])] = self.maxx[i] + 1
        for i in range(len(self.datax[0])):
            index = []
            for j in range(len(self.datax)):
                index.append(int(self.datax[j][i] - self.minx[j]))
            index.append(int(self.datay[i] - self.miny))
            self.markov_raw[tuple(index)] += 1

    def append_list(self, x: [int], index: int):
        if not (type(x[index]) is list):
            x[index] = [x[index]]
        not_added = True
        if self.maxx[index] - self.minx[index] + 1 in x[index]:
            x.pop()
            return x, index - 1
        if max(x[index]) + 1 <= self.maxx[index] - self.minx[index]:
            x[index].append(max(x[index]) + 1)
            not_added = False
        if min(x[index]) - 1 >= 0:
            x[index].append(min(x[index]) - 1)
            not_added = False
        if not_added:
            # if self.maxx[index] + 1 in x[index]:
            #    return x, index - 1
            x[index].append(self.maxx[index] - self.minx[index] + 1)
        return x, index

    def distribution(self, x: [int], minimum: int = 10):
        if not type(x) is list:
            x = [x]
        for i in range(len(x)):
            if x[i] < self.minx[i]:
                x[i] = self.minx[i]
            elif x[i] > self.maxx[i]:
                x[i] = self.maxx[i]
            elif np.isnan(x[i]):
                x[i] = self.maxx[i] + 1
            x[i] = int(x[i] - self.minx[i])
        index = len(x) - 1
        while minimum > np.sum(self.markov_raw[tuple(x)]) and len(x) > 0:
            x, index = self.append_list(x, index)  # TODO: Change that its on actual length not minx things
            # x = x[:len(x) - 1]
        y = self.markov_raw[tuple(x)].copy()
        while len(y.shape) != 1:
            y = np.sum(y, axis=0)
        return y / np.sum(y)

    def average(self, x: [int],
                minimum: int = 10):  # TODO: Change that average of nearest values will be taken, so if [2, 3] is not enough, it will be [2,2],[2,3],[2,4]
        if not type(x) is list:
            x = [x]
        dis = self.distribution(x, minimum=minimum)
        average = 0
        for i in range(len(dis)):
            average += dis[i] * (i + self.miny)
        return average

    def update(self, x: [int], y: int):
        assert not np.isnan(y)
        if not type(x) is list:
            x = [x]
        for i in range(len(x)):
            self.datax_list[i].append(x[i])
        self.datay_list.append(y)
        rebuild = False
        for i in range(len(x)):
            if np.isnan(x[i]):
                x[i] = self.maxx[i] + 1
            else:
                if not (self.minx[i] <= x[i] <= self.maxx[i]):
                    rebuild = True
                    self.minx[i] = min(x[i], self.minx[i])
                    self.maxx[i] = max(x[i], self.maxx[i])
        if not (self.miny <= y <= self.maxy):
            rebuild = True
            self.miny = min(y, self.miny)
            self.maxy = max(y, self.maxy)
        if rebuild:
            self.datax = []
            for i in range(len(self.datax_list)):
                self.datax.append(np.array(self.datax_list[i]))
            self.datay = np.array(self.datay_list)
            shape = self.markov_shape()
            self.markov = np.zeros(shape)
            self.markov_raw = np.zeros(shape)
            self.count()
        else:
            x.append(int(y - self.miny))
            for i in range(len(x) - 1):
                x[i] = int(x[i] - self.minx[i])
            self.markov_raw[tuple(x)] += 1


class Trader:

    def __init__(self):
        self.min_sell = None
        self.second_min_sell = None
        self.sell_list = None
        self.max_buy = None
        self.second_max_buy = None
        self.buy_list = None
        self.markov_ask_coco = None
        self.markov_bid_coco = None
        self.markov_bid_pearls = None
        self.markov_ask_bananas = None
        self.markov_bid_bananas = None
        self.markov_ask_pearls = None
        self.markov_ratio = None
        self.symbols = None
        self.state = None
        self.result = {}
        self.limits = {"PEARLS": 20, "BANANAS": 20, "COCONUTS": 600, "PINA_COLADAS": 300, "BERRIES": 250,
                       "DIVING_GEAR": 50, "BAGUETTE": 350, "DIP": 300, "UKULELE": 70, "PICNIC_BASKET": 70}
        self.bid_history_bananas = np.zeros(100000, dtype="int32")
        self.bid_history_bananas_second = np.zeros(100000)
        self.ask_history_bananas = np.zeros(100000, dtype="int32")
        self.ask_history_bananas_second = np.zeros(100000)
        self.ratio_history_precise = np.zeros(100000)
        self.ratio_history_floored = np.zeros(100000)
        self.ratio_history_floored_rough = np.zeros(100000)

        self.new_assets = {"PEARLS": {"bid": np.zeros(100000, dtype="int32"), "ask": np.zeros(100000, dtype="int32"),
                                      "bid_second": np.zeros(100000), "ask_second": np.zeros(100000)},
                           "BANANAS": {"bid": np.zeros(100000, dtype="int32"), "ask": np.zeros(100000, dtype="int32"),
                                       "bid_second": np.zeros(100000), "ask_second": np.zeros(100000)},
                           "COCONUTS": {"bid": np.zeros(100000, dtype="int32"), "ask": np.zeros(100000, dtype="int32"),
                                        "bid_second": np.zeros(100000), "ask_second": np.zeros(100000)},
                           "PINA_COLADAS": {"bid": np.zeros(100000, dtype="int32"),
                                            "ask": np.zeros(100000, dtype="int32"), "bid_second": np.zeros(100000),
                                            "ask_second": np.zeros(100000)},
                           "BERRIES": {"bid": np.zeros(100000, dtype="int32"), "ask": np.zeros(100000, dtype="int32"),
                                       "bid_second": np.zeros(100000), "ask_second": np.zeros(100000)},
                           "DIVING_GEAR": {"bid": np.zeros(100000, dtype="int32"),
                                           "ask": np.zeros(100000, dtype="int32"), "bid_second": np.zeros(100000),
                                           "ask_second": np.zeros(100000)},
                           "BAGUETTE": {"bid": np.zeros(100000, dtype="int32"), "ask": np.zeros(100000, dtype="int32"),
                                        "bid_second": np.zeros(100000), "ask_second": np.zeros(100000)},
                           "DIP": {"bid": np.zeros(100000, dtype="int32"), "ask": np.zeros(100000, dtype="int32"),
                                   "bid_second": np.zeros(100000), "ask_second": np.zeros(100000)},
                           "UKULELE": {"bid": np.zeros(100000, dtype="int32"), "ask": np.zeros(100000, dtype="int32"),
                                       "bid_second": np.zeros(100000), "ask_second": np.zeros(100000)},
                           "PICNIC_BASKET": {"bid": np.zeros(100000, dtype="int32"),
                                             "ask": np.zeros(100000, dtype="int32"), "bid_second": np.zeros(100000),
                                             "ask_second": np.zeros(100000)}}

        self.filled_till = 0
        self.bid_history_pearls = np.zeros(100000, dtype="int32")
        self.bid_history_pearls_second = np.zeros(100000)
        self.ask_history_pearls = np.zeros(100000, dtype="int32")
        self.ask_history_pearls_second = np.zeros(100000)

        self.dol_history = np.zeros(100000)
        self.extreme_iter = 100000  # so not in data included, because there wasnt one yet
        self.extreme_diff = 0  # there wasnt an extreme diff yet

        self.average_pina = 1.8758250007866768
        self.mini_pina = 1.863512509289076
        self.maxi_pina = 1.890557939914163

        self.average_etf = 1.0051022908343832
        self.mini_etf = 0.9997555576680202
        self.maxi_etf = 1.0105567783484144

        self.not_yet_init = True

    def append_as_ones(self, ticker: str, price: int, volume: int):
        if volume > 0:
            for i in range(volume):
                self.result[ticker].append(Order(ticker, price, 1))
        else:
            for i in range(-volume):
                self.result[ticker].append(Order(ticker, price, -1))

    def minimum_diff(self, max_bid, my_bid, min_ask, my_ask):
        if max_bid < my_bid:
            my_bid -= my_bid - max_bid + 1
        if min_ask > my_ask:
            my_ask += min_ask - my_ask - 1
        return my_bid, my_ask

    def assert_ask_bid_diff(self, ask, bid):
        while bid >= ask:
            ask += 1
            bid -= 1
        return ask, bid

    def find_multiplier_percentile_pina(self, value: float):
        if value >= self.average_pina:
            step = self.maxi_pina - self.average_pina
            for i in range(1, 101):
                if self.average_pina + (i / 100) * step > value:
                    return (i / 100)
        if value < self.average_pina:
            step = self.mini_pina - self.average_pina
            for i in range(1, 101):
                if self.average_pina + (i / 100) * step < value:
                    return -(i / 100)

    def find_multiplier_percentile_etf(self, value: float):
        if value >= self.average_etf:
            step = self.maxi_etf - self.average_etf
            for i in range(1, 101):
                if self.average_etf + (i / 100) * step > value:
                    return (i / 100)
        if value < self.average_etf:
            step = self.mini_etf - self.average_etf
            for i in range(1, 101):
                if self.average_etf + (i / 100) * step < value:
                    return -(i / 100)

    def price_management_pina(self, wanna_trade, expected_bid, expected_ask):
        if wanna_trade > 0:
            expected_bid += int(wanna_trade / 10)
            if expected_ask < expected_bid:
                expected_bid = expected_ask
            return expected_bid
        elif wanna_trade < 0:
            expected_ask += int(wanna_trade/ 10)
            if expected_ask < expected_bid:
                expected_ask = expected_bid
            return expected_ask

    def market_making_berries(self, ask, bid, wanna_trade, wanna_be_position):
        interval = 30
        if wanna_trade <= 0:
            max_allowed_long = int(
                min(wanna_be_position + interval, self.limits["BERRIES"]) - self.state.position["BERRIES"])
            # market_make = min(int(max_allowed_long / 20), -wanna_trade)
            market_make = int(max_allowed_long)
            max_additioanl_short = int(-self.limits["BERRIES"] - self.state.position["BERRIES"] - wanna_trade)
            # max_additioanl_short = max(wanna_be_position-self.state.position["BERRIES"], max_additioanl_short)
            go_additional_short = min(int(max_additioanl_short / 10), 0)
            if wanna_trade < -interval:
                ask += int(wanna_trade/interval)
            # print("BERRIES <0: ", ask, bid)
            print("BERRIES <0 with add: ", wanna_trade, "  ", go_additional_short)
            self.append_as_ones("BERRIES", ask - 1, int((wanna_trade + go_additional_short) / 4))
            self.append_as_ones("BERRIES", ask, int((wanna_trade + go_additional_short) / 2))
            self.append_as_ones("BERRIES", ask + 1,
                                wanna_trade + go_additional_short - int((wanna_trade + go_additional_short) / 2) - int(
                                    (wanna_trade + go_additional_short) / 4))
            if market_make > 0:
                print("MARKET MAKE wanna_trade < 0 ", market_make)
                self.append_as_ones("BERRIES", bid + 1, int(market_make / 4))
                self.append_as_ones("BERRIES", bid, int(market_make / 2))
                self.append_as_ones("BERRIES", bid - 1,
                                    market_make - int(market_make / 2) - int(market_make / 4))
        elif wanna_trade > 0:
            # max_allowed_short = int(-self.limits["BERRIES"] - self.state.position["BERRIES"])
            max_allowed_short = int(
                max(wanna_be_position - interval, -self.limits["BERRIES"]) - self.state.position["BERRIES"])

            # market_make = max(int(max_allowed_short/20), -wanna_trade)
            market_make = int(max_allowed_short)
            max_additional_long = int(self.limits["BERRIES"] - self.state.position["BERRIES"] - wanna_trade)
            go_additional_long = max(int(max_additional_long / 10), 0)
            # print("BERRIES >0: ", ask, bid)
            print("BERRIES >0 with add: ", wanna_trade, "  ", go_additional_long)

            if wanna_trade > interval:
                bid += int(wanna_trade/interval)
            if market_make < 0:
                print("MARKET MAKE wanna_trade > 0 ", market_make)
                self.append_as_ones("BERRIES", ask - 1, int(market_make / 4))
                self.append_as_ones("BERRIES", ask, int(market_make / 2))
                self.append_as_ones("BERRIES", ask + 1,
                                    market_make - int(market_make / 2) - int(market_make / 4))
            self.append_as_ones("BERRIES", bid + 1, int((wanna_trade + go_additional_long) / 4))
            self.append_as_ones("BERRIES", bid, int((wanna_trade + go_additional_long) / 2))
            self.append_as_ones("BERRIES", bid - 1,
                                (wanna_trade + go_additional_long) - int((wanna_trade + go_additional_long) / 2) - int(
                                    (wanna_trade + go_additional_long) / 4))

    def berries_manangement(self, ask, bid):
        long_delimiter = 4500  # 4500
        short_delimiter = 5500  # 5500
        if self.filled_till < long_delimiter:
            wanna_be_position = int((self.limits["BERRIES"] * self.filled_till) / long_delimiter)
            wanna_trade = wanna_be_position - self.state.position["BERRIES"]
            print("WANNA POSITION", wanna_be_position)
            # print("WANNA TRADE", wanna_trade)
            self.market_making_berries(ask, bid, wanna_trade=wanna_trade, wanna_be_position=wanna_be_position)
        elif self.filled_till > short_delimiter:
            wanna_be_position = -int(self.limits["BERRIES"])
            wanna_trade = wanna_be_position - self.state.position["BERRIES"]
            self.market_making_berries(ask, bid, wanna_trade=wanna_trade, wanna_be_position=wanna_be_position)
            print("WANNA POSITION", wanna_be_position)
            # print("WANNA TRADE", wanna_trade)
        else:
            wanna_be_position = -int(
                (2 * (self.filled_till - long_delimiter) / (short_delimiter - long_delimiter) - 1) * self.limits[
                    "BERRIES"])
            print("WANNA POSITION First", wanna_be_position)
            if wanna_be_position < 0:
                wanna_be_position = max(-self.limits["BERRIES"], wanna_be_position)
            else:
                wanna_be_position = min(self.limits["BERRIES"], wanna_be_position)
            # print("WANNA POSITION", wanna_be_position)
            wanna_trade = wanna_be_position - self.state.position["BERRIES"]
            # print(wanna_trade)
            self.market_making_berries(ask, bid, wanna_trade=wanna_trade, wanna_be_position=wanna_be_position)

        # those input prices are the prices I predict it to be next iteration

    def position_management_pina_coco(self, ask_coco, bid_coco, ask_pina, bid_pina,
                                      multi):  # TODO: Look if bid/ask average is better indikator than Markov with Indikator best-second
        # print("ASK BEGIN: ", ask_coco)
        pina_factor = 1.3
        # TODO: Update Average
        # multi = float(self.new_assets["PINA_COLADAS"]["ask"][self.filled_till-1]) / float(self.new_assets["COCONUTS"]["ask"][self.filled_till-1])
        if self.mini_pina > multi:
            self.mini_pina = multi
        elif self.maxi_pina < multi:
            self.maxi_pina = multi
        percentile = self.find_multiplier_percentile_pina(multi)
        if percentile > 0:
            percentile = min(1, percentile * pina_factor)
        else:
            percentile = max(-1, percentile * pina_factor)
        wanna_be_position_pina = -int(self.limits["PINA_COLADAS"] * percentile)
        wanna_be_position_coco = int(self.average_pina * self.limits["PINA_COLADAS"] * percentile)

        trades_wanna_do_pina = wanna_be_position_pina - self.state.position["PINA_COLADAS"]
        trades_wanna_do_coco = wanna_be_position_coco - self.state.position["COCONUTS"]
        # print("WANNA BE THERE: COCO PINA", wanna_be_position_coco, wanna_be_position_pina)
        # print("PERCENTILE COCO PINA", percentile, trades_wanna_do_coco, trades_wanna_do_pina)

        if trades_wanna_do_coco < 0:
            # if ask_coco - 1 > bid_coco:
            #    ask_coco -= 1
            ask_coco = self.price_management_pina(trades_wanna_do_coco, bid_coco, ask_coco)
            # print("ASK: ", self.min_sell["COCONUTS"], ask_coco)
            self.append_as_ones("COCONUTS", ask_coco, int(trades_wanna_do_coco / 4))
            self.append_as_ones("COCONUTS", ask_coco + 1, int(trades_wanna_do_coco / 2))
            self.append_as_ones("COCONUTS", ask_coco + 2,
                                trades_wanna_do_coco - int(trades_wanna_do_coco / 2) - int(trades_wanna_do_coco / 4))
        elif trades_wanna_do_coco > 0:
            # if bid_coco + 1 < ask_coco:
            #    bid_coco += 1
            bid_coco = self.price_management_pina(trades_wanna_do_coco, bid_coco, ask_coco)
            # print("BID: ", self.max_buy["COCONUTS"], bid_coco)
            self.append_as_ones("COCONUTS", bid_coco, int(trades_wanna_do_coco / 4))
            self.append_as_ones("COCONUTS", bid_coco - 1, int(trades_wanna_do_coco / 2))
            self.append_as_ones("COCONUTS", bid_coco - 2,
                                trades_wanna_do_coco - int(trades_wanna_do_coco / 2) - int(trades_wanna_do_coco / 4))
        if trades_wanna_do_pina < 0:
            # if ask_pina - 1 > bid_pina:
            #    ask_pina -= 1
            ask_pina = self.price_management_pina(trades_wanna_do_pina, bid_pina, ask_pina)
            self.append_as_ones("PINA_COLADAS", ask_pina, int(trades_wanna_do_pina / 4))
            self.append_as_ones("PINA_COLADAS", ask_pina + 1, int(trades_wanna_do_pina / 2))
            self.append_as_ones("PINA_COLADAS", ask_pina + 2,
                                trades_wanna_do_pina - int(trades_wanna_do_pina / 4) - int(trades_wanna_do_pina / 2))
        elif trades_wanna_do_pina > 0:
            # if bid_pina + 1 < ask_pina:
            #    bid_pina += 1
            bid_pina = self.price_management_pina(trades_wanna_do_pina, bid_pina, ask_pina)
            self.append_as_ones("PINA_COLADAS", bid_pina, int(trades_wanna_do_pina / 4))
            self.append_as_ones("PINA_COLADAS", bid_pina - 1, int(trades_wanna_do_pina / 2))
            self.append_as_ones("PINA_COLADAS", bid_pina - 2,
                                trades_wanna_do_pina - int(trades_wanna_do_pina / 4) - int(trades_wanna_do_pina / 4))

    def position_management_etf(self, ask_etf, bid_etf, ask_dip, bid_dip, ask_bag, bid_bag, ask_uku, bid_uku,
                                multi):  # TODO: Look if bid/ask average is better indikator than Markov with Indikator best-second
        etf_factor = 1.3
        # TODO: Update Average
        # multi = float(self.new_assets["PINA_COLADAS"]["ask"][self.filled_till-1]) / float(self.new_assets["COCONUTS"]["ask"][self.filled_till-1])
        if self.mini_etf > multi:
            self.mini_etf = multi
        elif self.maxi_etf < multi:
            self.maxi_etf = multi
        percentile = self.find_multiplier_percentile_etf(multi)
        if percentile > 0:
            percentile = min(1, percentile * etf_factor)
        else:
            percentile = max(-1, percentile * etf_factor)
        wanna_be_position_etf = -int(self.limits["PICNIC_BASKET"] * percentile)
        wanna_be_position_uku = int(self.average_etf * self.limits["PICNIC_BASKET"] * percentile)
        wanna_be_position_bag = 2 * int(self.average_etf * self.limits["PICNIC_BASKET"] * percentile)
        wanna_be_position_dip = 4 * int(self.average_etf * self.limits["PICNIC_BASKET"] * percentile)
        # print(wanna_be_position_etf, wanna_be_position_uku, wanna_be_position_bag, wanna_be_position_dip)

        trades_wanna_do_etf = wanna_be_position_etf - self.state.position["PICNIC_BASKET"]
        trades_wanna_do_uku = wanna_be_position_uku - self.state.position["UKULELE"]
        trades_wanna_do_bag = wanna_be_position_bag - self.state.position["BAGUETTE"]
        trades_wanna_do_dip = wanna_be_position_dip - self.state.position["DIP"]

        # print("PERCENTILE COCO PINA", percentile, trades_wanna_do_coco, trades_wanna_do_pina)
        print("WANNA POSITION, PRICE", "etf:", wanna_be_position_etf, bid_etf, "uku: ", wanna_be_position_uku, bid_uku, "bag: ", wanna_be_position_bag, bid_bag, "dip: ", wanna_be_position_dip, bid_dip)
        
        correction = 1/16

        spread_dip_moment = self.new_assets["DIP"]["bid"][self.filled_till-1] - self.new_assets["DIP"]["ask"][self.filled_till-1]
        spread_bag_moment = self.new_assets["BAGUETTE"]["bid"][self.filled_till-1] -self.new_assets["BAGUETTE"]["ask"][self.filled_till-1]
        spread_uku_moment = self.new_assets["UKULELE"]["bid"][self.filled_till-1] -self.new_assets["UKULELE"]["ask"][self.filled_till-1]
        spread_etf_moment = self.new_assets["PICNIC_BASKET"]["bid"][self.filled_till-1] - self.new_assets["PICNIC_BASKET"]["ask"][self.filled_till-1]

        if trades_wanna_do_etf < 0:
            #ask_etf = self.price_management_pina(trades_wanna_do_etf, bid_etf, ask_etf)
            #ask_etf -= int(correction * trades_wanna_do_etf * spread_etf_moment)
            self.append_as_ones("PICNIC_BASKET", ask_etf, int(trades_wanna_do_etf))
        elif trades_wanna_do_etf > 0:
            #bid_etf = self.price_management_pina(trades_wanna_do_etf, bid_etf, ask_etf)
            #bid_etf -= int(correction * trades_wanna_do_etf * spread_etf_moment)
            self.append_as_ones("PICNIC_BASKET", bid_etf, int(trades_wanna_do_etf))

        if trades_wanna_do_uku < 0:
            #ask_uku = self.price_management_pina(trades_wanna_do_uku, bid_uku, ask_uku)
            #ask_uku -= int(correction * trades_wanna_do_uku * spread_uku_moment)
            self.append_as_ones("UKULELE", ask_uku, int(trades_wanna_do_uku))
        elif trades_wanna_do_uku > 0:
            #bid_uku = self.price_management_pina(trades_wanna_do_uku, bid_uku, ask_uku)
            #bid_uku -= int(correction * trades_wanna_do_uku * spread_uku_moment)
            self.append_as_ones("UKULELE", bid_uku, int(trades_wanna_do_uku))
        
        if trades_wanna_do_bag < 0:
            #ask_bag = self.price_management_pina(trades_wanna_do_bag, bid_bag, ask_bag)
            #ask_bag -= int(0.5 * correction * trades_wanna_do_bag * spread_bag_moment)
            self.append_as_ones("BAGUETTE", ask_bag, int(trades_wanna_do_bag))
        elif trades_wanna_do_bag > 0:
            #bid_bag = self.price_management_pina(trades_wanna_do_bag, bid_bag, ask_bag)
            #bid_bag -= int(0.5 * correction * trades_wanna_do_bag * spread_bag_moment)
            self.append_as_ones("BAGUETTE", bid_bag, int(trades_wanna_do_bag))
        
        if trades_wanna_do_dip < 0:
            #ask_dip = self.price_management_pina(trades_wanna_do_dip, bid_dip, ask_dip)
            #ask_dip -= int(0.25 * correction * trades_wanna_do_dip * spread_dip_moment)
            self.append_as_ones("DIP", ask_dip, int(trades_wanna_do_dip))
        elif trades_wanna_do_dip > 0:
            #bid_dip = self.price_management_pina(trades_wanna_do_dip, bid_dip, ask_dip)
            #bid_dip -= int(0.25 * correction * trades_wanna_do_dip * spread_dip_moment)
            self.append_as_ones("DIP", bid_dip, int(trades_wanna_do_dip))


    def copying_min_max(self, ticker: str, min_sell: int, max_buy: int):
        long = int((self.limits[ticker] - self.state.position[ticker]) / 4)
        short = -int(abs(-self.limits[ticker] - self.state.position[ticker]) / 4)

        self.append_as_ones(ticker, max_buy, long)
        self.append_as_ones(ticker, min_sell, short)
        second_long = int((self.limits[ticker] - self.state.position[ticker] - long - 1) / 2)
        second_short = -int((abs(-self.limits[ticker] - self.state.position[ticker] + short) + 1) / 2)
        self.append_as_ones(ticker, max_buy - 1, second_long)
        self.append_as_ones(ticker, min_sell + 1, second_short)
        self.append_as_ones(ticker, max_buy - 2,
                            max(0, (self.limits[ticker] - self.state.position[ticker] - long - second_long)))
        self.append_as_ones(ticker, min_sell + 2,
                            min(0, -self.limits[ticker] - self.state.position[ticker] - short - second_short))

    def second_largest(self, liste: list):
        if len(liste) == 1:
            return np.NAN
        mx = max(liste[0], liste[1])
        secondmax = min(liste[0], liste[1])
        n = len(liste)
        for i in range(2, n):
            if liste[i] > mx:
                secondmax = mx
                mx = liste[i]
            elif secondmax < liste[i] != mx:
                secondmax = liste[i]
            elif mx == secondmax and \
                    secondmax != liste[i]:
                secondmax = liste[i]
        return secondmax

    def second_smallest(self, numbers):
        if len(numbers) == 1:
            return np.NAN
        m1 = m2 = float('inf')
        for x in numbers:
            if x <= m1:
                m1, m2 = x, m1
            elif x < m2:
                m2 = x
        return m2

    def extreme_points(self, ask, bid, dol_diff):
        if abs(dol_diff) > 7:
            self.extreme_iter = self.filled_till - 1
            self.extreme_diff = dol_diff
        if self.filled_till - 230 < self.extreme_iter < self.filled_till:
            if self.extreme_diff > 0:
                wanna_trade = self.limits["DIVING_GEAR"] - self.state.position["DIVING_GEAR"]
                self.append_as_ones("DIVING_GEAR", ask, wanna_trade)
            if self.extreme_diff < 0:
                wanna_trade = -self.limits["DIVING_GEAR"] - self.state.position["DIVING_GEAR"]
                self.append_as_ones("DIVING_GEAR", bid, wanna_trade)
        else:
            wanna_trade = - self.state.position["DIVING_GEAR"]
            if wanna_trade < 0:
                self.append_as_ones("DIVING_GEAR", bid, wanna_trade)
            if wanna_trade > 0:
                self.append_as_ones("DIVING_GEAR", ask, wanna_trade)

    def etf_pricing(self, ):
        pass

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        for symbol in ["BANANAS", "PEARLS", "COCONUTS", "PINA_COLADAS", "BERRIES", "DIVING_GEAR", "BAGUETTE", "DIP",
                       "UKULELE", "PICNIC_BASKET"]:
            liste = []
            try:
                for i in range(len(state.market_trades[symbol])):
                    liste.append([state.market_trades[symbol][i].price, state.market_trades[symbol][i].quantity])
            except KeyError:
                pass
            # print(symbol, ": ", liste)
        self.state = state
        self.result = {}
        self.buy_list = {}
        self.sell_list = {}
        self.max_buy = {}
        self.second_max_buy = {}
        self.min_sell = {}
        self.second_min_sell = {}
        for i in ["PEARLS", "BANANAS", "COCONUTS", "PINA_COLADAS", "BERRIES", "DIVING_GEAR", "BAGUETTE", "DIP",
                  "UKULELE", "PICNIC_BASKET"]:
            if not (i in self.state.position.keys()):
                self.state.position[i] = 0
            self.result[i] = []
            self.buy_list[i] = list(self.state.order_depths[i].buy_orders.keys())
            self.sell_list[i] = list(self.state.order_depths[i].sell_orders.keys())
            self.max_buy[i] = max(self.buy_list[i])
            self.second_max_buy[i] = self.second_largest(self.buy_list[i])
            self.min_sell[i] = min(self.sell_list[i])
            self.second_min_sell[i] = self.second_smallest(self.sell_list[i])
            self.new_assets[i]["bid"][self.filled_till] = self.max_buy[i]
            self.new_assets[i]["ask"][self.filled_till] = self.min_sell[i]
            self.new_assets[i]["bid_second"][self.filled_till] = self.second_max_buy[i]
            self.new_assets[i]["ask_second"][self.filled_till] = self.second_min_sell[i]
        print(self.state.position)

        buy_pearls_list = list(self.state.order_depths["PEARLS"].buy_orders.keys())
        sell_pearls_list = list(self.state.order_depths["PEARLS"].sell_orders.keys())
        buy_bananas_list = list(self.state.order_depths["BANANAS"].buy_orders.keys())
        sell_bananas_list = list(self.state.order_depths["BANANAS"].sell_orders.keys())
        buy_coco_list = list(self.state.order_depths["COCONUTS"].buy_orders.keys())
        sell_coco_list = list(self.state.order_depths["COCONUTS"].sell_orders.keys())
        buy_pina_list = list(self.state.order_depths["PINA_COLADAS"].buy_orders.keys())
        sell_pina_list = list(self.state.order_depths["PINA_COLADAS"].sell_orders.keys())
        buy_berries_list = list(self.state.order_depths["BERRIES"].buy_orders.keys())
        sell_berries_list = list(self.state.order_depths["BERRIES"].sell_orders.keys())
        buy_dive_list = list(self.state.order_depths["DIVING_GEAR"].buy_orders.keys())
        sell_dive_list = list(self.state.order_depths["DIVING_GEAR"].sell_orders.keys())

        max_buy_pearls = max(buy_pearls_list)
        min_sell_pearls = min(sell_pearls_list)
        max_buy_bananas = max(buy_bananas_list)
        min_sell_bananas = min(sell_bananas_list)
        max_buy_coco = max(buy_coco_list)
        min_sell_coco = min(sell_coco_list)
        max_buy_pina = max(buy_pina_list)
        min_sell_pina = min(sell_pina_list)
        max_buy_berries = max(buy_berries_list)
        min_sell_berries = min(sell_berries_list)
        max_buy_dive = max(buy_dive_list)
        min_sell_dive = min(sell_dive_list)

        second_buy_pearls = self.second_largest(buy_pearls_list)
        second_sell_pearls = self.second_smallest(sell_pearls_list)  # TODO: CHANGE TO SECOND LOWEST
        second_buy_bananas = self.second_largest(buy_bananas_list)
        second_sell_bananas = self.second_smallest(sell_bananas_list)
        second_buy_coco = self.second_largest(buy_coco_list)
        second_sell_coco = self.second_smallest(sell_coco_list)
        second_buy_pina = self.second_largest(buy_pina_list)
        second_sell_pina = self.second_smallest(sell_pina_list)
        second_buy_berries = self.second_largest(buy_berries_list)
        second_sell_berries = self.second_smallest(sell_berries_list)
        second_buy_dive = self.second_largest(buy_dive_list)
        second_sell_dive = self.second_smallest(sell_dive_list)

        self.bid_history_bananas[self.filled_till] = max_buy_bananas
        self.ask_history_bananas[self.filled_till] = min_sell_bananas
        self.ask_history_pearls[self.filled_till] = min_sell_pearls
        self.bid_history_pearls[self.filled_till] = max_buy_pearls
        self.new_assets["COCONUTS"]["bid"][self.filled_till] = max_buy_coco
        self.new_assets["COCONUTS"]["ask"][self.filled_till] = min_sell_coco
        self.new_assets["PINA_COLADAS"]["ask"][self.filled_till] = min_sell_pina
        self.new_assets["PINA_COLADAS"]["bid"][self.filled_till] = max_buy_pina
        self.new_assets["BERRIES"]["ask"][self.filled_till] = min_sell_berries
        self.new_assets["BERRIES"]["bid"][self.filled_till] = max_buy_berries
        self.new_assets["DIVING_GEAR"]["ask"][self.filled_till] = min_sell_dive
        self.new_assets["DIVING_GEAR"]["bid"][self.filled_till] = max_buy_dive

        self.bid_history_bananas_second[self.filled_till] = second_buy_bananas
        self.ask_history_bananas_second[self.filled_till] = second_sell_bananas
        self.bid_history_pearls_second[self.filled_till] = second_buy_pearls
        self.ask_history_pearls_second[self.filled_till] = second_sell_pearls
        self.new_assets["COCONUTS"]["bid_second"][self.filled_till] = second_buy_coco
        self.new_assets["COCONUTS"]["ask_second"][self.filled_till] = second_sell_coco
        self.new_assets["PINA_COLADAS"]["ask_second"][self.filled_till] = second_sell_pina
        self.new_assets["PINA_COLADAS"]["bid_second"][self.filled_till] = second_buy_pina
        self.new_assets["BERRIES"]["ask_second"][self.filled_till] = second_sell_berries
        self.new_assets["BERRIES"]["bid_second"][self.filled_till] = second_buy_berries
        self.new_assets["DIVING_GEAR"]["ask_second"][self.filled_till] = second_sell_dive
        self.new_assets["DIVING_GEAR"]["bid_second"][self.filled_till] = second_buy_dive
        self.ratio_history_precise[self.filled_till] = float(min_sell_pina) / float(min_sell_coco)
        self.ratio_history_floored[self.filled_till] = int(
            np.floor(10000 * float(min_sell_pina) / float(min_sell_coco)))
        self.ratio_history_floored_rough[self.filled_till] = int(
            np.floor(1000 * float(min_sell_pina) / float(min_sell_coco)))

        self.dol_history[self.filled_till] = self.state.observations["DOLPHIN_SIGHTINGS"]
        # print("FLOOR: ", self.ratio_history_floored[self.filled_till])
        self.filled_till += 1
        # print("BERRIES MARKET:", min_sell_berries, max_buy_berries)
        # print(self.state.observations)
        if self.filled_till > 99:
            if self.not_yet_init:
                spread_bananas = self.bid_history_bananas[1:self.filled_till - 1] - self.ask_history_bananas[
                                                                                    1:self.filled_till - 1]
                spread_pearls = self.bid_history_pearls[1:self.filled_till - 1] - self.ask_history_pearls[
                                                                                  1:self.filled_till - 1]
                spread_berries = self.new_assets["BERRIES"]["bid"][1:self.filled_till - 1] - self.new_assets["BERRIES"][
                                                                                                 "ask"][
                                                                                             1:self.filled_till - 1]
                self.markov_ask_bananas = Markov(
                    [self.ask_history_bananas[1:self.filled_till - 1] - self.ask_history_bananas[:self.filled_till - 2],
                     spread_bananas, self.ask_history_bananas[1:self.filled_till - 1] - self.ask_history_bananas_second[
                                                                                        1:self.filled_till - 1]],
                    self.ask_history_bananas[2:self.filled_till] - self.ask_history_bananas[1:self.filled_till - 1])
                self.markov_ask_pearls = Markov([self.ask_history_pearls[1:self.filled_till - 1],
                                                 self.ask_history_pearls[
                                                 1:self.filled_till - 1] - self.ask_history_pearls[
                                                                           :self.filled_till - 2], spread_pearls,
                                                 self.ask_history_pearls[
                                                 1:self.filled_till - 1] - self.ask_history_pearls_second[
                                                                           1:self.filled_till - 1]],
                                                self.ask_history_pearls[2:self.filled_till] - self.ask_history_pearls[
                                                                                              1:self.filled_till - 1])
                self.markov_bid_bananas = Markov(
                    [self.bid_history_bananas[1:self.filled_till - 1] - self.bid_history_bananas[:self.filled_till - 2],
                     spread_bananas, self.bid_history_bananas[1:self.filled_till - 1] - self.bid_history_bananas_second[
                                                                                        1:self.filled_till - 1]],
                    self.bid_history_bananas[2:self.filled_till] - self.bid_history_bananas[1:self.filled_till - 1])
                self.markov_bid_pearls = Markov([self.bid_history_pearls[1:self.filled_till - 1],
                                                 self.bid_history_pearls[
                                                 1:self.filled_till - 1] - self.bid_history_pearls[
                                                                           :self.filled_till - 2], spread_pearls,
                                                 self.bid_history_pearls[
                                                 1:self.filled_till - 1] - self.bid_history_pearls_second[
                                                                           1:self.filled_till - 1]],
                                                self.bid_history_pearls[2:self.filled_till] - self.bid_history_pearls[
                                                                                              1:self.filled_till - 1])
                self.markov_ask_coco = Markov([self.new_assets["COCONUTS"]["ask"][:self.filled_till - 1] -
                                               self.new_assets["COCONUTS"]["ask_second"][:self.filled_till - 1]],
                                              self.new_assets["COCONUTS"]["ask"][1:self.filled_till] -
                                              self.new_assets["COCONUTS"]["ask"][:self.filled_till - 1])
                self.markov_bid_coco = Markov([self.new_assets["COCONUTS"]["bid"][:self.filled_till - 1] -
                                               self.new_assets["COCONUTS"]["bid_second"][:self.filled_till - 1]],
                                              self.new_assets["COCONUTS"]["bid"][1:self.filled_till] -
                                              self.new_assets["COCONUTS"]["bid"][:self.filled_till - 1])
                self.markov_ask_pina = Markov([self.new_assets["PINA_COLADAS"]["ask"][:self.filled_till - 1] -
                                               self.new_assets["PINA_COLADAS"]["ask_second"][:self.filled_till - 1]],
                                              self.new_assets["PINA_COLADAS"]["ask"][1:self.filled_till] -
                                              self.new_assets["PINA_COLADAS"]["ask"][:self.filled_till - 1])
                self.markov_bid_pina = Markov([self.new_assets["PINA_COLADAS"]["bid"][:self.filled_till - 1] -
                                               self.new_assets["PINA_COLADAS"]["bid_second"][:self.filled_till - 1]],
                                              self.new_assets["PINA_COLADAS"]["bid"][1:self.filled_till] -
                                              self.new_assets["PINA_COLADAS"]["bid"][:self.filled_till - 1])
                self.markov_ratio = Markov([self.ratio_history_floored_rough[1:self.filled_till - 1],
                                            self.ratio_history_floored[
                                            1:self.filled_till - 1] - self.ratio_history_floored[
                                                                      :self.filled_till - 2]],
                                           self.ratio_history_floored[1:self.filled_till - 1])
                self.markov_ask_berries = Markov([self.new_assets["BERRIES"]["ask"][1:self.filled_till - 1] -
                                                  self.new_assets["BERRIES"]["ask"][:self.filled_till - 2],
                                                  spread_berries,
                                                  self.new_assets["BERRIES"]["ask"][1:self.filled_till - 1] -
                                                  self.new_assets["BERRIES"]["ask_second"][1:self.filled_till - 1]],
                                                 self.new_assets["BERRIES"]["ask"][2:self.filled_till] -
                                                 self.new_assets["BERRIES"]["ask"][1:self.filled_till - 1])
                self.markov_bid_berries = Markov([self.new_assets["BERRIES"]["bid"][1:self.filled_till - 1] -
                                                  self.new_assets["BERRIES"]["bid"][:self.filled_till - 2],
                                                  spread_berries,
                                                  self.new_assets["BERRIES"]["bid"][1:self.filled_till - 1] -
                                                  self.new_assets["BERRIES"]["bid_second"][1:self.filled_till - 1]],
                                                 self.new_assets["BERRIES"]["bid"][2:self.filled_till] -
                                                 self.new_assets["BERRIES"]["bid"][1:self.filled_till - 1])

                self.not_yet_init = False
                # print("INITIALIZED")
            else:
                spread_berries_update = self.new_assets["BERRIES"]["bid"][self.filled_till - 2] - \
                                        self.new_assets["BERRIES"]["ask"][self.filled_till - 2]
                spread_bananas_update = self.bid_history_bananas[self.filled_till - 2] - self.ask_history_bananas[
                    self.filled_till - 2]
                spread_pearls_update = self.bid_history_pearls[self.filled_till - 2] - self.ask_history_pearls[
                    self.filled_till - 2]
                self.markov_ask_bananas.update(
                    [self.ask_history_bananas[self.filled_till - 2] - self.ask_history_bananas[self.filled_till - 3],
                     spread_bananas_update,
                     self.ask_history_bananas[self.filled_till - 2] - self.ask_history_bananas_second[
                         self.filled_till - 2]],
                    self.ask_history_bananas[self.filled_till - 1] - self.ask_history_bananas[self.filled_till - 2])
                self.markov_ask_pearls.update([self.ask_history_pearls[self.filled_till - 2],
                                               self.ask_history_pearls[self.filled_till - 2] - self.ask_history_pearls[
                                                   self.filled_till - 3], spread_pearls_update,
                                               self.ask_history_pearls[self.filled_till - 2] -
                                               self.ask_history_pearls_second[self.filled_till - 2]],
                                              self.ask_history_pearls[self.filled_till - 1] - self.ask_history_pearls[
                                                  self.filled_till - 2])
                self.markov_bid_bananas.update(
                    [self.bid_history_bananas[self.filled_till - 2] - self.bid_history_bananas[self.filled_till - 3],
                     spread_bananas_update,
                     self.bid_history_bananas[self.filled_till - 2] - self.bid_history_bananas_second[
                         self.filled_till - 2]],
                    self.bid_history_bananas[self.filled_till - 1] - self.bid_history_bananas[self.filled_till - 2])
                self.markov_bid_pearls.update([self.bid_history_pearls[self.filled_till - 2],
                                               self.bid_history_pearls[self.filled_till - 2] - self.bid_history_pearls[
                                                   self.filled_till - 3], spread_pearls_update,
                                               self.bid_history_pearls[self.filled_till - 2] -
                                               self.bid_history_pearls_second[self.filled_till - 2]],
                                              self.bid_history_pearls[self.filled_till - 1] - self.bid_history_pearls[
                                                  self.filled_till - 2])
                self.markov_ask_coco.update([self.new_assets["COCONUTS"]["ask"][self.filled_till - 2] -
                                             self.new_assets["COCONUTS"]["ask_second"][self.filled_till - 2]],
                                            self.new_assets["COCONUTS"]["ask"][self.filled_till - 1] -
                                            self.new_assets["COCONUTS"]["ask"][self.filled_till - 2])
                self.markov_bid_coco.update([self.new_assets["COCONUTS"]["bid"][self.filled_till - 2] -
                                             self.new_assets["COCONUTS"]["bid_second"][self.filled_till - 2]],
                                            self.new_assets["COCONUTS"]["bid"][self.filled_till - 1] -
                                            self.new_assets["COCONUTS"]["bid"][self.filled_till - 2])
                self.markov_ask_pina.update([self.new_assets["PINA_COLADAS"]["ask"][self.filled_till - 2] -
                                             self.new_assets["PINA_COLADAS"]["ask_second"][self.filled_till - 2]],
                                            self.new_assets["PINA_COLADAS"]["ask"][self.filled_till - 1] -
                                            self.new_assets["PINA_COLADAS"]["ask"][self.filled_till - 2])
                self.markov_bid_pina.update([self.new_assets["PINA_COLADAS"]["bid"][self.filled_till - 2] -
                                             self.new_assets["PINA_COLADAS"]["bid_second"][self.filled_till - 2]],
                                            self.new_assets["PINA_COLADAS"]["bid"][self.filled_till - 1] -
                                            self.new_assets["PINA_COLADAS"]["bid"][self.filled_till - 2])
                self.markov_ratio.update([self.ratio_history_floored_rough[self.filled_till - 2],
                                          self.ratio_history_floored[self.filled_till - 2] - self.ratio_history_floored[
                                              self.filled_till - 3]], self.ratio_history_floored[self.filled_till - 1])
                self.markov_bid_berries.update([self.new_assets["BERRIES"]["bid"][self.filled_till - 2] -
                                                self.new_assets["BERRIES"]["bid"][self.filled_till - 3],
                                                spread_berries_update,
                                                self.new_assets["BERRIES"]["bid"][self.filled_till - 2] -
                                                self.new_assets["BERRIES"]["bid_second"][self.filled_till - 2]],
                                               self.new_assets["BERRIES"]["bid"][self.filled_till - 1] -
                                               self.new_assets["BERRIES"]["bid"][self.filled_till - 2])
                self.markov_ask_berries.update([self.new_assets["BERRIES"]["ask"][self.filled_till - 2] -
                                                self.new_assets["BERRIES"]["ask"][self.filled_till - 3],
                                                spread_berries_update,
                                                self.new_assets["BERRIES"]["ask"][self.filled_till - 2] -
                                                self.new_assets["BERRIES"]["ask_second"][self.filled_till - 2]],
                                               self.new_assets["BERRIES"]["ask"][self.filled_till - 1] -
                                               self.new_assets["BERRIES"]["ask"][self.filled_till - 2])

            spread_bananas_moment = self.bid_history_bananas[self.filled_till - 1] - self.ask_history_bananas[
                self.filled_till - 1]
            spread_pearls_moment = self.bid_history_pearls[self.filled_till - 1] - self.ask_history_pearls[
                self.filled_till - 1]
            spread_berries_moment = self.new_assets["BERRIES"]["bid"][self.filled_till - 1] - \
                                    self.new_assets["BERRIES"]["ask"][self.filled_till - 1]

            multi_pina = self.markov_ratio.average([self.ratio_history_floored_rough[self.filled_till - 1],
                                                    self.ratio_history_floored[self.filled_till - 1] -
                                                    self.ratio_history_floored[self.filled_till - 2]]) / 10000
            # print("PREDICTED FLOOR:", multi_pina)

            bid_berries_price = 1 + int(self.new_assets["BERRIES"]["bid"][self.filled_till - 1]) + int(
                self.markov_bid_berries.average([self.new_assets["BERRIES"]["bid"][self.filled_till - 1] -
                                                 self.new_assets["BERRIES"]["bid"][self.filled_till - 2],
                                                 spread_berries_moment,
                                                 self.new_assets["BERRIES"]["bid"][self.filled_till - 1] -
                                                 self.new_assets["BERRIES"]["bid_second"][self.filled_till - 1]]))

            ask_berries_price = 1 + int(self.new_assets["BERRIES"]["ask"][self.filled_till - 1]) + int(
                self.markov_bid_berries.average([self.new_assets["BERRIES"]["ask"][self.filled_till - 1] -
                                                 self.new_assets["BERRIES"]["ask"][self.filled_till - 2],
                                                 spread_berries_moment,
                                                 self.new_assets["BERRIES"]["ask"][self.filled_till - 1] -
                                                 self.new_assets["BERRIES"]["ask_second"][self.filled_till - 1]]))

            bid_bananas_price = 1 + int(max_buy_bananas) + int(self.markov_bid_bananas.average(
                [self.bid_history_bananas[self.filled_till - 1] - self.bid_history_bananas[self.filled_till - 2],
                 spread_bananas_moment,
                 self.bid_history_bananas[self.filled_till - 1] - self.bid_history_bananas_second[
                     self.filled_till - 1]]))

            ask_bananas_price = -1 + int(min_sell_bananas) + int(self.markov_ask_bananas.average(
                [self.ask_history_bananas[self.filled_till - 1] - self.ask_history_bananas[self.filled_till - 2],
                 spread_bananas_moment,
                 self.ask_history_bananas[self.filled_till - 1] - self.ask_history_bananas_second[
                     self.filled_till - 1]]))

            bid_pearls_price = 1 + int(max_buy_pearls) + int(self.markov_bid_pearls.average(
                [self.bid_history_pearls[self.filled_till - 1],
                 self.bid_history_pearls[self.filled_till - 1] - self.bid_history_pearls[self.filled_till - 2],
                 spread_pearls_moment, self.bid_history_pearls[self.filled_till - 1] - self.bid_history_bananas_second[
                     self.filled_till - 1]]))

            ask_pearls_price = - 1 + int(min_sell_pearls) + int(self.markov_ask_pearls.average(
                [self.ask_history_pearls[self.filled_till - 1],
                 self.ask_history_pearls[self.filled_till - 1] - self.ask_history_pearls[self.filled_till - 2],
                 spread_pearls_moment,
                 self.ask_history_pearls[self.filled_till - 1] - self.ask_history_pearls_second[self.filled_till - 1]]))

            bid_coco_price = int(max_buy_coco) + int(self.markov_bid_coco.average([self.new_assets["COCONUTS"]["bid"][
                                                                                       self.filled_till - 1] -
                                                                                   self.new_assets["COCONUTS"][
                                                                                       "bid_second"][
                                                                                       self.filled_till - 1]]))

            ask_coco_price = int(min_sell_coco) + int(self.markov_ask_coco.average([self.new_assets["COCONUTS"]["ask"][
                                                                                        self.filled_till - 1] -
                                                                                    self.new_assets["COCONUTS"][
                                                                                        "ask_second"][
                                                                                        self.filled_till - 1]]))

            bid_pina_price = int(max_buy_pina) + int(self.markov_bid_pina.average([self.new_assets["PINA_COLADAS"][
                                                                                       "bid"][self.filled_till - 1] -
                                                                                   self.new_assets["PINA_COLADAS"][
                                                                                       "bid_second"][
                                                                                       self.filled_till - 1]]))

            ask_pina_price = int(min_sell_pina) + int(self.markov_ask_pina.average([self.new_assets["PINA_COLADAS"][
                                                                                        "ask"][self.filled_till - 1] -
                                                                                    self.new_assets["PINA_COLADAS"][
                                                                                        "ask_second"][
                                                                                        self.filled_till - 1]]))

            # print(max_buy_bananas, min_sell_bananas, max_buy_pearls, min_sell_pearls)
            # print("BID BANANAS: ", bid_bananas_price, "ASK BANANAS: ", ask_bananas_price, "BID PEARLS: ", bid_pearls_price, "ASK PEARLS: ", ask_pearls_price)
            # ask_bananas_price, bid_bananas_price = self.minimum_diff(max_buy_bananas, bid_bananas_price, min_sell_bananas, ask_bananas_price)
            # TODO: move that outside this function
            self.position_management_pina_coco(ask_coco_price - 1, bid_coco_price + 1, ask_pina_price - 1,
                                               bid_pina_price + 1, multi=multi_pina)
            #  "BAGUETTE", "DIP", "UKULELE", "PICNIC_BASKET"

            add_bananas = int(self.state.position["BANANAS"] / 10)
            ask_bananas_price, bid_bananas_price = self.assert_ask_bid_diff(ask_bananas_price - add_bananas,
                                                                            bid_bananas_price - add_bananas)
            self.copying_min_max("BANANAS", ask_bananas_price, bid_bananas_price)

            # self.copying_min_max("BERRIES", ask_berries_price-1, bid_berries_price+1)
            self.berries_manangement(ask_berries_price - 1, bid_berries_price + 1)

            # ask_pearls_price, bid_pearls_price = self.minimum_diff(max_buy_pearls, bid_pearls_price, min_sell_pearls, ask_pearls_price)
            add_pearls = int(self.state.position["PEARLS"] / 10)
            ask_pearls_price, bid_pearls_price = self.assert_ask_bid_diff(ask_pearls_price - add_pearls,
                                                                          bid_pearls_price - add_pearls)
            self.copying_min_max("PEARLS", ask_pearls_price, bid_pearls_price)
            # self.extreme_points(min_sell_dive, max_buy_dive, self.dol_history[self.filled_till-1] - self.dol_history[self.filled_till-2])
            # add_coco = int(self.state.position["COCONUTS"] / 10)
            # ask_coco_price, bid_coco_price = self.assert_ask_bid_diff(ask_coco_price - add_coco, bid_coco_price - add_coco)
            # self.copying_min_max("COCONUTS", ask_coco_price, bid_coco_price)

        else:
            multi_pina = float(self.new_assets["PINA_COLADAS"]["ask"][self.filled_till - 1]) / float(
                self.new_assets["COCONUTS"]["ask"][self.filled_till - 1])

            self.copying_min_max("PEARLS", min_sell=int(min_sell_pearls), max_buy=int(max_buy_pearls))
            self.copying_min_max("BANANAS", min_sell=int(min_sell_bananas), max_buy=int(max_buy_bananas))

            # self.copying_min_max("BERRIES", min_sell_berries, max_buy_berries+1)
            self.berries_manangement(min_sell_berries - 1, max_buy_berries + 1)

            self.position_management_pina_coco(min_sell_coco - 1, max_buy_coco + 1, min_sell_pina - 1, max_buy_pina + 1,
                                               multi=multi_pina)
            # self.extreme_points(min_sell_dive, max_buy_dive,
            #                    self.dol_history[self.filled_till - 1] - self.dol_history[self.filled_till - 2])

        # print("RESULT: ", self.result)
        # TODO: TRADE REJECTED
        spread_dip_moment = self.new_assets["DIP"]["bid"][self.filled_till-1] - self.new_assets["DIP"]["ask"][self.filled_till-1]
        spread_bag_moment = self.new_assets["BAGUETTE"]["bid"][self.filled_till-1] -self.new_assets["BAGUETTE"]["ask"][self.filled_till-1]
        spread_uku_moment = self.new_assets["UKULELE"]["bid"][self.filled_till-1] -self.new_assets["UKULELE"]["ask"][self.filled_till-1]
        spread_etf_moment = self.new_assets["PICNIC_BASKET"]["bid"][self.filled_till-1] - self.new_assets["PICNIC_BASKET"]["ask"][self.filled_till-1]

        aggression_factor_etf = 0.5
        self.position_management_etf(self.min_sell["PICNIC_BASKET"] + np.ceil(spread_etf_moment*aggression_factor_etf), self.max_buy["PICNIC_BASKET"] - np.ceil(spread_etf_moment*aggression_factor_etf), self.min_sell["DIP"] + np.ceil(spread_dip_moment*aggression_factor_etf), self.max_buy["DIP"] - np.ceil(spread_dip_moment*aggression_factor_etf), self.min_sell["BAGUETTE"] + np.ceil(spread_bag_moment*aggression_factor_etf), self.max_buy["BAGUETTE"] - np.ceil(spread_bag_moment*aggression_factor_etf), self.min_sell["UKULELE"] + np.ceil(spread_uku_moment*aggression_factor_etf), self.max_buy["UKULELE"] - np.ceil(spread_uku_moment*aggression_factor_etf), multi=self.min_sell["PICNIC_BASKET"]/(self.min_sell["UKULELE"] + 2*self.min_sell["BAGUETTE"] + 4*self.min_sell["DIP"]))
        #self.position_management_etf(self.min_sell["PICNIC_BASKET"], self.max_buy["PICNIC_BASKET"], self.min_sell["DIP"], self.max_buy["DIP"], self.min_sell["BAGUETTE"], self.max_buy["BAGUETTE"], self.min_sell["UKULELE"], self.max_buy["UKULELE"], multi=self.min_sell["PICNIC_BASKET"]/(self.min_sell["UKULELE"] + 2*self.min_sell["BAGUETTE"] + 4*self.min_sell["DIP"]))
        if self.filled_till > 2:
            #    print(self.dol_history[self.filled_till - 1] - self.dol_history[self.filled_till - 2])
            self.extreme_points(min_sell_dive, max_buy_dive,
                                self.dol_history[self.filled_till - 1] - self.dol_history[self.filled_till - 2])
        print("RESULT\n", self.result["DIP"])
        return self.result

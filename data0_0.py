from jqdatasdk import *
auth('15918636606', 'Ucb16697')
import numpy as np
import datetime
import gc
import math
import pandas as pd
import time
from dateutil.relativedelta import relativedelta

class data:
    # 获取每天的行情数据,以 前x周的数据 判断一周后的价格变动（上涨还是下跌）
    # (休市的日期是跳过的）
    def __init__(self, index=['000001.XSHG', '000002.XSHG', '000016.XSHG',
                              '000903.XSHG', '000904.XSHG', '000905.XSHG',
                              '000906.XSHG', '000907.XSHG', '399001.XSHE', '399004.XSHE',
                              '399005.XSHE', '399006.XSHE', '399009.XSHE',
                              '399010.XSHE', '399011.XSHE', '399310.XSHE', '399310.XSHE',
                              '399310.XSHE', '000852.XSHG', '000300.XSHG'],
                 start_date='2012-05-31', end_date='2018-01-01', frequency='daily'):
        self.start_date = start_date
        self.end_date = end_date
        # 由聚宽平台获取指数历史行情
        # 返回行索引为日期，列索引为数据字段的dataframe

        self.index_data1 = get_price(security=index[0], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data2 = get_price(security=index[1], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data3 = get_price(security=index[2], start_date=start_date, end_date=end_date, frequency=frequency)

        self.index_data4 = get_price(security=index[3], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data5 = get_price(security=index[4], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data6 = get_price(security=index[5], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data7 = get_price(security=index[6], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data8 = get_price(security=index[7], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data9 = get_price(security=index[8], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data10 = get_price(security=index[9], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data11 = get_price(security=index[10], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data12 = get_price(security=index[11], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data13 = get_price(security=index[12], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data14 = get_price(security=index[13], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data15 = get_price(security=index[14], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data16 = get_price(security=index[15], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data17 = get_price(security=index[16], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data18 = get_price(security=index[17], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data19 = get_price(security=index[18], start_date=start_date, end_date=end_date, frequency=frequency)
        self.index_data20 = get_price(security=index[19], start_date=start_date, end_date=end_date, frequency=frequency)

        # 储存每个特征的平均值和标准差
        self.mean = []
        self.std = []




        self.data = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}, 10: {}, 11: {}, 12:{}, 13:{},
                     14:{}, 15:{}, 16:{}, 17:{}, 18:{}, 19:{}}
        self.label = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}, 10: {}, 11: {}, 12:{}, 13:{},
                     14:{}, 15:{}, 16:{}, 17:{}, 18:{}, 19:{}}

        # 训练输入
        self.x_train = []
        # 训练标签
        self.y_train = []
        # 测试输入
        self.x_test = []
        # 测试标签
        self.y_test = []

    # 一些技术指标
    def ema(self, p, n=26):
        # p为对应区间的 股票/大盘指数 收盘价的列表
        if n == len(p):
            alpha = 2 / (n + 1)
            emavalue = p[0]
            for i in range(1, n):
                emavalue = alpha * (p[i] - emavalue) + emavalue
                # EMAtoday=α*(Pricetoday-EMAyesterday)+EMAyesterday
            return emavalue
        else:
            if n > len(p):
                n = len(p)
                alpha = 2 / (n + 1)
                emavalue = p[0]
                for i in range(1, n):
                    emavalue = alpha * (p[i] - emavalue) + emavalue
                    # EMAtoday=α*(Pricetoday-EMAyesterday)+EMAyesterday
                return emavalue

    def macd(self, p, n1=6, n2=12, n3=9, n=26):
        # n1为EMA1的参数(常取12），n2为EMA2的参数（常取26），n3为DIF的参数（常取9），n为区间长度，p为对应时间区间内股票收盘价的列表
        # 返回值依次为dif，dea，macd
        dif = []
        try:
            for i in range(n - n3, n):
                ema1 = self.ema(p[i - n1 + 1:i + 1], n1)
                ema2 = self.ema(p[i - n2 + 1:i + 1], n2)
                # R表示对于时间区间的每日价格列表
                dif.append(ema1 - ema2)
            return [dif[-1], self.ema(dif, n3), dif[-1] - self.ema(dif, n3)]
        except:
            return [None, None, None]

    def rsi(self, p, n=26):
        # n为周期(默认为26天），p为对应时间区间内的每日收盘价格列表，需利用接口
        try:
            rise = 0
            drop = 0
            for i in range(1, n):
                if p[i] >= p[i - 1]:
                    rise = rise + (p[i] - p[i - 1])
                else:
                    drop = drop + (p[i - 1] - p[i])
            # rise为n日内收盘涨幅之和，drop为n日内收盘跌幅之和
            if (rise + drop) != 0:
                return (100 * rise) / (rise + drop)
            else:
                return 50
        except:
            return None

    def llt(self, p, n=26):  # （聚宽课堂上面的啥二阶滤波器，不知道啥玩意，随便拿来用一下）, p为对应时间的收盘价列表
        list = []
        try:
            alpha = 2 / 35
            list.append(self.ema([p[0]], 1))
            list.append(self.ema(p[0:2], 2))
            for i in range(2, n):
                list.append((alpha - alpha * alpha / 4) * p[i] + (alpha * alpha / 2) * p[i - 1] - (
                        alpha - 3 * alpha * alpha / 4) * p[i - 2]
                            + 2 * (1 - alpha) * list[i - 1] - (1 - alpha) * (1 - alpha) * list[i - 2])
            return list[-1]
        except:
            return None

    def kdj(self, cn, hn, ln, n=26):
        # n为周期，cn为第n日的收盘价，hn和ln分别为此n日的最高价和最低价，需利用接口
        try:
            if hn != ln:
                rsv = 100 * (cn - ln) / (hn - ln)
            else:
                rsv = 100
            k = d = j = 0
            for i in range(1, n):
                k = (2 / 3) * k + (1 / 3) * rsv
                d = (2 / 3) * d + (1 / 3) * k
                j = 3 * d - 2 * k
            return [k, d, j]
        except:
            return [None, None, None]

    def obv(self, v, c):
        # n为周期，v为对于时间区间内的成交量列表,c为对应的收盘价列表
        n = len(v)
        obvvalue = 0
        for i in range(1, n):
            if c[i] > c[i-1]:
                obvvalue = obvvalue + v[i]
            if c[i] < c[i-1]:
                obvvalue = obvvalue - v[i]
        return obvvalue

    def boll(self, c, n=26):
        try:# n为周期，c为对应的时间区间内的收盘价列表
            sp1 = 0
            for i in range(0, n):
                sp1 = sp1 + c[i]  # n日收盘价之和
            ma = sp1 / n  # n日内的收盘价之和÷n
            sp2 = 0
            for i in range(1, n):
                sp2 = sp2 + c[i]
            mb = sp2 / (n - 1)  # 价格线，（n－1）日的ma
            sp3 = 0
            for i in range(0, n):
                sp3 = sp3 + (c[i] - ma) * (c[i] - ma)
            md = math.sqrt(sp3 / n)  # 平方根n日的（c[i]－MA）的两次方之和除以n
            up = mb + 2 * md  # 上轨线UP是UP数值的连线，用黄色线表示
            dn = mb - 2 * md  # 下轨线DN是DN数值的连线，用紫色线表示
            return [up, dn, mb]
        except:
            return [None, None, None]

    def trix(self, p, n1=9, n2=12):
        try:
            # n1为ema参数,n2为trix参数，p是对应时间区间内的收盘价列表
            tr = []
            sp = 0
            for i in range(0, n2):
                p1 = p[i: n1 + i]
                tr.append(self.ema(p1, n1))
            for i in range(0, n2 - 1):
                sp = sp + ((tr[i+1] - tr[i]) / tr[i+1] * 100)
            return sp/n2
        except:
            return None

    def ar(self, c, h, l):
        try:
            # 对应时间区间内的每日收盘，最高，最低价列表
            arvalue1 = arvalue2 = 0
            for i in range(0, len(c)):
                arvalue1 = arvalue1 + (h[i] - c[i])
                arvalue2 = arvalue2 + (c[i] - l[i])
            if arvalue2 != 0:
                return arvalue1/arvalue2
            elif arvalue1 != 0:
                return arvalue1/0.001
            else:
                return 1
        except:
            return None

    def br(self, c, h, l):
        try:
            # 对应时间区间内的每日收盘，最高，最低价列表
            brvalue1 = brvalue2 = 0
            for i in range(1, len(c)):
                brvalue1 = brvalue1 + (h[i] - c[i-1])
                brvalue2 = brvalue2 + (c[i-1] - l[i])
            if brvalue2 != 0:
                brvalue = brvalue1/brvalue2
            elif brvalue1 != 0:
                brvalue = brvalue1/0.00001
            else:
                brvalue = 1
            return brvalue
        except:
            return None

    def cmo(self, c, n=26):  # n为周期，c为对应区间内的收盘价列表
        try:
            su = sd = 0
            for i in range(1, n):
                if c[i] >= c[i - 1]:
                    su = su + c[i] - c[i - 1]
                else:
                    sd = sd + c[i - 1] - c[i]
            if (su + sd) != 0:
                return 100 * (su - sd) / (su + sd)
            else:
                return 0
        except:
            return None

    def cr(self, o, c, h, l):
        # n为周期,o,c,h,l分别对应对应区间内的开盘，收盘，最高，最低价
        n = len(c)
        p1 = p2 = 0
        for i in range(1, n):
            m = (c[i-1] + o[i-1] + h[i-1] + l[i-1])/4
            p1 = p1 + h[i] - m
            p2 = p2 + m - l[i]
        if p2 != 0:
            return 100 * (p1/p2)
        elif p1 != 0:
            return 100 * (p1/0.001)
        else:
            return 100

    def dma(self, p, n1=6, n2=12):
        try:
            # n1为短期平均参数，n2为长期平均参数，n为区间长度，p为对应区间内的股票收盘价列表
            n = len(p)
            long = short = 0
            for i in range(n - n2, n):
                long = long + p[i]
            for i in range(n - n1, n):
                short = short + p[i]
            long = long/n2
            short = short/n1
            return short - long
        except:
            return None



    # 整理数据
    def get_data(self):
        # 最外层data是一个字典，其中key为日期，value为每日前十五日的每日行情数据序列（特征）
        # 添加技术因子


        for n in range(20):
        #for index_data in [self.index_data1, self.index_data2, self.index_data3]:
            index_data = [self.index_data1, self.index_data2, self.index_data3, self.index_data4,
                          self.index_data5, self.index_data6, self.index_data7, self.index_data8,
                          self.index_data9, self.index_data10, self.index_data11, self.index_data12,
                          self.index_data13, self.index_data14,
                          self.index_data15, self.index_data16, self.index_data17, self.index_data18,
                          self.index_data19, self.index_data20][n]
            macd1 = []
            macd2 = []
            macd3 = []
            rsi = []
            llt = []
            k = []
            d = []
            j = []
            ema = []
            obv = []
            up = []
            dn = []
            mb = []
            trix = []
            ar = []
            br = []
            cmo = []
            cr = []
            dma = []

            for i in range(index_data.shape[0]):
                p = (index_data['close'].values.tolist())[max(0, i - 25):i + 1]
                open = (index_data['open'].values.tolist())[max(0, i - 25):i + 1]
                low = (index_data['low'].values.tolist())[max(0, i - 25):i + 1]
                high = (index_data['high'].values.tolist())[max(0, i - 25):i + 1]
                lowest = np.min(low)
                highest = np.max(high)
                volume = (index_data['volume'].values.tolist())[max(0, i - 25):i + 1]

                macd1.append(self.macd(p)[0])
                macd2.append(self.macd(p)[1])
                macd3.append(self.macd(p)[2])
                rsi.append(self.rsi(p))
                llt.append(self.llt(p))
                k.append(self.kdj(p[-1], highest, lowest)[0])
                d.append(self.kdj(p[-1], highest, lowest)[1])
                j.append(self.kdj(p[-1], highest, lowest)[2])
                ema.append(self.ema(p))
                obv.append(self.obv(volume, p))
                up.append(self.boll(p)[0])
                dn.append(self.boll(p)[1])
                mb.append(self.boll(p)[2])
                trix.append(self.trix(p))
                ar.append(self.ar(p, high, low))
                br.append(self.br(p, high, low))
                cmo.append(self.cmo(p))
                cr.append(self.cr(open, p, high, low))
                dma.append(self.dma(p))


            #TODO：取对数？
            #self.index_data = self.index_data.apply(lambda x: np.log(x))

            #TODO: 加入技术指标？
            index_data['macd1'] = macd1
            index_data['macd2'] = macd2
            index_data['macd3'] = macd3
            index_data['rsi'] = rsi
            index_data['llt'] = llt
            index_data['k'] = k
            index_data['d'] = d
            index_data['j'] = j
            index_data['ema'] = ema
            index_data['obv'] = obv
            index_data['up'] = up
            index_data['dn'] = dn
            index_data['mb'] = mb
            index_data['trix'] = trix
            index_data['ar'] = ar
            index_data['br'] = br
            index_data['cmo'] = cmo
            index_data['cr'] = cr
            index_data['dma'] = dma

            index_data = index_data.dropna(axis=0, how='any')

            if n == 19:
                index_data.to_csv('./data.csv')


            self.mean = (index_data.apply(np.mean)).tolist()
            self.std = (index_data.apply(np.std)).tolist()

            # 数据预处理（标准化）
            post_index_data = index_data.apply(lambda x: (x - np.mean(x)) / np.std(x))
            # self.od = self.index_data.values


            for i in range(30, post_index_data.shape[0] - 7):   # 隔5个交易日（一周）取一次样

                date_time = datetime.datetime.strptime((post_index_data.index[i])._date_repr, '%Y-%m-%d')
                seq = []
                # 前5*j天的每一天数据都加入序列中
                for d in range(5 * 3):
                    seq.append(post_index_data.iloc[i - 5 * 3 + d])
                self.data[n][date_time] = seq
                # label是一个字典，其中key为日期，value为该日对应的标签

                close_list = []
                for d in range(5):
                    close_list.append(index_data.iloc[i + 1 + d]['close'])

                #f = self.fluct(close_list)

                if index_data.iloc[i+1]['llt'] > 1.0001*index_data.iloc[i]['llt']:
                    self.label[n][date_time] = 1

                if index_data.iloc[i+1]['llt'] < 0.92*index_data.iloc[i]['llt']:
                    self.label[n][date_time] = 0





    #数据划分，分出训练集和测试集（特征及标签）
    def divide_data(self):

        for n in range(20):
            day = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
            while day < datetime.datetime.strptime(self.end_date, '%Y-%m-%d'):
                if day in self.label[n].keys():
                    # 6月,9月及12月的数据划入测试集
                    if day.month >= 13:
                        self.x_test.append(self.data[n][day])
                        self.y_test.append(self.label[n][day])

                    # 其他月份作为训练集
                    else:
                        self.x_train.append(self.data[n][day])
                        self.y_train.append(self.label[n][day])

                day = day + datetime.timedelta(days=1)


# 训练过程中的采样函数,j为batch中的序列长度
    def get_mini_batch(self, batch_size):
        batch_x = []
        batch_y = []
        index_list = []
        i = 0
        while i < batch_size:
            # 产生随机数作索引以随机取样
            index = np.random.randint(low=0, high=len(self.x_train))

            # 避免重复
            if index not in index_list:
                batch_x.append(self.x_train[index])
                batch_y.append(self.y_train[index])
                i = i + 1
                index_list.append(index)

            # batch_x.append(self.x_train[j][index])
            # batch_y.append(self.y_train[j][index])
            #i = i + 1
        del i
        del index_list
        gc.collect()
        return batch_x, batch_y


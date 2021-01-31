# -*- coding: utf-8 -*-
##二手车价格预测
# lianghong.liu & mengrui.liu
# 2021.1.8
# copyright USTC

##实现分析
# 预测二手车价格：读取爬虫抓取的数据->数据清洗->数据预处理->主成分分析->定义模型->训练模型->评估模型

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential  # 顺序模型
from keras.layers import Dense,Dropout
import keras.backend as K
from sklearn.metrics import r2_score
from keras.utils import plot_model
from keras.models import load_model     #
import matplotlib.pyplot as plt     # 绘图工具

from sklearn.decomposition import PCA   # 主成分分析
import seaborn as sns   # 画图工具
from sklearn.preprocessing import LabelEncoder     # 标准化标签

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


# 读取数据文件
def get_data():
    """
    读取数据
    :return:原始数据
    """
    car_df = pd.read_csv("car3.csv")
    # car_df = pd.read_excel('car2_data.xlsx')
    print(car_df.head())
    car_df = pd.DataFrame(car_df)
    return car_df
    # 查看数据大小
    # print(car_df.shape)
    # 显示数据前5行
    # print(car_df.head())


def clean_data(car_df):
    """
    数据清洗：缺失值处理、数值换算等
    :param car_df:读取的数据
    :return:
    """
    # 1、缺失值、异常值处理
    # print(car_df.isnull().sum()) # 检查所有列是否含有缺失值
    car_df = car_df.drop('行李箱容积', axis=1)
    car_df = car_df.dropna()  # 删除所有含有缺失值的行
    car_df = car_df[~car_df['上牌时间'].isin(['未知年份'])]  # 删除含有'未知年份'中文的行
    # print(car_df.shape)
    # print(car_df.isnull().sum())
    # 2、数值转换
    # （1）把排量单位为T转换为L
    deal_data = car_df['排量'].str.contains('T').fillna(False)
    # 将 T 转换为 L，1T = 1.4L
    for i, lbs_row in car_df[deal_data].iterrows():
        # iterrows() 是在数据框中的行进行迭代的一个生成器，它返回每行的索引及一个包含行本身的对象
        weight = float(lbs_row['排量'][:-1]) * 1.4
        car_df.at[i, '排量'] = '{}L'.format(weight)
        # df.at 的作用：获取某个位置的值，例如：获取第0行、第a列的值，即：index = 0，columns = 'a'， data = df.at[0, 'a']
    # （2）把年份重新重新赋值为：2021-年份
    car_df['time'] = '2021'
    car_df['上牌时间'] = car_df['time'].astype(int) - car_df['上牌时间'].astype(int)


    # 3、去除异常字符
    car_df['原价'] = car_df['原价'].str.strip('万').astype(float)
    car_df['售价'] = car_df['售价'].str.strip('万').astype(float)
    car_df['排量'] = car_df['排量'].str.strip('L').astype(float)
    car_df = car_df[~car_df['轴距'].isin(['-'])]
    car_df = car_df[~car_df['整备质量'].isin(['-'])]
    car_df = car_df[~car_df['进气形式'].isin(['-'])]
    car_df['气缸数'] = car_df['气缸数'].str.strip('缸').astype(float)
    car_df['表显里程'] = car_df['表显里程'].str.strip('万公里').astype(float)

    car_df = car_df.drop(car_df[car_df["原价"] > 150].index)
    car_df = car_df.drop(car_df[car_df["上牌时间"] < -100].index)
    car_df = car_df.drop(car_df[(car_df["排量"] > 4.5) & (car_df["售价"] < 40)].index)

    # 4、对离散型特征进行编码
    label_mapping = {"自动": 1, "手动": 0}
    car_df['变速箱'] = car_df.变速箱.map(label_mapping)
    label_mapping = {"国三": 1, "国四": 2, "国五": 3, "国六": 4}
    car_df['排放标准'] = car_df.排放标准.map(label_mapping)
    label_mapping = {"0次过户": 1, "1次过户": 2, "2次过户": 3, "3次过户": 4, "4次过户": 5, "5次过户": 6, "6次过户": 7, "7次过户": 8, "9次过户": 8}
    car_df['过户次数'] = car_df.过户次数.map(label_mapping)
    l = car_df['车长/宽/高'].str.split('/').tolist()
    car_df[['length', 'width', 'height']] = l
    cols = ( "产权性质", "进气形式", "燃料类型", "燃油标号", "供油方式", "前悬挂类型", "助力类型", "驱动方式",
            "后悬挂类型", "前制动类型", "后制动类型", "驱车制动类型", "abs", "esp", "定速巡航", "gps", "倒车雷达", "倒车影像系统",
            "驱车制动类型", "主/副驾驶安全气囊", "前/后排侧气囊", "胎压检测", "车内中控锁", "儿童座椅接口", "无钥匙启动", "电动天窗", "全景天窗",
            "电动吸合门", "感应后备箱", "感应雨刷", "后雨刷", "前/后电动车窗", "后视镜电动调节", "后视镜加热", "多功能方向盘", "定速巡航", "后排独立空调",
            "空调控制方式", "gps", "倒车雷达", "倒车影像系统", "真皮座椅", "前/后排座椅加热", "前/后排头部气囊")
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(car_df[c].values))
        car_df[c] = lbl.transform(list(car_df[c].values))

    # 5、 删除无用数据
    car_df = car_df.drop(['城市','名称','品牌','前轮胎规格','后轮胎规格','time','车长/宽/高',"使用性质", "看车方式"] ,axis=1)
    car_df.dropna(inplace=True)



    return car_df


def pre_processing_data(car_df):
    """
    数据预处理：数据集划分
    :param car_df:
    """
    #  1、主成分分析（注意：如果是选择mle自动分析，则需要返回最终降维之后的特征个数）

    #  2、划分数据集
    y = np.array(car_df['售价'])  # 目标值
    x = np.array(car_df.drop(['售价'], axis=1))  # 特征值
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    #  3、特征数据标准化
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)

    # 查看数据集
    print('训练集x：', x_train.shape)
    print('测试集x：', x_test.shape)
    print('训练集y：', y_train.shape)
    print('测试集y：', y_test.shape)

    return x_train, x_test, y_train, y_test


def pca(x_train, x_test):
    """
    降维
    :param x_train:训练集
    :param x_test:测试集
    :return:
    """
    # pca2 = PCA(n_components=55, svd_solver="full")
    # train_x_pca2 = pca2.fit_transform(x_train)
    pca = PCA(n_components=0.95 , svd_solver="full")
    train_x_pca = pca.fit_transform(x_train)
    test_x_pca = pca.transform(x_test)
    n_input = len(pca.explained_variance_ratio_)
    print(n_input)


    train_x_pd=car_df.drop('售价', axis=1)
    colnames = train_x_pd.columns
    rate_abs = abs(pca.components_[0])
    # rate=pca2.explained_variance_ratio_
    # 对特征和贡献率绝对值进行数组关联排序
    Z = zip(rate_abs, colnames)
    Z = sorted(Z, reverse=True)
    rate_abs, colnames = zip(*Z)
    # 求累加贡献率
    # rate_sum = np.array(rate)
    # rate2 = rate_sum.cumsum()

    print("n_input", n_input)
    return train_x_pca, test_x_pca, n_input


def def_model(inputCell_num, denseCell_num, inputNum):
    """
    定义神经网络模型
    :param inputCell_num:输入层神经元个数
    :param denseCell_num: 隐藏层神经元个数
    :param inputNum: 输入层特征数
    :return: 模型
    """
    model = Sequential()  # 先建立一个顺序模型
    # 向顺序模型里加入第一个隐藏层，第一层一定要有一个输入数据的大小，需要有input_shape参数
    model.add(Dense(inputCell_num, activation='relu', input_dim=inputNum))  # 这个input_dim和input_shape一样，就是少了括号和逗号
    model.add(Dropout(0.02))
    model.add(Dense(denseCell_num, activation='relu'))
    model.add(Dropout(0.02))
    model.add(Dense(1))  # 因为我们是预测二手车价格，不是分类，所以最后一层可以不用激活函数
    return model


def r2(y_true, y_pred):
    """
    自定义
    :param y_true:
    :param y_pred:
    :return:
    """
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b / e
    return f


def run_model(model, x_train, y_train, training_epochs, batch_size):
    """
    训练模型
    :param model:定义好的模型
    :param x_train: 训练集x
    :param y_train: 训练集y
    :param training_epochs:训练迭代次数
    :param batch_size: 每批次要取的数据量
    :return: 训练好的模型
    """
    model.compile(loss='mse', optimizer='adam', metrics=['mae', r2])
    history = model.fit(x_train, y_train,verbose=0, batch_size=batch_size, epochs=training_epochs)
    # history = model.fit(x_train, y_train,
    #                     epochs=training_epochs, batch_size=batch_size,
    #                     shuffle=True,  validation_split=0.4,
    #                     validation_data=(x_test, y_test))
    return model


def save_model(model):
    """
    保存模型
    :param model:训练好的模型
    """
    # 保存模型
    model.save('model_MLP.h5')  # 生成模型文件 'my_model.h5'
    # 模型可视化 需要安装pydot pip install pydot
    plot_model(model, to_file='model_MLP.png', show_shapes=True)


def evaluate_model(model, x_test, y_test):
    """
    评估模型
    :param model:训练好的模型
    :param x_test: 测试集x
    :param y_test: 测试集y
    :return: 预测准确率，使用模型预测的数据集
    """
    pred_test_y = model.predict(x_test)  # 预测的结果
    pred_acc = r2_score(y_test, pred_test_y)  # 准确率
    print('模型准确率：', pred_acc)
    return pred_acc, pred_test_y


def running_to_best(x_train, y_train, x_test, y_test, input_num):
    best_dense1Cell_num = 3
    best_dense2Cell_num = 3
    best_training_epochs = 2
    best_batch_size = 10
    best_acc = 0
    sign = 0
    for dense1Cell_num in range(31, 34):
        for dense2Cell_num in range(31, 34):
            model = def_model(dense1Cell_num, dense2Cell_num, input_num)
            for training_epochs in range(200, 201):
                for batch_size in range(20, 32):
                    train_model = run_model(model, x_train, y_train, training_epochs, batch_size)
                    pred_acc, pred_test_y = evaluate_model(train_model, x_test, y_test)
                    if (pred_acc > best_acc):
                        best_dense1Cell_num = dense1Cell_num
                        best_dense2Cell_num = dense2Cell_num
                        best_training_epochs = training_epochs
                        best_batch_size = batch_size
                        best_acc = pred_acc
                        print('当前最大准确率为：', best_acc)
                        print('当前最佳参数为：dense1Cell_num：', dense1Cell_num,
                              'dense2Cell_num:', dense2Cell_num, 'training_epochs:', training_epochs,
                              'batch_size:', batch_size)
                    if (pred_acc > 0.99):
                        sign = 1
                        best_dense1Cell_num = dense1Cell_num
                        best_dense2Cell_num = dense2Cell_num
                        best_training_epochs = training_epochs
                        best_batch_size = batch_size
                        print('最佳参数为：dense1Cell_num：', dense1Cell_num,
                              'dense2Cell_num:', dense2Cell_num, 'training_epochs:', training_epochs,
                              'batch_size:', batch_size)
                        break
                if (sign == 1): break
            if (sign == 1): break
        if (sign == 1): break
    return best_dense1Cell_num, best_dense2Cell_num, best_training_epochs, best_batch_size


def draw_result(y_test, predict_y_test):
    """
    预测结果可视化
    :param y_test: 测试集y
    :param predict_y_test:模型的预测集
    """
    # 设置字体为中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 设置图形大小
    _, ax = plt.subplots(figsize=(24, 20))
    plt.plot(range(len(y_test)), y_test, ls='-.', lw=2, c='r', label='真实值')
    plt.plot(range(len(predict_y_test)), predict_y_test, ls='-', lw=2, c='b', label='预测值')

    # 绘制网格
    plt.title('预测结果对比', fontdict={'weight': 'normal', 'size': 80})  # 改变图标题字体
    plt.xlabel('编号', fontdict={'weight': 'normal', 'size': 60})  # 改变坐标轴标题字体
    plt.ylabel('二手车价格', fontdict={'weight': 'normal', 'size': 60})  # 改变坐标轴标题字体
    plt.tick_params(labelsize = 50)

    plt.grid(alpha=0.4, linestyle=':')
    plt.legend(loc="upper right",prop={'size':40})
    plt.savefig('./result.jpg') #保存图片
    # 展示
    plt.show()


if __name__ == '__main__':
    # 读取数据
    car_df = get_data()
    # 数据清洗
    car_df = clean_data(car_df)
    # 数据预处理
    x_train, x_test, y_train, y_test = pre_processing_data(car_df)
    # 主成分分析
    x_train, x_test, input_num = pca(x_train, x_test)
    # 定义模型
    model = def_model(64, 32, input_num)
    # 训练模型
    # best_dense1Cell_num, best_dense2Cell_num, best_training_epochs, best_batch_size = running_to_best(x_train, y_train, x_test, y_test, input_num)
    model = run_model(model, x_train, y_train, 200, 20)
    # 保存模型
    save_model(model)
    # 加载模型:如果有自定义的评价函数，则需要加上custom_objects={'loss_fun': loss}
    train_model = load_model('model_MLP.h5', custom_objects={'r2': r2})
    # 评估模型
    accuracy, predict_y_test = evaluate_model(train_model, x_test, y_test)

    # 预测结果可视化
    draw_result(y_test, predict_y_test)



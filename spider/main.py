# -*- coding: utf-8 -*-
##瓜子二手车爬虫
# jiawei.liu
# 2021.1.4

##实现分析
# 抓取瓜子二手车上的网站
# 页面网址: https://www.guazi.com/cs/buy/o50/#bread
# 获取到网页详情地址: https://www.guazi.com/gz/05322e9d8e472b30x.htm#fr_page=list&fr_pos=city&fr_no=0
# 获取所有的数据: 目标网址->发送请求(请求方式,请求头,乱码)->解析网页(正则表达式,xpath,BeautifulSoup)->保存数据(txt,csv,excel)
import time
import requests as rq  # 网络请求库
from bs4 import BeautifulSoup as bs
import csv
import re

debug = False
csv_menu = []
sleep_time = 5

# 请求头信息
# UA:身份认证
# Cookie:Cookie有过期时间,过期之后就无法获取数据了,需要更换Cookie

# chrome浏览器
headers1 = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
    'Cookie': 'antipas=42753262W9207405H3363169981'
}

# edge浏览器
headers2 = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36 Edg/87.0.664.66',
    'Cookie': 'antipas=2N7593105499097086O91792XO'
}

headers = [headers1, headers2]


# 请求城市页面获取车辆url
def getCarUrl(city_url, car_brif):
    count = 0
    count_all = 0
    base_url = 'https://www.guazi.com'
    for city in city_url:
        try:
            myheaders = headers[count_all % len(headers)]
            print('正在获取：' + city)
            pages = 51
            if debug:
                pages = 2
            for index in range(1, pages):
                try:
                    print('正在获取' + city + '第：' + str(index) + ' 页')

                    # 1.url
                    url = 'https://www.guazi.com/' + city + '/buy/o' + str(index) + '/#bread'

                    # 2.模拟浏览器发送请求,接受响应
                    # response = rq.get(url) #无请求头返回203
                    respones = rq.get(url=url, headers=myheaders, timeout=3)
                    # print(respones.status_code) #输出返回码
                    # print(response.text) #以文本形式输出网页源代码,该方式会输出乱码
                    # print(response.content.decode('utf-8')) #以二进制形式输出网页源代码,根据utf-8解码

                    # 3.网页解析
                    if respones.status_code == 200:
                        text = respones.content.decode('utf-8')
                        # print(text)
                        raw_html = bs(text, 'lxml')  # 返回网页
                        # <ul class="carlist clearfix js-top">
                        car_ul = raw_html.select_one('.carlist')  # ul
                        # print(car_ul)

                        # 获取car_ul中的a标签
                        car = car_ul.select('a')

                        for c in car:
                            # 原价
                            try:
                                original_price = c.select('.line-through')[0].get_text()
                                car_brif[base_url + c.get('href')] = original_price
                                # 每个城市获取条目计数
                                count += 1
                            except:
                                continue
                    else:
                        print('respones.status_code' + str(respones.status_code))
                        # time.sleep(sleep_time)  #降低ip频率,防反爬休眠
                except:
                    continue

            count_all += count
            print(city + '共获取：' + str(count) + '条')
            count = 0
        except:
            continue

    print('\n所有城市共获取：' + str(count_all) + '条')


# 请求车辆页面获取数据
def getCarDetail(car_brif):
    count = 0
    count10 = 0
    car_info_list = []
    print('\n获取信息ing...')

    for url in car_brif:
        # 防止网站反爬拒绝访问导致整个程序崩溃
        try:
            myheaders = headers[count % len(headers)]
            respones = rq.get(url=url, headers=myheaders, timeout=5)
            if respones.status_code == 200:
                text = respones.content.decode('utf-8')
                # print(text)
                raw_html = bs(text, 'lxml')  # 返回car页面
                # print(raw_html)

                # 城市
                city = raw_html.select_one('title').text[0:2].strip()
                # 名称
                title = raw_html.select_one('.titlebox').text.strip()
                searchObj1 = re.search(r'([\u4e00-\u9fa5]+.).+', title, re.I)
                if searchObj1:
                    real_title = searchObj1.group().replace('\r', '')
                else:
                    real_title = 'null'

                # 品牌
                brand = real_title.split(' ', 1)[0].strip()
                # 上牌时间
                searchObj2 = re.search(r'\d\d\d\d', real_title, re.I)
                if searchObj2:
                    year = searchObj2.group()
                else:
                    year = '未知年份'

                # 车辆信息
                info = raw_html.select('ul.assort span')
                # 表显里程
                length = info[1].text.strip()
                # 排量
                power = info[2].text.strip()
                # 变速箱
                gearbox = info[3].text.strip()

                # 原价
                original_price = car_brif[url];
                # 售价
                price = raw_html.select('div.price-main span')[0].get_text().strip()

                basic_eleven_info = raw_html.select_one('.basic-eleven')
                # 排放标准
                emission_standard = basic_eleven_info.select_one('.four').get_text().split("\n")[1].strip()
                # 过户次数
                transfer_times = basic_eleven_info.select_one('.seven').get_text().split("\n")[1].strip()
                # 使用性质
                use_type = basic_eleven_info.select_one('.nine').get_text().strip()
                # 产权性质
                right_type = basic_eleven_info.select_one('.ten').get_text().strip()
                # 看车方式
                watching_mode = basic_eleven_info.select_one('.eight').get_text().strip()
                # 异常点
                abnormal = raw_html.select('.fc-org-text')
                abnormal_num = 0
                for i in range(1, len(abnormal)):
                    abnormal_num += int(raw_html.select('.fc-org-text')[i].text[0:-3])

                # 详细配置
                detail_content = raw_html.select_one('.detailcontent').select('table')
                # 车长/宽/高
                carlwh = detail_content[0].select('td')[11].text.strip()
                # 轴距
                wheelbase = detail_content[0].select('td')[13].text.strip()
                # 行李箱容积
                cargo_volume = detail_content[0].select('td')[15].text.strip()
                # 整备质量
                curb_weight = detail_content[0].select('td')[17].text.strip()
                # 进气形式
                air_form = detail_content[1].select('td')[3].text.strip()
                # 气缸数
                cylinder = detail_content[1].select('td')[5].text.strip()
                # 最大马力
                max_horsepower = detail_content[1].select('td')[7].text.strip()
                # 最大扭矩
                max_torque = detail_content[1].select('td')[9].text.strip()
                # 燃料类型
                fuel_type = detail_content[1].select('td')[11].text.strip()
                # 燃油标号
                oil_type = detail_content[1].select('td')[13].text.strip()
                # 供油方式
                oil_supply_form = detail_content[1].select('td')[15].text.strip()
                # 驱动方式
                drive_type = detail_content[2].select('td')[1].text.strip()
                # 助力类型
                assistance_type = detail_content[2].select('td')[3].text.strip()
                # 前悬挂类型
                front_suspension_type = detail_content[2].select('td')[5].text.strip()
                # 后悬挂类型
                back_suspension_type = detail_content[2].select('td')[7].text.strip()
                # 前制动类型
                front_brake_type = detail_content[2].select('td')[9].text.strip()
                # 后制动类型
                back_brake_type = detail_content[2].select('td')[11].text.strip()
                # 驱车制动类型
                driving_brake_type = detail_content[2].select('td')[13].text.strip()
                # 前轮胎规格
                front_tire_size = detail_content[2].select('td')[15].text.strip()
                # 后轮胎规格
                back_tire_size = detail_content[2].select('td')[17].text.strip()
                # 主/副驾驶安全气囊
                main_cc_driver_airbag = detail_content[3].select('td')[15].text.strip()
                # 前/后排侧气囊
                front_rear_side_airbags = detail_content[3].select('td')[3].text.strip()
                # 前/后排头部气囊
                front_rear_head_airbags = detail_content[3].select('td')[5].text.strip()
                # 胎压检测
                tire_pressure_detection = detail_content[3].select('td')[7].text.strip()
                # 车内中控锁
                interior_central_locking = detail_content[3].select('td')[9].text.strip()
                # 儿童座椅接口
                child_seat_interface = detail_content[3].select('td')[11].text.strip()
                # 无钥匙启动
                keyless_start = detail_content[3].select('td')[13].text.strip()
                # abs
                abs = detail_content[3].select('td')[15].text.strip()
                # esp
                esp = detail_content[3].select('td')[17].text.strip()

                # 电动天窗
                electric_skylight = detail_content[4].select('td')[1].text.strip()
                # 全景天窗
                panoramic_sunroof = detail_content[4].select('td')[3].text.strip()
                # 电动吸合门
                electric_pull_in_door = detail_content[4].select('td')[5].text.strip()
                # 感应后备箱
                induction_trunk = detail_content[4].select('td')[7].text.strip()
                # 感应雨刷
                induction_wiper = detail_content[4].select('td')[9].text.strip()
                # 后雨刷
                rear_wiper = detail_content[4].select('td')[11].text.strip()
                # 前/后电动车窗
                front_rear_power_windows = detail_content[4].select('td')[13].text.strip()
                # 后视镜电动调节
                rearview_mirror_electric_adjustment = detail_content[4].select('td')[15].text.strip()
                # 后视镜加热
                rearview_mirror_heating = detail_content[4].select('td')[17].text.strip()

                # 多功能方向盘
                multi_function_steering_wheel = detail_content[5].select('td')[1].text.strip()
                # 定速巡航
                cruise_control = detail_content[5].select('td')[3].text.strip()
                # 后排独立空调
                rear_independent_air_conditioner = detail_content[5].select('td')[5].text.strip()
                # 空调控制方式
                air_conditioning_control_mode = detail_content[5].select('td')[7].text.strip()
                # gps
                gps = detail_content[5].select('td')[9].text.strip()
                # 倒车雷达
                pdc = detail_content[5].select('td')[11].text.strip()
                # 倒车影像系统
                rcpa = detail_content[5].select('td')[13].text.strip()
                # 真皮座椅
                leather_seat = detail_content[5].select('td')[15].text.strip()
                # 前/后排座椅加热
                front_rear_seat_heating = detail_content[5].select('td')[17].text.strip()

                car = {
                    '城市':
                        city,
                    '名称':
                        real_title,
                    '上牌时间':
                        year,
                    '品牌':
                        brand,
                    '表显里程':
                        length,
                    '排量':
                        power,
                    '变速箱':
                        gearbox,
                    '原价':
                        original_price,
                    '售价':
                        price,
                    '排放标准':
                        emission_standard,
                    '过户次数':
                        transfer_times,
                    '使用性质':
                        use_type,
                    '产权性质':
                        right_type,
                    '看车方式':
                        watching_mode,
                    '异常点':
                        abnormal_num,
                    '车长/宽/高':
                        carlwh,
                    '轴距':
                        wheelbase,
                    '行李箱容积':
                        cargo_volume,
                    '整备质量':
                        curb_weight,
                    '进气形式':
                        air_form,
                    '气缸数':
                        cylinder,
                    '最大马力':
                        max_horsepower,
                    '最大扭矩':
                        max_torque,
                    '燃料类型':
                        fuel_type,
                    '燃油标号':
                        oil_type,
                    '供油方式':
                        oil_supply_form,
                    '驱动方式':
                        drive_type,
                    '助力类型':
                        assistance_type,
                    '前悬挂类型':
                        front_suspension_type,
                    '后悬挂类型':
                        back_suspension_type,
                    '前制动类型':
                        front_brake_type,
                    '后制动类型':
                        back_brake_type,
                    '驱车制动类型':
                        driving_brake_type,
                    '前轮胎规格':
                        front_tire_size,
                    '后轮胎规格':
                        back_tire_size,
                    '主/副驾驶安全气囊':
                        main_cc_driver_airbag,
                    '前/后排侧气囊':
                        front_rear_side_airbags,
                    '前/后排头部气囊':
                        front_rear_head_airbags,
                    '胎压检测':
                        tire_pressure_detection,
                    '车内中控锁':
                        interior_central_locking,
                    '儿童座椅接口':
                        child_seat_interface,
                    '无钥匙启动':
                        keyless_start,
                    'abs':
                        abs,
                    'esp':
                        esp,
                    '电动天窗':
                        electric_skylight,
                    '全景天窗':
                        panoramic_sunroof,
                    '电动吸合门':
                        electric_pull_in_door,
                    '感应后备箱':
                        induction_trunk,
                    '感应雨刷':
                        induction_wiper,
                    '后雨刷':
                        rear_wiper,
                    '前/后电动车窗':
                        front_rear_power_windows,
                    '后视镜电动调节':
                        rearview_mirror_electric_adjustment,
                    '后视镜加热':
                        rearview_mirror_heating,
                    '多功能方向盘':
                        multi_function_steering_wheel,
                    '定速巡航':
                        cruise_control,
                    '后排独立空调':
                        rear_independent_air_conditioner,
                    '空调控制方式':
                        air_conditioning_control_mode,
                    'gps':
                        gps,
                    '倒车雷达':
                        pdc,
                    '倒车影像系统':
                        rcpa,
                    '真皮座椅':
                        leather_seat,
                    '前/后排座椅加热':
                        front_rear_seat_heating
                }
            else:
                print('respones.status_code' + str(respones.status_code))
                # time.sleep(sleep_time)  #降低ip频率,防反爬休眠
        except:
            continue

        # 讲字典加入汽车信息(car_info_list)列表
        car_info_list.append(car)
        # 计数+1
        count += 1;

        # 只写一条数据测试
        if debug:
            break;

        # 整10输出，并清零计数，并休眠两秒（防反爬）
        if (count % 10) == 0:
            count10 += count
            print('已获取：' + str(count10) + '条')
            time.sleep(0.2)
            count = 0

    print('\n已获取：' + str(len(car_info_list)) + ' 条')

    # 生成csv_menu
    for key in car.keys():
        csv_menu.append(key)
    return car_info_list


# 储存数据
def save_csv(car_info_list):
    print('\n写入ing...')
    file = open('车辆信息.csv', 'w', newline='', encoding='utf-8')
    csvWriter = csv.writer(file)

    csvWriter.writerow(csv_menu)
    for item in car_info_list:
        # 按行写入
        csv_dat = []
        for i in range(0, len(csv_menu)):
            csv_dat.append(item.get(csv_menu[i]))
        csvWriter.writerow(csv_dat)
    file.close()


if __name__ == '__main__':
    # 每个城市的url
    # 石家庄 北京 上海 广州 深圳 成都 重庆 杭州 苏州 沈阳 武汉 天津 西安 兰州 合肥 长春 南昌 南京 南宁 西宁
    city_url = ['sjz', 'bj', 'sh', 'gz', 'sz', 'cd', 'cq', 'hz', 'su', 'sy', 'wh', 'tj', 'xa', 'lz', 'hf', 'cc',  'nc',
                'nj', 'nn', 'xn']
    if debug:
        city_url = ['sh']

    # 车辆页面的url与车的原价
    # 使用字典可以做到自动去重
    car_brif = {}

    getCarUrl(city_url, car_brif)

    # 获取到的车辆信息
    car_info_list = getCarDetail(car_brif)

    # 储存
    save_csv(car_info_list)

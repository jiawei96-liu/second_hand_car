

from flask import Flask, render_template, request
import csv
# from flask_paginate import Pagination,get_page_parameter
app = Flask(__name__)
# 指定文件名，然后使用 with open() as 打开


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/cars')
def cars():
    dataList = []
    filename = 'data.csv'
    with open(filename, 'r', encoding='utf-8') as f:
        # 创建阅读器（调用csv.reader()将前面存储的文件对象最为实参传给它）
        reader = csv.reader(f)
        # 调用了next()一次，所以这边只调用了文件的第一行，并将头文件存储在header_row中
        header_row = next(reader)
        for row in reader:
            dataList.append(row)
    return render_template("cars.html", cars = dataList)



@app.route('/city')
def city():
    return render_template("city.html")

@app.route('/year')
def year():
    return render_template("year.html")

@app.route('/carsDetail')
def carsDetal():
    return render_template("carsDetail.html")

@app.route('/model')
def model():
    return render_template("model.html")


if __name__ == '__main__':
    app.run()

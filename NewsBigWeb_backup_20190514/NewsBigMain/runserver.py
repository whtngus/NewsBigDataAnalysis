# -*- coding: utf-8 -*-

'''
Created on 2019. 5. 5.
App Name : NewsImpact  (2019 News Bigdata Contest)
@author: sunmi
'''

from flask import Flask,render_template,request

application = Flask(__name__)

@application.route("/")
def main():
    #return "Welcome!"
    return render_template('index.html')

@application.route('/result', methods=['POST','GET'])
def result() -> 'html':
    if request.method == 'POST':
#        result = request.form['searchKeyword']
         result = request.form.to_dict()
         return render_template("keyword_result.html", result=result)


@application.route("/chart")
def chart_js():
    return render_template('chart_js.html')

@application.route("/chart_line")
def chart_line_js():
    return render_template('chart_line.html')

@application.route("/chart_line_2")
def chart_line2_js():
    return render_template('chart_line_2.html')


@application.route("/chart_line_3")
def chart_line3_js():
    return render_template('chart_line_3.html')

@application.route("/chart_line_4")
def chart_line4_js():
    return render_template('chart_line_4.html')



@application.route("/chart_bar")
def chart_bar_js():
    return render_template('chart_horizontal_bar.html')

@application.route("/chart_bar_news")
def chart_bar_news_js():
    return render_template('chart_horizontal_bar_news.html')

#@application.route("/search", method=['POST'])
# def search():
#     try:
#         _keyword = request.form['searchKeyword'] 
#         # connect to mysql
#         con = mysql.connect()
#         cursor = con.cursor()
#     except Exception as e:
#         return render_template('error.html', error=str(e))

if __name__ == '__main__':
    application.run(debug=True)
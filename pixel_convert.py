# coding:utf-8
from flask import Flask, render_template, request, redirect, url_for, abort, logging
import os
import cv2
from PIL import Image
import hashlib
import datetime as dt
from pixel import make_dot

app = Flask(__name__)
config = {'MAX_CONTENT_LENGTH': 1024 * 1024 * 2, 'DEBUG': False}
app.config.update(config)


@app.route('/', methods=['GET'])
def index():
    return render_template('pixel.html')


@app.route('/', methods=['POST'])
def post():
    img = request.files['image']
    if not img:
        error = 'ファイルを選択してね'
        return render_template('pixel.html', error=error)
    k = int(request.form['k'])
    scale = int(request.form['scale'])
    blur = int(request.form['blur'])
    erode = int(request.form['erode'])
    img_name = hashlib.md5(str(dt.datetime.now()).encode('utf-8')).hexdigest()
    img_path = os.path.join('static/img', img_name + os.path.splitext(img.filename)[-1])
    result_path = os.path.join('static/results', img_name + '.png')
    img.save(img_path)
    with Image.open(img_path) as img_pl:
        if max(img_pl.size) > 1024:
            img_pl.thumbnail((1024, 1024), Image.ANTIALIAS)
            img_pl.save(img_path)
            # os.remove(img_path)
            # return render_template('pixel.html', error=error)
    img_res = make_dot(img_path, k=k, scale=scale, blur=blur, erode=erode)
    cv2.imwrite(result_path, img_res)
    return render_template('pixel.html', org_img=img_path, result=result_path)


@app.errorhandler(413)
def error_file_size(e):
    error = 'ファイルサイズが大きすぎます。アップロード可能サイズは2MBまでです。'
    return render_template('pixel.html', error=error), 413


@app.errorhandler(404)
def not_found(e):
    error = 'らめぇ'
    return render_template('pixel.html', error=error), 404


if __name__ == '__main__':
    app.debug = True
    app.run()

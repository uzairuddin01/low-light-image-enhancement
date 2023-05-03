from flask import Flask,render_template,session,redirect,url_for,request, make_response
import os
import requests
import cv2
import torch
import numpy as np
from model import NNetwork
import sqlite3
# Ignore warnings
import json
import warnings
warnings.filterwarnings("ignore")
import time
from functools import wraps, update_wrapper
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
        
    return update_wrapper(no_cache, view)

# Load Saved Model 
fully_conv_network = NNetwork().to('cpu')
fully_conv_network.load_state_dict(torch.load('model/saved_model.pt',map_location= torch.device('cpu')))

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():
	user = request.args.get('user','')
	email = request.args.get('email','')
	password = request.args.get('pass','')
	number = request.args.get('mobile','')
	name = request.args.get('name','')
	con = sqlite3.connect('signup.db')
	cur = con.cursor()
	cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(user,email,password,number,name))
	con.commit()
	con.close()
	return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('pass','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("home.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html")
    else:
        return render_template("signup.html")

@app.route('/home', methods=['POST','GET'])
@nocache
def home():
	if request.method == 'GET':
		return render_template('home.html')

	
	# Get uploaded image
	get_picture=request.files['fileToUpload']
	time_stamp = str(time.time()).replace('.','')

	get_picture.save("./static/"+time_stamp+".jpg")
	uploaded_image=cv2.imread("./static/"+time_stamp+".jpg")

	# Resize image to dimensions (512,512) and switch channels
	input_image = np.array(uploaded_image)
	height,width = input_image.shape[0],input_image.shape[1]
	input_image = cv2.resize(input_image,(512,512))
	input_image = np.moveaxis(input_image, -1, 0)

	dark_imgs=[]
	dark_imgs.append(input_image)

	# Pass Image as input to the Neural Network
	out = fully_conv_network(torch.as_tensor(dark_imgs).float())
	out = out.detach().cpu().numpy()

	# Resize image to original dimensions
	output_image = np.moveaxis(out[0], -1, 0)
	output_image = np.moveaxis(output_image, -1, 0)
	output_image = cv2.resize(output_image,(width,height))

	original_picture_path="./static/"+time_stamp+".jpg"
	enhanced_picture_path="./static/"+time_stamp+"1.jpg"

	cv2.imwrite("./static/"+time_stamp+"1.jpg",output_image)

	return render_template("result.html",data={"ori":original_picture_path,"pro":enhanced_picture_path})

@app.route('/main')
def main():
	return render_template('home.html')

@app.after_request
def add_header(r):
    
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == "__main__":
    app.run(debug=True)



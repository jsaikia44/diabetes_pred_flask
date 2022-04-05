import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/new")
def new():
    return render_template('new.html')

@app.route("/result", methods=['POST'])
def result():
    form_data = request.form
    list1=[]
    for key,value in form_data.items():
        list1.append(value)
    vect = [float(i) for i in list1]
    vect[0]=vect[0]/100
    vect= np.array([vect])
    #model declearation and fitting

    logres1 = pickle.load(open('dia_pred.pkl','rb'))
    pred4 = logres1.predict(vect)
    
    if pred4[0]==0.0:
        op ="No risk"
    else:
        op = "There is a chance"
    return render_template('result.html', result=op)

app.run(debug=True)


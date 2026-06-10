from unittest import result

from flask import Flask,render_template,request,jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

application=Flask(__name__)
app=application

## import models
diabetics_model=pickle.load(open('models/Diabetics_model.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))


@app.route("/")
def index():
         return render_template("index.html")



@app.route("/predictdata",methods=["GET","POST"])
def predict_data():
    if request.method=="POST":
        Pregnancies = int(request.form.get("Pregnancies"))
        Glucose = int(request.form.get("Glucose")) 
        BloodPressure = int(request.form.get("BloodPressure"))
        SkinThickness = int(request.form.get("SkinThickness"))
        Insulin = int(request.form.get("Insulin"))
        BMI = float(request.form.get("BMI"))
        DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
        Age = int(request.form.get("Age"))

        data=np.array([[
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
]]) 
        if not (0<=Pregnancies <=20):
             return "Invalid input for Pregnancies. Please enter a value between 0 and 20."
        if not (0<=Glucose <=200):
             return "Invalid input for Glucose. Please enter a value between 0 and 200."
        if not (0<=BloodPressure <=200):
             return "Invalid input for Blood Pressure. Please enter a value between 0 and 200."
        if not (0<=SkinThickness <=100):
             return "Invalid input for Skin Thickness. Please enter a value between 0 and 100."
        if not (0<=Insulin <=850):
             return "Invalid input for Insulin. Please enter a value between 0 and 850."
        if not (0<=BMI <=60):
             return "Invalid input for BMI. Please enter a value between 0 and 60."
        if not (0.078<=DiabetesPedigreeFunction <=2.42):
             return "Invalid input for Diabetes Pedigree Function. Please enter a value between 0.078 and 2.42."
        if not (18<=Age <=90):
             return "Invalid input for Age. Please enter a value between 18 and 90."

        scaled_data=standard_scaler.transform(data)
        prediction=diabetics_model.predict(scaled_data)

        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        return render_template("index.html", results=result)
    else:     
        # Return the prediction result
        return render_template("index.html")

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
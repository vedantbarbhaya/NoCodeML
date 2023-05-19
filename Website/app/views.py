from app import app
from flask import Flask, jsonify, request, render_template, url_for, flash, redirect
from werkzeug.utils import secure_filename
import io
import string
import time
import os
import numpy as np
from PIL import Image
import sys
import pandas as pd
from pandas_profiling import ProfileReport
import warnings
from flask import send_file

# regression files
from reg_data_validation import data_check as reg_dc
import reg_driver as reg_driver
import regression_results as regres

# classification files
from cla_data_validation import data_check as cla_dc
import cla_driver as cla_driver
from cla_predict import predict_function

warnings.filterwarnings("ignore")

# Global variables  
problemType = ""
df = ""
dep_col = ""
dc_obj = ""
dproutput = ""
dpcoutput = ""
bestmodel = ""
predict_df = ""


@app.route("/")
def upload_form():
    return render_template("fileupload.html")


@app.route("/uploads", methods=["GET", "POST"])
def upload_file():
    global problemType, df, dc_obj
    if request.method == "POST" or request.method == "GET":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected for uploading")
            return redirect(request.url)
        if file:  # and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            flash("File successfully uploaded")
            flash("Conducting data validation check")
            
            if problemType =="Regression":
                dc_obj = reg_dc(filepath)
            else:
                dc_obj = cla_dc(filepath)
            
            fileCheck = dc_obj.identify_file()
            if fileCheck == "None":
                flash("\nFile type is supported")
            else:
                flash("Error: ", fileCheck)
                flash("\nSupported file types: - [csv, tsv, xlsx, json]")
                sys.exit()

            df = dc_obj.file_to_dataframe()
            dataValidation = dc_obj.validation_check(df)
            if dataValidation == "None":
                flash("\nData format is supported")
            else:
                sys.exit(dataValidation)
            html = df.head().to_html(classes="table table-hover table-dark")

            # write html to file
            text_file = open(
                "/Users/vishalkundar/Downloads/Website/app/templates/index.html", "w"
            )
            text_file.write(html)
            text_file.close()
            return render_template("fileupload.html")


@app.route("/display_df", methods=["GET", "POST"])
def display_df():
    if request.method == "POST" or request.method == "GET":
        return render_template("table.html")


@app.route("/results", methods=["GET", "POST"])
def display_results():
    global df, problemType, dc_obj, dproutput, dpcoutput, bestmodel
    if request.method == "POST" or request.method == "GET":
        dep_col = str(request.form.get("depvar"))
        problemType = dc_obj.identify_problem(df, dep_col.strip())
        if problemType != "Regression" and problemType != "Classification":
            sys.exit(problemType)

        if problemType == "Regression":
            dproutput, bestmodel = reg_driver.runNoCodeML(
                df, dep_col
            )
        else:
            dpcoutput, bestmodel = cla_driver.runNoCodeML(
                df, dep_col
            )

        if df.shape[0] > 2000:
            profile = ProfileReport(df, minimal=True)
        else:
            profile = ProfileReport(df)
        profile.to_file(
            "/Users/vishalkundar/Downloads/Website/app/templates/user_report.html"
        )
        return render_template("Final_page.html")


@app.route("/user_report", methods=["GET", "POST"])
def display_user_report():
    return render_template("user_report.html")


@app.route("/predict", methods=["GET", "POST"])
def display_userReport():
    global problemType, dproutput, dpcoutput, bestmodel, predict_df
    if request.method == "POST" or request.method == "GET":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected for uploading")
            return redirect(request.url)
        if file:  # and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            flash("File successfully uploaded")
            flash("Conducting data validation check")
            
            if problemType == "Regresion":
                dc_obj = reg_dc(filepath)
            else:
                dc_obj = cla_dc(filepath)
                
            fileCheck = dc_obj.identify_file()
            if fileCheck == "None":
                flash("\nFile type is supported")
            else:
                flash("Error: ", fileCheck)
                flash("\nSupported file types: - [csv, tsv, xlsx, json]")
                sys.exit()

            df = dc_obj.file_to_dataframe()

            # Predict based on problem Type
            if problemType == "Regression":
                result_df = regres.predict_function(dproutput, bestmodel[0], df)
           
            else:
                i = 0 
                for key,val in bestmodel.items():
                    if i>=1:
                        break
                    else:
                        modeln = key
                        modeld = val
                    i+=1
                
                result_df = predict_function(dpcoutput, modeln,modeld, df)
                

            # Saving
            html = result_df.head().to_html(classes="table table-dark table-striped")

            # write html to file
            text_file = open(
                "/Users/vishalkundar/Downloads/Website/app/templates/predtable.html",
                "w",
            )
            text_file.write(html)
            text_file.close()
            return render_template("predict.html")


@app.route("/download_results", methods=["GET", "POST"])
def download_results_ascsv():
    path = "/Users/vishalkundar/Downloads/Website/predicted_data/results.csv"
    return send_file(path, as_attachment=True)

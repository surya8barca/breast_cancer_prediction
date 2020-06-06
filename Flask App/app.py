from flask import Flask,jsonify,request
import joblib
import pandas as pd
from flask.templating import render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('main.html')

@app.route("/predict/",methods=['GET'])
def predict():
    values=request.args
    ans=[[float(values['mean_radius']),float(values['mean_texture']),float(values['mean_perimeter']),float(values['mean_area']),float(values['mean_smoothness']),float(values['mean_compactness']),float(values['mean_concavity']),float(values['mean_concavepoints']),float(values['mean_symmetry']),float(values['mean_fractal_dimension']),float(values['radius_error']),float(values['texture_error']),float(values['perimeter_error']),float(values['area_error']),float(values['smoothness_error']),float(values['compactness_error']),float(values['concavity_error']),float(values['concave_points_error']),float(values['symmetry_error']),float(values['fractal_dimension_error']),float(values['worst_radius']),float(values['worst_texture']),float(values['worst_perimeter']),float(values['worst_area']),float(values['worst_smoothness']),float(values['worst_compactness']),float(values['worst_concavity']),float(values['worst_concavepoints']),float(values['worst_symmetry']),float(values['worst_fractal_dimension'])]]
    model_value=values['model']
    model_type=''
    values=''
    if(model_value=='gcv'):
        model_type='Grid Search CV'
        model=joblib.load('Grid_Search_CV_model.sav')
        prediction=model.predict(ans)
    else:
        model_type='Suport Vector Classifier'
        model=joblib.load('Support_Vector_Classifier_model.sav')
        prediction=model.predict(ans)
    if(prediction[0]==0):
       values='Breast Cancer Absent'
    else:
       values='Breast Cancer Present'
    return jsonify({'model':model_type,'result': values})


if __name__ == '__main__':
    app.run()
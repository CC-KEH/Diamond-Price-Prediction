from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictionPipeline


application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    
    else:
        data = CustomData(
            carat = float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
            cut = float(request.form.get('cut')),
            clarity = float(request.form.get('clarity')),
            color = float(request.form.get('color')), 
        )
        final_data = data.get_data_as_dataframe(data)
        prediction_pipeline = PredictionPipeline()
        prediction = prediction_pipeline.predict(final_data)
        
        result = round(prediction[0],2)
        
        return render_template('form.html',final_result = result)
        
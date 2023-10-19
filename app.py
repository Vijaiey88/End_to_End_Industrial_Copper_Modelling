from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Update the code to retrieve the input features from the form
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        
        # Add code to retrieve the additional input features for 'selling_price'
        item_date = request.form.get('item_date')
        quantity_tons = float(request.form.get('quantity_tons'))
        customer = request.form.get('customer')
        country = request.form.get('country')
        status = request.form.get('status')
        item_type = request.form.get('item_type')
        application = request.form.get('application')
        thickness = float(request.form.get('thickness'))
        width = float(request.form.get('width'))
        material_ref = request.form.get('material_ref')
        product_ref = request.form.get('product_ref')
        delivery_date = request.form.get('delivery_date')

        # Create a dictionary with the input features
        selling_price_data = {
            'item_date': item_date,
            'quantity_tons': quantity_tons,
            'customer': customer,
            'country': country,
            'status': status,
            'item_type': item_type,
            'application': application,
            'thickness': thickness,
            'width': width,
            'material_ref': material_ref,
            'product_ref': product_ref,
            'delivery_date': delivery_date,
        }
        
        # Combine the input features for 'selling_price' with the existing data
        data.update_data(selling_price_data)
        
        pred_df = data.get_data_as_data_frame()

        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")

        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")

from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Route for serving the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gather input data from the user
        con_Pclass = int(request.form['Pclass'])
        con_Age = float(request.form['Age'])
        con_SibSp = int(request.form['SibSp'])
        con_Parch = int(request.form['Parch'])
        con_Fare = float(request.form['Fare'])
        cat_Sex = request.form['Sex']
        cat_Embarked = request.form['Embarked']

        # Create a DataFrame from the user input
        input_data = {
            'Pclass': [con_Pclass],
            'Age': [con_Age],
            'SibSp': [con_SibSp],
            'Parch': [con_Parch],
            'Fare': [con_Fare],
            'Sex': [cat_Sex],
            'Embarked': [cat_Embarked]
        }
        input_df = pd.DataFrame(input_data)

        # Load the trained model
        with open('NOTEBOOK/pipeline_rfc_best_predict.pkl', 'rb') as file1:
            pipeline_rfc_best = pickle.load(file1)
        
        # Predict survival probability
        y_predict = pipeline_rfc_best.predict(input_df)
        prob = pipeline_rfc_best.predict_proba(input_df).max()

        # Construct prediction message
        prediction = f"Predicted label: {y_predict[0]}, Probability of survival: {prob:.2f}"
        return jsonify(prediction=prediction)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        return jsonify(error=error_message), 400

if __name__ == '__main__':
    app.run(debug=True)

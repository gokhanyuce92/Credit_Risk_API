from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = pd.DataFrame([data])
    
    # Encoder'ı yükleme
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    # Kategorik değişkenleri OneHotEncoder ile dönüştürme
    encoded_data = encoder.transform(features[['loan_grade', 'loan_intent', 'person_home_ownership', 'cb_person_default_on_file']])
    encoded_columns = encoder.get_feature_names_out(['loan_grade', 'loan_intent', 'person_home_ownership', 'cb_person_default_on_file'])
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

    features = features.join(encoded_df)
    features = features.drop(['loan_grade', 'loan_intent', 'person_home_ownership', 'cb_person_default_on_file'], axis=1)
    for index, row in features.iterrows():
        print(row)
    # Load the model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
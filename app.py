from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# 모델 로드와 레이블인코더들 로드
model = joblib.load('random_Forest_model.pkl')
label_encoder_대중소분류 = joblib.load('label_encoder_대중소분류.pkl')
label_encoder_차종 = joblib.load('label_encoder_차종.pkl')
label_encoder_차명 = joblib.load('label_encoder_차명.pkl')

def predict_processing_time(year, month, day, hour, truck_type, ship_count):
    input_data = pd.DataFrame([{
        '입문시각_연도': year,
        '입문시각_월': month,
        '입문시각_일': day,
        '입문시각_시간': hour,
        '차종': truck_type,
        '선박_갯수': ship_count
    }])
    
    # 차종에 인코딩 적용
    input_data['차종'] = label_encoder_차종.transform(input_data['차종'])
    
    predicted_time = model.predict(input_data)
    return predicted_time[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    year = data['year']
    month = data['month']
    day = data['day']
    hour = data['hour']
    truck_type = data['truck_type']
    ship_count = data['ship_count']
    
    prediction = predict_processing_time(year, month, day, hour, truck_type, ship_count)
    return jsonify({'predicted_time': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

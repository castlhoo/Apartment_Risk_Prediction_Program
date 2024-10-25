# app.py
from flask import Flask, request, jsonify, render_template
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

app = Flask(__name__)

# 1. Spark 세션 생성
spark = SparkSession.builder \
    .appName("Apartment Crime Safety Prediction") \
    .getOrCreate()

# 2. 저장된 모델 로드 및 파이프라인 정보 출력
model_save_path = "/home/username/CPTED_Model"
try:
    model = PipelineModel.load(model_save_path)
    print("\n=== 모델 파이프라인 정보 ===")
    for i, stage in enumerate(model.stages):
        print(f"Stage {i}: {stage}")
        if hasattr(stage, 'getInputCols'):
            print(f"Input Columns: {stage.getInputCols()}")
        if hasattr(stage, 'getOutputCol'):
            print(f"Output Column: {stage.getOutputCol()}")
    print("========================\n")
except Exception as e:
    print(f"모델 로드 실패: {str(e)}")
    model = None

# 3. 예측 함수 정의
def predict_crime_risk(accessibility, surveillance, crime_count):
    try:
        # Schema 정의
        schema = StructType([
            StructField("Accessibility", DoubleType(), False),
            StructField("Surveillance", DoubleType(), False),
            StructField("Crime Count", IntegerType(), False)  # 스페이스가 있는 원래 컬럼명 사용
        ])
        
        # 입력 데이터를 DataFrame으로 생성
        input_data = spark.createDataFrame(
            [(float(accessibility), float(surveillance), int(crime_count))],
            schema=schema
        )
        
        # 디버깅을 위한 DataFrame 정보 출력
        print("\n=== 입력 데이터 정보 ===")
        print("Input DataFrame Schema:")
        input_data.printSchema()
        print("Input Data:")
        input_data.show()
        
        # 모델을 사용해 예측 수행
        predictions = model.transform(input_data)
        
        # 디버깅을 위한 예측 결과 출력
        print("\n=== 예측 결과 정보 ===")
        print("Predictions DataFrame Schema:")
        predictions.printSchema()
        print("Predictions Data:")
        predictions.show()
        print("=====================\n")
        
        # 예측된 Risk Score 추출
        if predictions.count() > 0:
            predicted_risk = predictions.select("prediction").collect()[0]["prediction"]
            return int(predicted_risk)  # 예측값을 정수로 반환
        
    except Exception as e:
        print(f"예측 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    return None

# 4. Flask 라우팅: 메인 페이지 렌더링
@app.route('/')
def index():
    return render_template('index.html')

# 5. Flask 라우팅: 예측 요청 처리
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # HTML 폼으로부터 데이터 받기
        accessibility = float(request.form['Accessibility'])
        surveillance = float(request.form['Surveillance'])
        crime_count = int(request.form['CrimeCount'])
        
        # 입력값 로깅
        print(f"Input values: accessibility={accessibility}, surveillance={surveillance}, crime_count={crime_count}")
        
        # 입력값 유효성 검사
        if not (1 <= accessibility <= 5 and 1 <= surveillance <= 5 and crime_count >= 0):
            return jsonify({
                'success': False,
                'error': '잘못된 입력값입니다. 범위를 확인해주세요.'
            }), 400
        
        # 예측 수행
        predicted_risk_score = predict_crime_risk(accessibility, surveillance, crime_count)
        
        # 결과를 JSON으로 반환
        if predicted_risk_score is not None:
            return jsonify({
                'success': True,
                'predicted_risk_score': predicted_risk_score
            })
        else:
            return jsonify({
                'success': False,
                'error': '예측에 실패했습니다. 입력값을 확인해주세요.'
            }), 400
            
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': '잘못된 입력 형식입니다. 숫자만 입력해주세요.'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'서버 오류가 발생했습니다: {str(e)}'
        }), 500

# 6. Flask 애플리케이션 실행
if __name__ == '__main__':
    if model is None:
        print("경고: 모델이 로드되지 않았습니다.")
    app.run(host='0.0.0.0', port=5000, debug=True)
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql.functions import when, col
from pyspark.ml import PipelineModel

# Spark 세션 생성
spark = SparkSession.builder.appName("Apartment Crime Safety Prediction").getOrCreate()

# 1. 학습용 데이터 로드
train_data_path = "/home/username/APT_Crime_Data.csv"  # 학습용 CSV 파일 경로
train_data = spark.read.csv(train_data_path, header=True, inferSchema=True)

# 2. 필요한 열을 선택
train_data = train_data.select("Accessibility", "Surveillance", "CrimeCount", "Risk Score")

# 3. `Risk Score` 값이 100인 경우 99로 조정하고, 정수로 변환
train_data = train_data.withColumn("Risk Score", when(col("Risk Score") >= 100, 99).otherwise(col("Risk Score").cast("integer")))

# 중복된 feature 컬럼이 있는지 확인하고 삭제
if "features" in train_data.columns:
    train_data = train_data.drop("features")

# 4. Feature Vector 생성
assembler = VectorAssembler(inputCols=["Accessibility", "Surveillance", "CrimeCount"], outputCol="features")

# 5. 분류 모델 생성 (Random Forest Classifier)
rf = RandomForestClassifier(labelCol="Risk Score", featuresCol="features")

# 6. 파라미터 그리드 생성 (필요에 따라 하이퍼파라미터 추가 조정 가능)
paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [10, 20]).addGrid(rf.maxDepth, [5, 10]).build()

# 7. 평가자 설정 (정확도 평가)
evaluator = MulticlassClassificationEvaluator(labelCol="Risk Score", predictionCol="prediction", metricName="accuracy")

# 8. 파이프라인 설정
pipeline = Pipeline(stages=[assembler, rf])

# 9. 교차 검증 설정 (3-Fold Cross Validation)
crossval = CrossValidator(estimator=pipeline, 
                          estimatorParamMaps=paramGrid, 
                          evaluator=evaluator, 
                          numFolds=3)

# 10. 모델 학습 (교차 검증을 사용한 학습)
cv_model = crossval.fit(train_data)

# 11. 학습 데이터의 정확도 평가
train_predictions = cv_model.transform(train_data)
train_accuracy = evaluator.evaluate(train_predictions)
print(f"Cross-validated Train Accuracy = {train_accuracy}")

# **모델 저장**
model_save_path = "/home/username/CPTED_Model"
cv_model.bestModel.write().overwrite().save(model_save_path)
print(f"Model saved to {model_save_path}")

# Spark 세션 종료
spark.stop()

# 🏢 Apartment Risk Prediction Program 🏢

## 🛠 Tech Stack
- Python 🐍
- Flask 🌐
- HTML 🖥️

## 📋 Project Overview
This project builds on the previously developed CPTED (Crime Prevention Through Environmental Design) model and integrates it into a web application using Flask. CPTED is a model designed to evaluate and reduce crime risk in residential environments by focusing on two key aspects: **accessibility** and **surveillance**.

In this project, we measure these two factors to assess the crime vulnerability of apartment complexes. As shown in the comparison images below, accessibility can be influenced by physical barriers such as fences or vehicle barriers. Surveillance, on the other hand, can be enhanced through the presence of security guards or an increased number of CCTV cameras.

### 🔍 What This Model Does:
This model is intended for insurance companies, specifically in recommending comprehensive home insurance policies (such as burglary protection). By assessing an apartment's crime risk based on accessibility and surveillance, this tool helps insurance agents communicate effectively with clients, minimizing crime-related losses.

![Example Image](https://github.com/user-attachments/assets/bb85d50a-7fd4-45b7-91ee-3657fcefd626)
![Another Example Image](https://github.com/user-attachments/assets/5415a7f4-0d70-413f-8ea3-53043b785880)

![image](https://github.com/user-attachments/assets/5ba4d50d-e94c-4dd3-a341-9e9136e42d36)
![image](https://github.com/user-attachments/assets/cdd01020-ac93-4712-8c70-b2639c5a760e)


The model assigns a score between **1 and 5** for both accessibility and surveillance, with **higher scores** indicating a safer environment. Additionally, the model takes into account the **number of burglary incidents** in the area, ensuring more objective crime risk predictions.

## 🧠 Training the Model
The model is trained using PySpark's Random Forest Classifier to classify apartment crime risk based on three main features: **accessibility**, **surveillance**, and **crime count**. Below is the code snippet that demonstrates the training process:

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql.functions import when, col
from pyspark.ml import PipelineModel

# Initialize Spark session
spark = SparkSession.builder.appName("Apartment Crime Safety Prediction").getOrCreate()

# Load training data
train_data = spark.read.csv("/home/username/APT_Crime_Data.csv", header=True, inferSchema=True)

# Select necessary columns
train_data = train_data.select("Accessibility", "Surveillance", "CrimeCount", "Risk Score")

# Adjust and transform the 'Risk Score'
train_data = train_data.withColumn("Risk Score", when(col("Risk Score") >= 100, 99).otherwise(col("Risk Score").cast("integer")))

# Create feature vector
assembler = VectorAssembler(inputCols=["Accessibility", "Surveillance", "CrimeCount"], outputCol="features")

# Set up Random Forest Classifier
rf = RandomForestClassifier(labelCol="Risk Score", featuresCol="features")

# Configure hyperparameter grid
paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [10, 20]).addGrid(rf.maxDepth, [5, 10]).build()

# Set up pipeline
pipeline = Pipeline(stages=[assembler, rf])

# Perform cross-validation
crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=MulticlassClassificationEvaluator(labelCol="Risk Score", predictionCol="prediction", metricName="accuracy"), numFolds=3)

# Train model
cv_model = crossval.fit(train_data)

# Evaluate training accuracy
train_predictions = cv_model.transform(train_data)
train_accuracy = MulticlassClassificationEvaluator(labelCol="Risk Score", predictionCol="prediction", metricName="accuracy").evaluate(train_predictions)
print(f"Cross-validated Train Accuracy = {train_accuracy}")

# Save the best model
cv_model.bestModel.write().overwrite().save("/home/username/CPTED_Model")
spark.stop()
```

## 🖥️ Flask Integration
To integrate the trained model into a web application, we use Flask. The web interface accepts input data for **accessibility**, **surveillance**, and **crime count**, and outputs the predicted crime risk score. Below is the key part of the Flask application:

```python
from flask import Flask, request, jsonify, render_template
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

app = Flask(__name__)

# Initialize Spark session and load the model
spark = SparkSession.builder.appName("Apartment Crime Safety Prediction").getOrCreate()
model = PipelineModel.load("/home/username/CPTED_Model")

# Define prediction function
def predict_crime_risk(accessibility, surveillance, crime_count):
    input_data = spark.createDataFrame([(float(accessibility), float(surveillance), int(crime_count))], ["Accessibility", "Surveillance", "Crime Count"])
    predictions = model.transform(input_data)
    predicted_risk = predictions.select("prediction").collect()[0]["prediction"]
    return int(predicted_risk)

# Define Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        accessibility = float(request.form['Accessibility'])
        surveillance = float(request.form['Surveillance'])
        crime_count = int(request.form['CrimeCount'])
        predicted_risk_score = predict_crime_risk(accessibility, surveillance, crime_count)
        return jsonify({'success': True, 'predicted_risk_score': predicted_risk_score})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)
```

## 🌐 Frontend (HTML Form)
Here's how the web interface works. Users enter **accessibility**, **surveillance**, and **crime count** data. Upon submission, the model's prediction is returned as the crime risk score.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Apartment Crime Prediction</title>
</head>
<body>
    <h1>Predict Crime Risk for Apartments</h1>
    <form id="predictionForm">
        <label for="Accessibility">Accessibility (1-5):</label>
        <input type="number" id="Accessibility" name="Accessibility" min="1" max="5" required><br>
        
        <label for="Surveillance">Surveillance (1-5):</label>
        <input type="number" id="Surveillance" name="Surveillance" min="1" max="5" required><br>
        
        <label for="CrimeCount">Crime Count:</label>
        <input type="number" id="CrimeCount" name="CrimeCount" min="0" required><br>
        
        <button type="submit">Predict</button>
    </form>

    <div id="result"></div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const result = await response.json();
            document.getElementById('result').textContent = result.success ? `Predicted Risk Score: ${result.predicted_risk_score}` : `Error: ${result.error}`;
        });
    </script>
</body>
</html>
```

## 🎉 Final Prediction Result
Here is an example result of the prediction:
![image](https://github.com/user-attachments/assets/4d33973c-b5f9-41e4-aa79-4fb106c25698)

![Prediction Output](https://github.com/user-attachments/assets/e2d625e4-e083-4d04-99d4-ffc134137fdb)
When entering accessibility = 3, surveillance = 4, and a burglary count of 1233, the model outputs a risk score of 30. Based on this result, we can further recommend appropriate insurance products for the client.

## 🏁 Conclusion
Using Flask for web programming and combining it with my passion for crime prevention and financial insurance resulted in a valuable business model. The idea of preventing crime through accurate prediction models is both fascinating and impactful. This project helped fuel my interest in data analysis and predictive modeling, motivating me to pursue further projects in this area!

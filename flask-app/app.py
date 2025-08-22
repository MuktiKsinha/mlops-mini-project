from flask import Flask,render_template,request
import mlflow
from preprocessing_utility import normalize_text
import dagshub
import pickle

#set tracking uri
mlflow.set_tracking_uri('https://dagshub.com/MuktiKsinha/mlops-mini-project.mlflow')
dagshub.init(repo_owner='MuktiKsinha', repo_name='mlops-mini-project', mlflow=True)

app=Flask(__name__)

#load your model from model registry
model_name = "my_model"
model_version= 3   ###make it dynamic via code

model_uri=f'models:/{model_name}/{model_version}' # Set up MLflow tracking URI
model=mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict',methods=['POST'])
def predict():
    text=request.form['text']
    
    #clean
    text=normalize_text(text)

    #bow
    features=vectorizer.transform([text])

    #final prediction
    result=model.predict(features)

    #show user 

    return render_template('index.html', result= result[0])


app.run(debug=True)
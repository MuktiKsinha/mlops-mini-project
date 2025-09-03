# base file
FROM python:3.11
# working dir
WORKDIR /app

# copy
COPY flask_app/ /app/
COPY models/vectorizer.pkl /app/models/vectorizer.pkl
#run
RUN pip install -r requirements.txt

#nltk dowloader 
RUN python -m nltk.downloader stopwords wordnet

#Expose port 5000 for flask
EXPOSE 5000   
#command
#CMD ["python","app.py"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]

FROM continuumio/anaconda3
COPY . /usr/app
EXPOSE 8080
WORKDIR /usr/app
RUN pip install -r requirements.txt
CMD python spam_detector_app.py
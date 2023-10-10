import gradio as gr
from app import demo
import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    app = gr.mount_gradio_app(app, demo, path='/gradio')
    
    return {"message": "Hello World"}

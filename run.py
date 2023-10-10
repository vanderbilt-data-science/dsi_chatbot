from fastapi import FastAPI
import gradio as gr

from app import demo
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

app = gr.mount_gradio_app(app, demo, path='/gradio')

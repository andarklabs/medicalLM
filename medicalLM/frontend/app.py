import gradio as gr
from  inferand import Model as md

def chat(message):
    model = md("biogpt")
    reply = model.infer(message)
    return reply

demo = gr.Interface(fn=chat, 
                    inputs="text", 
                    outputs="text")
    
if __name__ == "__main__":
    demo.launch(show_api=False)   
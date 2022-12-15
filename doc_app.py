import fitz
import random
import gradio as gr
import textract
import docx2txt
import fitz
from striprtf.striprtf import rtf_to_text
import joblib
import json
import shap


def read_file(filename: str):
    if filename.endswith("docx"):
        text = docx2txt.process(filename)
    elif filename.endswith("pdf"):
        doc = fitz.open(filename)
        text = []
        for page in doc:
            text.append(page.get_text())
        text = " ".join(text)
    elif filename.endswith("doc"):
        text = reinterpret(textract.process(filename))
        text = remove_convert_info(text)
    elif filename.endswith("rtf"):
        with open(filename) as f:
            content = f.read()
            text = rtf_to_text(content)
    else:
        return {"text": []}
    return {"text": text}


def reinterpret(text: str):
    return text.decode('utf8')


def remove_convert_info(text: str):
    for i, s in enumerate(text):
        if s == ":":
            break
    return text[i + 6:]


with gr.Blocks() as demo:
    gr.Interface(read_file, "file", "json")


def classifier(text):
    model = joblib.load('model_v0_1.joblib')
    vectorizer = joblib.load('vectorizer_v0_1.joblib')
    cls = {0: "Договоры поставки", 1: "Договоры оказания услуг", 2: "Договоры подряда", 3: "Договоры аренды", 4: "Договоры купли-продажи"}
    vec = vectorizer.transform(text)
    print(vec)
    return {"label": cls[model.predict(vec.reshape(1, -1))]}



def interpretation_function(text):
    explainer = shap.Explainer(classifier)
    shap_values = explainer([text])

    # Dimensions are (batch size, text size, number of classes)
    # Since we care about positive sentiment, use index 1
    scores = list(zip(shap_values.data[0], shap_values.values[0, :, 1]))


    # Scores contains (word, score) pairs


    return {"original": text, "interpretation": scores}

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            file = gr.File(label="Input File")
            with gr.Row():
                classify = gr.Button("Classify document")
                interpret = gr.Button("Interpret")
        with gr.Column():
            label = gr.Label(label="Predicted Document Class")
        with gr.Column():
            interpretation = gr.components.Interpretation(file)
    classify.click(classifier, file, label)
    interpret.click(interpretation_function, file, interpretation)


if __name__ == "__main__":
    demo.launch()
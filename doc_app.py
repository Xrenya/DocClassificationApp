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


def read_file(file_obj):
    if isinstance(file_obj, list):
        file_obj = file_obj[0]
    filename = file_obj.name
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
    return text#{"text": text}


def reinterpret(text: str):
    return text.decode('utf8')


def remove_convert_info(text: str):
    for i, s in enumerate(text):
        if s == ":":
            break
    return text[i + 6:]




def classifier(file_obj):
    text  = read_file(file_obj)
    model = joblib.load('model_v0_1.joblib')
    vectorizer = joblib.load('vectorizer_v0_1.joblib')
    cls = {0: "Договоры поставки", 1: "Договоры оказания услуг", 2: "Договоры подряда", 3: "Договоры аренды", 4: "Договоры купли-продажи"}
    classes = ["Договоры поставки", "Договоры оказания услуг", "Договоры подряда", "Договоры аренды", "Договоры купли-продажи"]
    vec = vectorizer.transform([text])
    # output = {"Договоры поставки": 0, "Договоры оказания услуг": 0, "Договоры подряда": 0, "Договоры аренды": 0, "Договоры купли-продажи": 0}
    cls[model.predict(vec)[0]]
    # for idx, c in cls.items():
    #     output
    proba = model.predict_proba(vec)[0]
    out = {}
    for c, val in zip(classes, proba):
        out[c] = val
    return out



def interpretation_function(file_obj):
    text  = read_file(file_obj)
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
            # # with gr.Blocks() as demo:
            # file = gr.Interface(file, "file", "json")
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

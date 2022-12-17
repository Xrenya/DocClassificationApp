import io
import re
import string

import docx2txt
import fitz
import gradio as gr
import joblib
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import shap
import textract
import torch
from lime.lime_text import LimeTextExplainer
from striprtf.striprtf import rtf_to_text
from transformers import BertForSequenceClassification, BertTokenizer, pipeline

from preprocessing import TextCleaner

cleaner = TextCleaner()
pipe = joblib.load('pipe_v1_natasha.joblib')

model_path = "finetunebert"
tokenizer = BertTokenizer.from_pretrained(model_path,
                                          padding='max_length',
                                          truncation=True)
# tokenizer.init_kwargs["model_max_length"] = 512
model = BertForSequenceClassification.from_pretrained(model_path)
document_classifier = pipeline("text-classification",
                               model=model,
                               tokenizer=tokenizer,
                               return_all_scores=True)

classes = [
    "Договоры поставки", "Договоры оказания услуг", "Договоры подряда",
    "Договоры аренды", "Договоры купли-продажи"
]


def old__pipeline(text):
    clean_text = text_preprocessing(text)
    tokens = tokenizer.batch_encode_plus([clean_text],
                                         max_length=512,
                                         padding=True,
                                         truncation=True)
    item = {k: torch.tensor(v) for k, v in tokens.items()}
    preds = model(**item).logits.detach()
    preds = torch.softmax(preds, dim=1)[0]
    output = [{
        'label': cls,
        'score': score
    } for cls, score in zip(classes, preds)]

    return output


def read_doc(file_obj):
    """Read file
    :param file_obj: file object
    :return: string
    """
    text = read_file(file_obj)
    return text


def read_docv2(file_obj):
    """Read file and collect neighbour for visual output
    :param file_obj: file object
    :return: string
    """
    text = read_file(file_obj)
    explainer = LimeTextExplainer(class_names=classes)
    text = cleaner.execute(text)
    exp = explainer.explain_instance(text,
                                     pipe.predict_proba,
                                     num_features=10,
                                     labels=[0, 1, 2, 3, 4])
    scores = exp.as_list()
    scores_desc = sorted(scores, key=lambda t: t[1])[::-1]
    selected_words = [word[0] for word in scores_desc]
    sent = text.split()
    indices = [i for i, word in enumerate(sent) if word in selected_words]
    neighbors = []
    for ind in indices:
        neighbors.append(" ".join(sent[max(0, ind - 3):min(ind +
                                                           3, len(sent))]))
    return "\n\n".join(neighbors)


def classifier(file_obj):
    """Classify
    :param file_obj: file object
    :return: Dict[str, int]
    """
    text = read_file(file_obj)
    clean_text = text_preprocessing(text)
    tokens = tokenizer.batch_encode_plus([clean_text],
                                         max_length=512,
                                         padding=True,
                                         truncation=True)
    item = {k: torch.tensor(v) for k, v in tokens.items()}
    preds = model(**item).logits.detach()
    preds = torch.softmax(preds, dim=1)[0]
    return {cls: p.item() for cls, p in zip(classes, preds)}


def clean_text(text):
    """Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers."""
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def text_preprocessing(text):
    """Cleaning and parsing the text."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    #remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(tokenized_text)
    return combined_text


def read_file(file_obj):
    """Read file and fixing encoding
    :param file_obj: file object
    :return: string
    """
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
    return text


def reinterpret(text: str):
    return text.decode('utf8')


def remove_convert_info(text: str):
    for i, s in enumerate(text):
        if s == ":":
            break
    return text[i + 6:]


def plot_weights(file_obj):
    text = read_file(file_obj)
    explainer = LimeTextExplainer(class_names=classes)
    text = cleaner.execute(text)
    exp = explainer.explain_instance(text,
                                     pipe.predict_proba,
                                     num_features=10,
                                     labels=[0, 1, 2, 3, 4])
    scores = exp.as_list()
    scores_desc = sorted(scores, key=lambda t: t[1])[::-1]
    plt.rcParams.update({'font.size': 35})
    fig = plt.figure(figsize=(20, 20))
    sns.barplot(x=[s[0] for s in scores_desc[:10]],
                y=[s[1] for s in scores_desc[:10]])
    plt.title("Top words contributing to positive sentiment")
    plt.ylabel("Weight")
    plt.xlabel("Word")
    plt.title("Interpreting text predictions with LIME")
    plt.xticks(rotation=20)
    plt.tight_layout()
    return fig


def interpretation_function(file_obj):
    text = read_file(file_obj)
    clean_text = text_preprocessing(text)
    explainer = shap.Explainer(document_classifier)
    shap_values = explainer([clean_text])

    # Dimensions are (batch size, text size, number of classes)
    # Since we care about positive sentiment, use index 1
    scores = list(zip(shap_values.data[0], shap_values.values[0, :, 1]))
    # Scores contains (word, score) pairs
    # Format expected by gr.components.Interpretation
    return {"original": clean_text, "interpretation": scores}


def as_pyplot_figure(file_obj):
    text = read_file(file_obj)
    explainer = LimeTextExplainer(class_names=classes)
    text = cleaner.execute(text)
    exp = explainer.explain_instance(text,
                                     pipe.predict_proba,
                                     num_features=10,
                                     labels=[0, 1, 2, 3, 4])
    buf = io.BytesIO()
    fig = exp.as_pyplot_figure()
    fig.tight_layout()
    plt.rcParams.update({'font.size': 10})
    plt.savefig(buf)
    buf.seek(0)
    return fig


with gr.Blocks() as demo:
    gr.Markdown("""**Document classification**""")
    with gr.Row():
        with gr.Column():
            file = gr.File(label="Input File")
            with gr.Row():
                classify = gr.Button("Classify document")
                read = gr.Button("Get text")
                interpret_lime = gr.Button("Interpret LIME")
                interpret_shap = gr.Button("Interpret SHAP")
        with gr.Column():
            label = gr.Label(label="Predicted Document Class")
            plot = gr.Plot()
        with gr.Column():
            text = gr.Text(label="Selected keywords")
        with gr.Column():
            interpretation = gr.components.Interpretation(text)
    classify.click(classifier, file, label)
    read.click(read_docv2, file, [text])
    interpret_shap.click(interpretation_function, file, interpretation)
    interpret_lime.click(as_pyplot_figure, file, plot)

if __name__ == "__main__":
    demo.launch()

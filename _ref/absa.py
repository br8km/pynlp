
# https://huggingface.co/spaces/ywuan/model_api/blob/main/app.py


import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn

name = ["negative","neutral","positive"]


def main_note(sentence,aspect):
    tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
    model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
    # model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-large-absa-v1.1")
    input_str = "[CLS]" + sentence + "[SEP]" + aspect + "[SEP]"
    # input_str = "[CLS] when tables opened up, the manager sat another party before us. [SEP] manager [SEP]"
    inputs = tokenizer(input_str, return_tensors="pt")
    outputs = model(**inputs)
    softmax = nn.Softmax(dim=1)
    outputs = softmax(outputs.logits)
    result = [round(i,4) for i in outputs.tolist()[0]]
    # print(result)
    return dict(zip(name,result))

# main_note("","")

iface = gr.Interface(
    fn = main_note,
    inputs=["text","text"],
    outputs = gr.outputs.Label(),
    examples=[["1.) Instead of being at the back of the oven, the cord is attached at the front right side.","cord"],
    ["The pan I received was not in the same league as my old pan, new is cheap feeling and does not have a plate on the bottom.","pan"],
    ["The pan I received was not in the same league as my old pan, new is cheap feeling and does not have a plate on the bottom.","bottom"],
    ["They seem much more durable and less prone to staining, retaining their white properties for a much longer period of time.","durability"],
    ["It took some time to clean and maintain, but totally worth it!","clean"],
    ["this means that not only will the smallest burner heat up the pan, but it will also vertically heat up 1\" of the handle.","handle"]])

iface.launch()
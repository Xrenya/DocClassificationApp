## Russian document classification application

## Application UI:
#### Simple user interface
![image](https://user-images.githubusercontent.com/51479797/207982633-5b59cf2b-92f4-4a5f-85a1-2d9f51d9fd49.png)
#### UI for prediction
![image](https://user-images.githubusercontent.com/51479797/207985762-e46ce031-bb2f-4480-9e0b-7cd62113b2bf.png)
#### Example UI explainablity:
.. Pending

## Progress:

1. Supported files extensions:
    - pdf
    - rtf
    - doc
    - docx
2. Inference:
    - Prediction label
    - Model explainablity: words weights/attention
3. UI:
    - Allow upload user files
    - Visualize predicted label
    - Visualzie model explainability of its prediction.
    
#### Model metrics on test dataset (20%):
|Metrics        |      |
|---------------|------|
|accuracy_score |0.9583|
|precision_score|0.9583|
|f1_score       |0.9583|
|recall_score   |0.9583|

## TODO:
1. - [X] Make a UI page using gradio with SHAP
    - Solution: [gradioXshap](https://gradio.app/advanced_interface_features/#interpreting-your-predictions), [custom](https://gradio.app/custom_interpretations_with_blocks/)
2. - [X] Unify documents: accept doc/docx to transform into pdf and process pdf file  
    - Solution: [doc2pdf](https://stackoverflow.com/questions/6011115/doc-to-pdf-using-python), [docx2pdf](https://ysko909.github.io/posts/docx-convert-to-pdf-with-python/)
3. - [X] Train two simple model (Bag-of-words, Tf-Idf), text preprocessing 
4. - [ ] Fune-tune RuBert model
5. - [ ] Make a presentation

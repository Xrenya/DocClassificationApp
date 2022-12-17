## Russian document classification application

## Application UI:
#### Simple user interface
![image](https://user-images.githubusercontent.com/51479797/207982633-5b59cf2b-92f4-4a5f-85a1-2d9f51d9fd49.png)
#### UI for prediction
![image](https://user-images.githubusercontent.com/51479797/207985762-e46ce031-bb2f-4480-9e0b-7cd62113b2bf.png)
#### Example UI explainablity:
![image](https://user-images.githubusercontent.com/51479797/208134217-53c79844-1743-4489-acd6-95252b14674b.png)
#### Final UI
![image](https://user-images.githubusercontent.com/51479797/208227961-0d6ff870-0ae2-42c1-85d0-797b2e6d582c.png)


## Progress:
#### Developed:
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
4. Model analysis using eli5:
    - Identified keywords which model using to classify documents, only '1' and '2' classes have bias as a top feature, which probably should be tackles on the next stage. 
5. SHAP words highlight based on bert output:  
![image](https://user-images.githubusercontent.com/51479797/208228018-fa495161-dfb2-487c-9abb-8a99bc29f899.png)

    
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
4. - [X] Fune-tune RuBert model
5. - [X] Make a presentation

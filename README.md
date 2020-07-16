You can use the MOCO.py to train the ERNIE model on the MOCO task.

****1. Data Preparation

You need to prepare a tsv file which contain the two augmentation of samples(Back-Translation would be recommended).
Then youc an put the file in the path: ./mocodata/train

****2. Train on the MOCO task

you can use the following command:python3 MOCO.py

****3. Finetune

we recommend you use the class ErnieModelForSequenceClassification to build the ERNIE model of your own, like finetune.py(for CoLA task) 

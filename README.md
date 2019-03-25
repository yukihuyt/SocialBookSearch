# SocialBookSearch
A text classification and linking assignment based on a social book search and task finished by SBS lab on 2016 with LibraryThing and Reddit posts data. This assignment was the final project assignmnet of course Text Mining in Leiden University, only study/education purpose usage is allowed. 


Environment config:
```shell
conda env create -f environment.yml
pip install -r requirements.txt
```
_**During the whole experiments period, please remember to check the directory you save your data and change the usage in scripts with correct corresponding ones.**_

<!-- text -->

To run the experiments, first you need to process the .xml file with `xml_repro.py` with the original xml data or use my processed csv files directly. Detailed information of data can be found the `data` folder.     

Then if you want to train a doc2vec model with given training data, run `exp.py`.  

To use non-deep learning models, run `svc_mlp.py` with train and test data, and run `deep_exp.py` for deep experiments. `nnmodel.py` stores the applied neural network models implemented with keras.

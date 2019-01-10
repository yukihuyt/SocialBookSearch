import os
import pandas as pd
import  xml.dom.minidom

LT_train = "D:\学习\作业\TM\FinalAS\SBS\data\sbs16mining-classification-training-threads-librarything" #文件夹目录
LT_test = "D:\学习\作业\TM\FinalAS\SBS\data\sbs16mining-classification-test-threads-librarything"
Reddit_train = "D:\学习\作业\TM\FinalAS\SBS\data\sbs16mining-classification-training-threads-reddit\sbs16mining-classification-training-threads-reddit"
Reddit_test = "D:\学习\作业\TM\FinalAS\SBS\data\sbs16mining-classification-test-threads-reddit"
label_dir = "D:\学习\作业\TM\FinalAS\SBS\data"
LT_tag = 'raw_text'
Reddit_tag = 'title'

def load_data(data_dir, label_path, text_tag, second_tag=None):
    data_files= os.listdir(data_dir)
    df_text = pd.DataFrame(columns=['id','text'])

    row_id = 0
    for file in data_files:
         if not os.path.isdir(file):
            dom = xml.dom.minidom.parse(data_dir+"/"+file)
            root = dom.documentElement
            text = root.getElementsByTagName(text_tag)[0].firstChild.data
            if second_tag:
                second_node = root.getElementsByTagName(second_tag)[0].firstChild
                if second_node is not None:
                    text = text+': '+second_node.data
            if root.hasAttribute('id'):
                id = root.getAttribute("id")
            else:
                id = root.getElementsByTagName('thread')[0].getAttribute("id")

            if '/'in id:
                id = id.split('/')[1]

            df_text.loc[row_id]={'id':id, 'text':text}
            row_id = row_id+1

    diff_file = 'sbs16mining-classification-test-labels-librarything'

    df_label = pd.DataFrame(columns=['id', 'label'])
    row_id = 0
    with open(label_path, 'r') as f:
        for line in f.readlines():
            line_ele = line.split('\t')
            if diff_file in label_path:
                df_label.loc[row_id]={'id':line_ele[0],'label':line_ele[2]}
            else:
                df_label.loc[row_id] = {'id': line_ele[0],'label': (line_ele[1].split('\n'))[0]}
            row_id = row_id+1

    df_text.set_index('id',inplace=True)
    df_label.set_index('id',inplace=True)
    df=pd.concat([df_text, df_label], axis=1, join='inner')

    return df


if __name__ == "__main__":
    LT_train_label = label_dir + '\sbs16mining-classification-training-labels-librarything.csv'
    LT_test_label = label_dir+'\sbs16mining-classification-test-labels-librarything.csv'
    Reddit_train_label = label_dir + '\sbs16mining-classification-training-labels-reddit.csv'
    Reddit_test_label = label_dir+'\sbs16mining-classification-test-labels-reddit.csv'

    df_LT_train = load_data(LT_train, LT_train_label, LT_tag)
    df_LT_test = load_data(LT_test, LT_test_label, LT_tag)
    df_Reddit_train = load_data(Reddit_train, Reddit_train_label, Reddit_tag, second_tag='body')
    df_Reddit_test = load_data(Reddit_test, Reddit_test_label, Reddit_tag, second_tag='body')

    df_LT_train.to_csv(label_dir+'\LT_train.csv')
    df_LT_test.to_csv(label_dir + '\LT_test.csv')
    df_Reddit_train.to_csv(label_dir + '\Reddit_train.csv')
    df_Reddit_test.to_csv(label_dir + '\Reddit_test.csv')












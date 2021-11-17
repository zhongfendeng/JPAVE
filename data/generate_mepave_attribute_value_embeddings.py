import torch
from transformers import BertModel, BertTokenizer
from collections import defaultdict
import json

model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
#tokenizer.save_vocabulary('bert_vocab.txt')
params = list(bert_model.base_model.parameters())
params2 = list(bert_model.named_parameters())
para = bert_model.base_model.parameters()

all_labels_bert_embedding = defaultdict()
all_attributes_bert_embedding = defaultdict()


# load label vocab
id2label_dict = defaultdict()
label2id_dict = defaultdict()
with open('../vocab/label.dict', 'r') as f_in:
    for i, line in enumerate(f_in):
        data = line.rstrip().split('\t')
        assert len(data) == 2
        label2id_dict[data[0]] = i
        id2label_dict[i] = data[0]

# load attribute vocab
id2attribute_dict = defaultdict()
attribute2id_dict = defaultdict()
with open('../vocab/attribute.dict', 'r') as f_in:
    for i, line in enumerate(f_in):
        data = line.rstrip().split('\t')
        assert len(data) == 2
        attribute2id_dict[data[0]] = i
        id2attribute_dict[i] = data[0]

labeltokens2labelids_map = defaultdict()
tagmaster_file = "tagmaster.json"
tagmaster_dict = {}
with open(tagmaster_file, 'r') as jf:
    for i, line in enumerate(jf):
        data = json.loads(line)
        attribute = data['attribute']
        attr_input_ids = tokenizer.encode(attribute, add_special_tokens=True)
        attr_input_ids = torch.tensor([attr_input_ids])
        # obtain last hidDen state of BERT
        with torch.no_grad():
            attr_output = bert_model(attr_input_ids)
            attr_last_hidden_states = attr_output.last_hidden_state
            attr_cls_embedding = attr_last_hidden_states[:, 0, :]
            attr_cls_embedding = attr_cls_embedding.squeeze(0).numpy().tolist()
            all_attributes_bert_embedding[attribute2id_dict[attribute]]=attr_cls_embedding
        values = data['values']
        tagmaster_dict[attribute]=values
        for each_label in values:
            label_text = attribute + "[SEP]" + each_label
            input_ids = tokenizer.encode(label_text, add_special_tokens=True)
            input_ids = torch.tensor([input_ids])
            # obtain last hidDen state of BERT
            with torch.no_grad():
                output = bert_model(input_ids)
                last_hidden_states = output.last_hidden_state
                cls_embedding = last_hidden_states[:, 0, :]
                cls_embedding = cls_embedding.squeeze(0).numpy().tolist()
                all_labels_bert_embedding[label2id_dict[each_label]]=cls_embedding
                # last_hidden_states = bert_model(input_ids)[0]  # Models outputs are now tuples
            # build labeltokens2labelids_map here
            input_ids_temp = tokenizer.encode(each_label, add_special_tokens=False)
            #label_text_tokenized_ids.append({'id':each_label,'text':input_ids_temp})
            label_id = label2id_dict[each_label]
            for each_tokenid in input_ids_temp:
                if each_tokenid not in labeltokens2labelids_map:
                    labeltokens2labelids_map[each_tokenid]=[label_id]
                else:
                    if label_id not in labeltokens2labelids_map[each_tokenid]:
                        labeltokens2labelids_map[each_tokenid].append(label_id)

bert_vocab = tokenizer.vocab
for k,v in bert_vocab.items():
    if v not in labeltokens2labelids_map:
        labeltokens2labelids_map[v]=[]
with open('mepave_labeltokens2labelids_map.json','w') as jf:
    json.dump(labeltokens2labelids_map,jf)
with open('mepave_value_embeddings.json', 'w') as jf:
    json.dump(all_labels_bert_embedding,jf)
with open('mepave_attribute_embeddings.json', 'w') as jf:
    json.dump(all_attributes_bert_embedding,jf)



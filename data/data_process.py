import json
tagmaster={}
for tag in ["train", "valid", "test"]:

    print(tag)
    fin = "./jdair.jave.{}.txt".format(tag)
    #fin = "./jdai.jave.fashion.{}.sample".format(tag)
    lines = open(fin, "r", encoding="utf-8").read().strip().split("\n")
    input_seqs = []
    output_seqs = []
    output_labels = []
    indexs = []
    data = []
    
    for line in lines:
        items = line.split("\t")
        cid = items[0]
        sid = items[1]
        doc = items[2].lower()
        doc_p = items[3].lower()
        input_seq = []
        output_seq = []
        attrs = []
        attr_values = []
        attr_value_nestedlist = []
        attr_values_map_dict = {}
        index = 0

        try:
            assert " " not in doc + doc_p
            while index < len(doc_p):
                if doc_p[index] == "<":
                    index += 1
                    attr = ""
                    while doc_p[index] != ">":
                        attr += doc_p[index]
                        index += 1
                    index += 1
                    input_seq.append(doc_p[index])
                    output_seq.append("B-"+attr)
                    attr_value = doc_p[index]
                    index += 1
                    while doc_p[index] != "<":
                        input_seq.append(doc_p[index])
                        output_seq.append("I-"+attr)
                        attr_value += doc_p[index]
                        index += 1
                    index += 1
                    assert doc_p[index] == "/"
                    index += 1
                    attr_end = ""
                    while doc_p[index] != ">":
                        attr_end += doc_p[index]
                        index += 1
                    index += 1
                    assert attr_end == attr
                    attrs.append(attr)
                    if attr not in attr_values_map_dict:
                        attr_values_map_dict[attr]=[attr_value]
                    else:
                        #original_list = attr_values_map_dict[attr]
                        attr_values_map_dict[attr].append(attr_value)
                    # build global tagmaster below
                    if attr not in tagmaster:
                        tagmaster[attr]=[attr_value]
                    else:
                        if attr_value not in tagmaster[attr]:
                            tagmaster[attr].append(attr_value)
                    #attr_values.append(attr_value)
                    #attr_value_nestedlist.append([attr, attr_value])
                else:
                    input_seq.append(doc_p[index])
                    output_seq.append("O")
                    index += 1
            assert "".join(input_seq) == doc
            
            indexs.append(cid + "\t" + sid)
            input_seqs.append(input_seq)
            output_seqs.append(output_seq)
            if attrs == []:
                #attrs = ["[PAD]"]
                attrs = []
                unique_attrs =[]
                attr_values=[]
                attr_value_nestedlist=[]
            else:
                unique_attrs=[]
                attr_values = []
                attr_value_nestedlist = []
                for k,v in attr_values_map_dict.items():
                    unique_attrs.append(k)
                    attr_values += v
                    attr_value_nestedlist.append([k]+v)
            output_labels.append(sorted(list(set(attrs))))
            data.append({"token":input_seq,"label_nestedlist": attr_value_nestedlist,"attribute":unique_attrs,"label":attr_values})
        
        except (AssertionError, IndexError):
            print("wrong line:", doc, doc_p, "".join(input_seq), "".join(output_seq))
            #exit()

    assert len(input_seqs) == len(output_seqs) == len(output_labels) == len(indexs)==len(data)
    print("data num:", len(input_seqs))


    with open("./{}/indexs".format(tag), "w", encoding="utf-8") as w:
        for index in indexs:
            w.write(index + "\n")

    with open("./{}/input.seq".format(tag), "w", encoding="utf-8") as w:
        for input_seq in input_seqs:
            w.write(" ".join(input_seq).lower() + "\n")

    with open("{}.json".format(tag), 'w', encoding='utf-8') as jf:
        for line in data:
            json.dump(line, jf, ensure_ascii=False)
            jf.write('\n')

    with open("./{}/output.seq".format(tag), "w", encoding="utf-8") as w:
        for output_seq in output_seqs:
            w.write(" ".join(output_seq) + "\n")

    with open("./{}/output.label".format(tag), "w", encoding="utf-8") as w:
        for output_label in output_labels:
            w.write(" ".join(output_label) + "\n")


tagmaster_list = []
for k, v in tagmaster.items():
    tagmaster_list.append({'attribute': k, 'values': v})
with open('tagmaster.json','w',encoding="utf-8") as jf:
    for line in tagmaster_list:
        json.dump(line, jf, ensure_ascii=False)
        jf.write('\n')


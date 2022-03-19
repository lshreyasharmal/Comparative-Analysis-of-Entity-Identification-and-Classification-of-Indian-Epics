import pandas

def relabel_tokens(label_list, living_entity_tags, non_living_entity_tags):
    print("relabelling tokens...")
    new_label_list = []
    
    if label_list:
        for label in label_list:
            if label in living_entity_tags:
                new_label_list.append(1)
            elif label in non_living_entity_tags:
                new_label_list.append(2)
            else:
                new_label_list.append(0)
    print("relabelling done!")
    return new_label_list
    
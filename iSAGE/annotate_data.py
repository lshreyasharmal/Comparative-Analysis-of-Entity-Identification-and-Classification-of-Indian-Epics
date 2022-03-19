from utility import *

living_entity_tags = ['ANIMAL','PERSON','GROUP','TITLE']
non_living_entity_tags = ['BOOK','PLACE','WEAPON','SPECIAL_OBJECT','PLANT','CONCEPT','WATER']

relabel_tokens(['O','ANIMAL', 'BOOK'], living_entity_tags, non_living_entity_tags)

import pandas as pd
annotations = pd.read_csv("./chap1-4_maha.csv", index_col=['id'])

relabelled_tags = relabel_tokens(annotations.tag.tolist(), living_entity_tags, non_living_entity_tags)

annotations['relabelled_tags'] = relabelled_tags
annotations.to_csv("annotations.csv")
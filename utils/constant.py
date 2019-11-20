"""
Define constants.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1
SAME_TOKEN = '<SAME_TOKEN>'

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]


# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'ORGANIZATION': 2, 'PERSON': 3}

OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

# DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, SAME_TOKEN: 24, 'dep': 2, 'aux': 3, 'auxpass': 4, 'cop': 5, 'ccomp': 6, 'xcomp': 6, 'mark': 6, 'compound': 6, 'compound:prt': 6, 'dobj': 7, 'iobj': 8, 'punct': 9, 'nsubj': 10, 'nsubjpass': 11, 'csubj': 12, 'csubjpass': 13, 'cc': 14, 'cc:preconj': 14, 'conj': 15, 'expl': 16, 'amod': 17, 'appos': 17, 'advcl': 17, 'det': 17, 'advmod': 17, 'neg': 17, 'nmod': 17, 'nmod:poss': 17, 'nummod': 17, 'nmod:tmod': 17, 'nmod:npmod': 17, 'det:predet': 17, 'parataxis': 18, 'case': 19, 'ROOT': 20, 'root': 20, 'acl:relcl': 21, 'acl': 21, 'mwe': 22, 'discourse': 23}
# DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'dep': 2, 'aux': 3, 'auxpass': 4, 'cop': 5, 'ccomp': 6, 'xcomp': 6, 'mark': 6, 'compound': 6, 'compound:prt': 6, 'dobj': 7, 'iobj': 8, 'punct': 9, 'nsubj': 10, 'nsubjpass': 11, 'csubj': 12, 'csubjpass': 13, 'cc': 14, 'cc:preconj': 14, 'conj': 15, 'expl': 16, 'amod': 17, 'appos': 17, 'advcl': 17, 'det': 17, 'advmod': 17, 'neg': 17, 'nmod': 17, 'nmod:poss': 17, 'nummod': 17, 'nmod:tmod': 17, 'nmod:npmod': 17, 'det:predet': 17, 'parataxis': 18, 'case': 19, 'ROOT': 20, 'root': 20, 'acl:relcl': 21, 'acl': 21, 'mwe': 22, 'discourse': 23, 'r-dep': 24, 'r-aux': 25, 'r-auxpass': 26, 'r-cop': 27, 'r-ccomp': 28, 'r-xcomp': 28, 'r-mark': 28, 'r-compound': 28, 'r-compound:prt': 28, 'r-dobj': 29, 'r-iobj': 30, 'r-punct': 31, 'r-nsubj': 32, 'r-nsubjpass': 33, 'r-csubj': 34, 'r-csubjpass': 35, 'r-cc': 36, 'r-cc:preconj': 36, 'r-conj': 37, 'r-expl': 38, 'r-amod': 39, 'r-appos': 39, 'r-advcl': 39, 'r-det': 39, 'r-advmod': 39, 'r-neg': 39, 'r-nmod': 39, 'r-nmod:poss': 39, 'r-nummod': 39, 'r-nmod:tmod': 39, 'r-nmod:npmod': 39, 'r-det:predet': 39, 'r-parataxis': 40, 'r-case': 41, 'r-ROOT': 42, 'r-root': 42, 'r-acl:relcl': 43, 'r-acl': 43, 'r-mwe': 44, 'r-discourse': 45}

DEPREL_COUNT = 22
NEGATIVE_LABEL = 'no_relation'

LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}
# LABEL_TO_ID = {'Other': 0, 'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2, 'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4, 'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6, 'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8, 'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10, 'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12, 'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14, 'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16, 'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}
INFINITY_NUMBER = 1e12

from spacy import util
import re
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support
from celery import shared_task, current_task
import random
from pathlib import Path
import spacy
from collections import defaultdict
import os
import pandas as pd
import hashlib
import django
import datetime
import copy
import json
import math
from django.contrib.contenttypes.models import ContentType
from django.db import IntegrityError
#os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'active_learning.settings')
#django.setup()
#from spacyal.models import al_project


def mix_train_data(nlp, TRAIN_DATA):
    res = []
    for text, annotations in TRAIN_DATA:
        doc = nlp(text)
        new_ents = []
        for ner in doc.ents:
            for x in annotations['entities']:
                if x[0] > ner.start_char:
                    new_ents.append((
                        ner.start_char, ner.end_char, ner.label_, 1))
                    break
                elif x[0] == ner.start_char and x[1] == ner.end_char and x[3] == 1:
                    new_ents.append(x)
                    break
                elif x[0] == ner.start_char and x[1] == ner.end_char and x[3] == 2:
                    break
        for ner in annotations:
            if ner not in new_ents and ner[3] == 1:
                new_ents.append(ner)
        res.append((text, {'entities': [(x[0], x[1], x[2]) for x in new_ents]}))
    return res


def dec_use(index, row, lst_cases, lst_ners, lst_hashes):
    z = copy.deepcopy(lst_ners)
    res = False
    for x in lst_cases:
        if x.hash == index:
            if x.decission == 1:
                return False
            elif x.decission == 2 or x.decission == 0:
                try:
                    z.remove(x.suggestion)
                    #print(z, row[z].fillna(0).idxmax())
                    res = row[z].fillna(0).idxmax()
                    print(row[res])
                    if math.isnan(row[res]):
                        res = False
                    elif (index, res) in lst_hashes:
                        res = False
                except Exception as e:
                    print(e)
                    res = False
    return res


class test_model():
    @staticmethod
    def find_tok_number(toklist, start, end):
        count = 0
        count2 = 0
        start_tok = False
        end_tok = False
        while count < start:
            count += len(toklist[count2])
            count2 += 1
        start_tok = count2
        while count <= end:
            count += len(toklist[count2])
            count2 += 1
        end_tok = count2-1
        return start_tok, end_tok

    @staticmethod
    def harmonize_doc(doc, ent_attr):
        res = ['', {ent_attr: []}]
        int_plus = 0
        for d in doc:
            if len(res[0]) > 0:
                res[0] += ' ' + d[0]
                int_plus += 1
            else:
                res[0] += d[0]
            for ent in d[1][ent_attr]:
                ent_new = (ent[0]+int_plus, ent[1]+int_plus, ent[2])
                if len(ent) == 4:
                    ent_new = (ent[0]+int_plus, ent[1]+int_plus, ent[2], ent[3])
                res[1][ent_attr].append(ent_new)
            int_plus += len(d[0])
        return res

    def compute_f1(self):
        f1 = precision_recall_fscore_support(self.text_1, self.text_2, average='weighted', labels=self.positive_labels)
        return {'precission': f1[0], 'recall': f1[1], 'fbeta': f1[2], 'support': f1[3]}

    def __init__(self, text, nlp, attribute='entities'):
        text = self.harmonize_doc(text, 'entities')
        self.text = text
        t1 = re.split(r'(\s)', text[0])
        t2 = t1
        a1 = []
        a2 = []
        start_end_lst = ['{}-{}'.format(x[0], x[1]) for x in text[1][attribute]]
        nlp_ents = [(e.start_char, e.end_char, e.label_) for e in nlp(text[0]).ents if '{}-{}'.format(e.start_char, e.end_char) in start_end_lst]
        lst_attr = {'Nothing': 0}
        print(start_end_lst)
        print([(e.start_char, e.end_char, e.label_) for e in nlp(text[0]).ents])
        print(nlp_ents)
        for a, t, lst_ents in [(a1, t1, text[1][attribute]), (a2, t2, nlp_ents)]:
            for ent in lst_ents:
                print(ent)
                if len(ent) == 4:
                    if ent[3] != 1:
                        continue
                if ent[2] not in lst_attr.keys():
                    lst_attr[ent[2]] = len(lst_attr.keys())+1
                start, end = self.find_tok_number(t, ent[0], ent[1])
                print(start, end)
                a.extend([0]*(start-len(a)))
                a.extend([lst_attr[ent[2]]]*(end-start+1))
            a.extend([0]*(len(t1)-len(a)))
        self.text_1 = a1
        self.text_2 = a2
        self.labels = lst_attr
        self.positive_labels = [lst_attr[x] for x in lst_attr.keys() if x != 'Nothing']
            

@shared_task(time_limit=1800)
def retrain_model(project, model=None, n_iter=30):
    """Load the model, set up the pipeline and train the entity recognizer."""
    dropout_rates = util.decaying(util.env_opt('dropout_from', 0.2),
                                  util.env_opt('dropout_to', 0.2),
                                  util.env_opt('dropout_decay', 0.0))
    batch_sizes = util.compounding(util.env_opt('batch_from', 1),
                                   util.env_opt('batch_to', 16),
                                   util.env_opt('batch_compound', 1.001))
    if model == 'model_1':
        output_model = 'model_2'
    else:
        output_model = 'model_1'
    al_project = ContentType.objects.get(app_label="spacyal",
                                         model="al_project").model_class()
    project = al_project.objects.get(pk=project)
    base_d = '/'.join(project.texts.path.split('/')[:-1])
    output_dir = os.path.join(base_d, output_model)
    if project.len_training_data() < project.num_retrain:
        message = {'folder': os.path.join(base_d, model),
                   'retrained': False, 'project': project.pk}
        return message
    TRAIN_DATA, eval_data, hist_object = project.get_training_data(
        include_all=True, include_negative=True)
    nlp = spacy.load(os.path.join(base_d, model))# load existing spaCy model
    if project.project_history_set.all().count() == 1:
        project_history = ContentType.objects.get(
            app_label="spacyal", model="project_history").model_class()
        ev = test_model(eval_data, nlp)
        f1 = ev.compute_f1()
        hist2 = project_history.objects.create(
            project=project, eval_f1=f1['fbeta'],
            eval_precission=f1['precission'], eval_recall=f1['recall'])
        hist2.cases_training.add(*list(hist_object.cases_training.all()))
        hist2.cases_evaluation.add(*list(hist_object.cases_evaluation.all()))
    TRAIN_DATA = mix_train_data(nlp, TRAIN_DATA)
    with open(os.path.join(base_d, 'training_data.json'), 'w') as outp:
        json.dump(TRAIN_DATA, outp)
    r = nlp.get_pipe('ner')
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    count_prog_list = list(range(0, n_iter, int(n_iter/10)))
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            if itn in count_prog_list:
                current_task.update_state(
                    state='PROGRESS',
                    meta={'progress': count_prog_list.index(itn)*10,
                          'model': output_model, 'project': project.pk})
            random.shuffle(TRAIN_DATA)
            losses = {}
            for batch in util.minibatch(TRAIN_DATA, size=batch_sizes):
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=next(dropout_rates),  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
    if not Path(output_dir).exists():
        Path(output_dir).mkdir()
    nlp.to_disk(output_dir)
    print(eval_data)
    ev = test_model(eval_data, nlp)
    f1 = ev.compute_f1()
    hist_object.eval_f1 = f1['fbeta']
    hist_object.eval_precission = f1['precission']
    hist_object.eval_recall = f1['recall']
    hist_object.model_path = output_dir
    hist_object.save()
    message = {'folder': output_dir, 'retrained': True, 'project': project.pk, 'f1': f1}
    return message


@shared_task(time_limit=300)
def get_cases(project, model='model_1', retrained=True):
    al_project = ContentType.objects.get(app_label="spacyal", model="al_project").model_class()
    project = al_project.objects.get(pk=project)
    base_d = '/'.join(project.texts.path.split('/')[:-1])
    case = ContentType.objects.get(app_label="spacyal", model="case").model_class()

    directory = os.fsencode(os.path.join(base_d, 'texts'))
    df_cols = ['start', 'end', 'start_char',
               'end_char', 'sentence', 'dist 1-2', 'max score',
               'median score', 'file']
    df_b = pd.DataFrame(columns=df_cols)
    lst_cases = list(case.objects.filter(project=project).defer(
        'start', 'end', 'start_char', 'end_char', 'text_file', 'user', 'project'
    ))
    lst_hashes = [(x.hash, x.suggestion) for x in lst_cases]
    if retrained or project.last_cases_list is None:
        model = spacy.load(os.path.join(base_d, model))
        for file in os.listdir(directory):
            txt = open(os.path.join(directory, file), 'r').read()
            with model.disable_pipes('ner'):
                doc = model(txt)
            (beams, somethingelse) = model.entity.beam_parse(
                [doc], beam_width=16, beam_density=0.0001)
            entity_scores = defaultdict(float)
            for beam in beams:
                for score, ents in model.entity.moves.get_beam_parses(beam):
                    for start, end, label in ents:
                        hsh = hashlib.md5((os.fsdecode(os.path.join(directory, file))+':'+str(start)+'-'+str(end)).encode('utf8')).hexdigest()
                        entity_scores[(start, end, hsh, label)] += score
            for k in entity_scores.keys():
                if k[0] < 0 or k[1] < 0 or doc[k[0]:k[1]].text == '\n' or not re.search(r'[a-zA-Z0-9]+', doc[k[0]:k[1]].text):
                    continue
                df_b.loc[k[2], 'start'] = k[0]
                df_b.loc[k[2], 'end'] = k[1]
                df_b.loc[k[2], 'file'] = os.path.join(directory, file).decode('utf8')
                df_b.loc[k[2], k[3]] = entity_scores[k]
                for sent in doc.sents:
                    if sent.start <= k[0] and sent.end >= k[1]:
                        s = [t.text_with_ws for t in sent if t.i < k[0]]
                        s.append('<mark>'+doc[k[0]:k[1]].text+'</mark>')
                        s.append(doc[k[1]-1].whitespace_)
                        s.extend([t.text_with_ws for t in sent if t.i >= k[1]])
                        df_b.loc[k[2], 'sentence'] = ''.join(s).strip()
                        df_b.loc[k[2], 'start_char'] = doc[:k[0]].end_char+1
                        df_b.loc[k[2], 'end_char'] = doc[:k[1]].end_char
    elif project.last_cases_list is not None:
        df_b = pd.DataFrame.from_csv(os.path.join(
            base_d, project.last_cases_list))
    lst_ners = [a for a in df_b.columns if a not in df_cols]

    if project.threshold_probability:
        if 0 < project.threshold_probability < 1:
            for row in df_b[lst_ners].iterrows():
                if row[1].max() < project.threshold_probability:
                    df_b.drop([row[0]], inplace=True)
    df_b['max score'] = df_b[lst_ners].max(axis=1)
    df_b['median score'] = df_b[lst_ners].median(axis=1)
    df_b = df_b.sort_values(by=[project.al_strategy])
    count = 0
    it_row = df_b.iterrows()
    end_of_examples = False
    while count < (project.num_retrain + project.num_plus_retrain):
        try:
            index, row = next(it_row)
        except StopIteration:
            end_of_examples = True
            break
        s_label = False
        if (index, row[lst_ners].fillna(0).idxmax()) in lst_hashes:
            dec = dec_use(index, row, lst_cases, lst_ners, lst_hashes)
            if dec:
                s_label = dec
            else:
                continue
        else:
            s_label = row[lst_ners].fillna(0).idxmax()
        count += 1
        if s_label and not math.isnan(row[s_label]):
            try:
                c = case.objects.create(
                    sentence=row['sentence'], hash=index, project=project,
                    suggestion=s_label, probability=row[s_label],
                    text_file=row['file'], start=row['start'], end=row['end'], use=True,
                    start_char=row['start_char'], end_char=row['end_char'])
            except IntegrityError:
                print("hash: {} / suggestion: {} already exists".format(index, s_label))
    ts = str(datetime.datetime.now()).split('.')[0].replace(':', '_').replace(' ', '_')
    f_path = os.path.join(
        base_d, 'data/df_cases_{}_{}.csv'.format(project.pk, ts))
    df_b.to_csv(f_path, encoding="utf8")
    project.last_cases_list = 'data/df_cases_{}_{}.csv'.format(project.pk, ts)
    project.save()
    return 'created {} / {}'.format(f_path, end_of_examples)

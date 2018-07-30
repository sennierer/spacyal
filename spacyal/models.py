from django.db import models
from django.contrib.auth.models import User
import spacy
import os
from spacyal.tasks import get_cases
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import uuid
from zipfile import ZipFile
from django.db.models.signals import post_save
from django.db import transaction
from stat import S_IRWXU, S_IRWXG, S_IROTH


fs_models = getattr(settings, "SPACY_AL_STORAGE", 'data')
fs_media = getattr(settings, "MEDIA_ROOT")
fs_store = os.path.join(fs_media, fs_models)


def get_filepath_texts(instance, filename):
    print(fs_store)
    return os.path.join(fs_store, '{}/texts.zip'.format(uuid.uuid4()))


def get_filepath_model(instance, filename):
    return os.path.join(fs_store, '{}/model.zip'.format(uuid.uuid4()))


class al_project(models.Model):
    name = models.CharField(max_length=255)
    ch_strategy = (('max score', 'max score'),
                   ('median score', 'median score'))
    texts = models.FileField(upload_to=get_filepath_texts)
    model = models.FileField(
        upload_to=get_filepath_model, blank=True, null=True)
    num_retrain = models.PositiveSmallIntegerField(
        verbose_name='number of examples before retraining',
        default=10)
    num_plus_retrain = models.PositiveSmallIntegerField(
        default=5,
        verbose_name='number of cases to work ahead')
    ner_type = models.CharField(max_length=50,
                                verbose_name='ner types to work on',
                                null=True, blank=True)
    keep_models = models.BooleanField(
        default=False,
        verbose_name='wether to keep old models after retraining')
    al_strategy = models.CharField(
        max_length=50, choices=ch_strategy,
        verbose_name='strategy to use for picking cases')
    language_model = models.CharField(verbose_name='spacy language abbreviation',
                                      default='de', max_length=3)
    last_cases_list = models.CharField(
        blank=True, null=True, verbose_name="used to save last cases csv sheet",
        max_length=255)
    users_allowed = models.ManyToManyField(User)

    def __str__(self):
        return self.name

    def len_training_data(self, include_all=False, include_negative=False):
        q = {'decission__in': [1], 'project_id': self.pk}
        if include_negative:
            q['decission__in'].append(2)
        if not include_all:
            q['use'] = True
        return case.objects.filter(**q).count()

    def get_training_data(self, include_all=False, include_negative=False):
        res = dict()
        q = {'decission__in': [1], 'project_id': self.pk}
        if include_negative:
            q['decission__in'].append(2)
        if not include_all:
            q['use'] = True
        for c in case.objects.filter(**q).order_by('start_char'):
            print(c)
            if c.text_file not in res.keys():
                res[c.text_file] = [(
                    c.start_char, c.end_char, c.suggestion, c.decission)]
            else:
                res[c.text_file].append((
                    c.start_char, c.end_char, c.suggestion, c.decission))
            c.use = False
            c.save()
        res2 = []
        for k in res.keys():
            txt = open(k, 'r').read()
            res2.append((txt, {'entities': res[k]}))
        return res2


class case(models.Model):
    choices = ((0, 'skip'), (1, 'correct'), (2, 'wrong'))
    decission = models.PositiveSmallIntegerField(choices=choices, null=True, blank=True)
    user = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    project = models.ForeignKey(al_project, on_delete=models.CASCADE)
    hash = models.CharField(verbose_name='hash of the sentence file and position',
                            max_length=255)
    sentence = models.TextField(verbose_name='field to store the complete case')
    suggestion = models.CharField(max_length=50, null=True, blank=True,
                                  verbose_name='the models suggestion')
    probability = models.FloatField(null=True, blank=True,
                                    verbose_name='probability of the model for the case')
    use = models.BooleanField(default=False,
                              verbose_name='whether to use the case in the next iteration')
    start = models.PositiveIntegerField(verbose_name='token offset start')
    end = models.PositiveIntegerField(verbose_name='token offset end')
    start_char = models.PositiveIntegerField(verbose_name='charcter offset start')
    end_char = models.PositiveIntegerField(verbose_name='character offset end')
    text_file = models.CharField(max_length=255,
                                 verbose_name='path of the text file')

    def as_dict(self):
        return {
            'id': self.pk,
            'hash': self.hash,
            'sentence': self.get_sentence()
        }

    def get_sentence(self):
        n = '<mark data-id="{}" data-ner="{}" data-probability="{}">'.format(
            self.pk, self.suggestion, self.probability)
        return self.sentence.replace('<mark>', n).replace('\n', '')

    class Meta:
        unique_together = ("hash", "suggestion", "project")


def start_get_cases(sender, instance, created, **kwargs):
    def on_commit():
        get_cases.delay(instance.pk, model=os.path.join(base_d, 'model_1'))
    if created:
        base_d = '/'.join(instance.texts.path.split('/')[:-1])
        os.makedirs(os.path.join(base_d, 'data'))
        os.makedirs(os.path.join(base_d, 'model_2'))
        if not instance.model:
            nlp = spacy.load(instance.language_model)
            nlp.to_disk(
                os.path.join(base_d, 'model_1'))
        else:
            with ZipFile(instance.model) as myzip:
                myzip.extractall(path=os.path.join(base_d, 'model_1'))
        with ZipFile(instance.texts) as myzip:
            myzip.extractall(path=os.path.join(base_d, 'texts'))
        os.chmod(base_d, S_IRWXU | S_IRWXG | S_IROTH)
        for root, dirs, files in os.walk(base_d):
            for momo in dirs:
                os.chmod(
                    os.path.join(root, momo),
                    S_IRWXU | S_IRWXG | S_IROTH)
            for momo in files:
                os.chmod(
                    os.path.join(root, momo),
                    S_IRWXU | S_IRWXG | S_IROTH)
        transaction.on_commit(on_commit)


post_save.connect(start_get_cases, sender=al_project)

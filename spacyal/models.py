import math
import os
import random
import uuid
from stat import S_IROTH, S_IRWXG, S_IRWXU
from zipfile import ZipFile

import spacy
from django.conf import settings
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from django.db import models, transaction
from django.db.models.signals import post_save

from spacyal.tasks import get_cases

fs_models = getattr(settings, "SPACY_AL_STORAGE", "data")
fs_media = getattr(settings, "MEDIA_ROOT")
fs_store = os.path.join(fs_media, fs_models)


def get_filepath_texts(instance, filename):
    print(fs_store)
    return os.path.join(fs_store, "{}/texts.zip".format(uuid.uuid4()))


def get_filepath_model(instance, filename):
    return os.path.join(fs_store, "{}/model.zip".format(uuid.uuid4()))


class al_project(models.Model):
    name = models.CharField(max_length=255)
    ch_strategy = (("max score", "max score"), ("median score", "median score"))
    texts = models.FileField(upload_to=get_filepath_texts)
    model = models.FileField(upload_to=get_filepath_model, blank=True, null=True)
    num_retrain = models.PositiveSmallIntegerField(
        verbose_name="number of examples before retraining", default=40
    )
    num_plus_retrain = models.PositiveSmallIntegerField(
        default=5, verbose_name="number of cases to work ahead"
    )
    threshold_probability = models.FloatField(
        null=True, blank=True, verbose_name="probability threshold"
    )
    evaluation_share = models.FloatField(
        null=True,
        blank=True,
        verbose_name="share of evaluation cases",
        help_text="specify a share of decided cases to use for evaluation",
    )
    ner_type = models.CharField(
        max_length=50, verbose_name="ner types to work on", null=True, blank=True
    )
    keep_models = models.BooleanField(
        default=False, verbose_name="wether to keep old models after retraining"
    )
    al_strategy = models.CharField(
        max_length=50,
        choices=ch_strategy,
        verbose_name="strategy to use for picking cases",
    )
    language_model = models.CharField(
        verbose_name="spacy language abbreviation", default="de", max_length=3
    )
    last_cases_list = models.CharField(
        blank=True,
        null=True,
        verbose_name="used to save last cases csv sheet",
        max_length=255,
    )
    users_allowed = models.ManyToManyField(User)

    def __str__(self):
        return self.name

    def len_training_data(
        self, include_all=False, include_negative=False, evaluation_data=True
    ):
        q = {"decission__in": [1], "project_id": self.pk}
        if include_negative:
            q["decission__in"].append(2)
        if not include_all:
            q["use"] = True
        if evaluation_data:
            nmb = case.objects.filter(**q).count()
            return nmb - math.floor(nmb * self.evaluation_share)
        else:
            return case.objects.filter(**q).count()

    def get_training_data(
        self, include_all=False, include_negative=False, evaluation_data=True
    ):
        res = dict()
        q = {"decission__in": [1], "project_id": self.pk}
        if include_negative:
            q["decission__in"].append(2)
        if not include_all:
            q["use"] = True
        d_cases = list(case.objects.filter(**q).order_by("start_char"))
        history = project_history.objects.create(project=self)
        if evaluation_data:
            res_ev = dict()
            print(self.evaluation_share)
            nmb_cases = math.floor(len(d_cases) * self.evaluation_share)
            ev_cases = random.sample(d_cases, nmb_cases)
            t_cases = [x for x in d_cases if x not in ev_cases]
            history.cases_evaluation.add(*ev_cases)
            for c in ev_cases:
                if c.text_file not in res_ev.keys():
                    res_ev[c.text_file] = [
                        (c.start_char, c.end_char, c.suggestion, c.decission)
                    ]
                else:
                    res_ev[c.text_file].append(
                        (c.start_char, c.end_char, c.suggestion, c.decission)
                    )
                c.use = True
                c.save()
            res_ev2 = []
            for k in res_ev.keys():
                txt = open(k, "r", encoding="utf8").read()
                res_ev2.append((txt, {"entities": res_ev[k]}))

        else:
            t_cases = d_cases
        history.cases_training.add(*t_cases)
        for c in t_cases:
            if c.text_file not in res.keys():
                res[c.text_file] = [
                    (c.start_char, c.end_char, c.suggestion, c.decission)
                ]
            else:
                res[c.text_file].append(
                    (c.start_char, c.end_char, c.suggestion, c.decission)
                )
            c.use = False
            c.save()
        res2 = []
        for k in res.keys():
            txt = open(k, "r", encoding="utf8").read()
            res2.append((txt, {"entities": res[k]}))
        if evaluation_data:
            return res2, res_ev2, history
        else:
            return res2


class case(models.Model):
    choices = ((0, "skip"), (1, "correct"), (2, "wrong"))
    decission = models.PositiveSmallIntegerField(choices=choices, null=True, blank=True)
    user = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    project = models.ForeignKey(al_project, on_delete=models.CASCADE)
    hash = models.CharField(
        verbose_name="hash of the sentence file and position", max_length=255
    )
    sentence = models.TextField(verbose_name="field to store the complete case")
    suggestion = models.CharField(
        max_length=50, null=True, blank=True, verbose_name="the models suggestion"
    )
    probability = models.FloatField(
        null=True, blank=True, verbose_name="probability of the model for the case"
    )
    use = models.BooleanField(
        default=False, verbose_name="whether to use the case in the next iteration"
    )
    start = models.PositiveIntegerField(verbose_name="token offset start")
    end = models.PositiveIntegerField(verbose_name="token offset end")
    start_char = models.PositiveIntegerField(verbose_name="charcter offset start")
    end_char = models.PositiveIntegerField(verbose_name="character offset end")
    text_file = models.CharField(max_length=255, verbose_name="path of the text file")

    def as_dict(self):
        return {"id": self.pk, "hash": self.hash, "sentence": self.get_sentence()}

    def get_sentence(self):
        n = '<mark data-id="{}" data-ner="{}" data-probability="{}">'.format(
            self.pk, self.suggestion, self.probability
        )
        return self.sentence.replace("<mark>", n).replace("\n", "")

    class Meta:
        unique_together = ("hash", "suggestion", "project")


class project_history(models.Model):
    project = models.ForeignKey(al_project, on_delete=models.CASCADE)
    cases_training = models.ManyToManyField(case, related_name="cases_training")
    cases_evaluation = models.ManyToManyField(case, null=True, blank=True, related_name="cases_evaluation")
    eval_f1 = models.FloatField(blank=True, null=True)
    eval_precission = models.FloatField(blank=True, null=True)
    eval_recall = models.FloatField(blank=True, null=True)
    model_path = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now=True)


def start_get_cases(sender, instance, created, **kwargs):
    def on_commit():
        get_cases.delay(instance.pk, model=os.path.join(base_d, "model_1"))

    if created:
        base_d = "/".join(instance.texts.path.split("/")[:-1])
        os.makedirs(os.path.join(base_d, "data"))
        os.makedirs(os.path.join(base_d, "model_2"))
        if not instance.model:
            nlp = spacy.load(instance.language_model)
            nlp.to_disk(os.path.join(base_d, "model_1"))
        else:
            with ZipFile(instance.model) as myzip:
                myzip.extractall(path=os.path.join(base_d, "model_1"))
        with ZipFile(instance.texts) as myzip:
            myzip.extractall(path=os.path.join(base_d, "texts"))
        os.chmod(base_d, S_IRWXU | S_IRWXG | S_IROTH)
        for root, dirs, files in os.walk(base_d):
            for momo in dirs:
                os.chmod(os.path.join(root, momo), S_IRWXU | S_IRWXG | S_IROTH)
            for momo in files:
                os.chmod(os.path.join(root, momo), S_IRWXU | S_IRWXG | S_IROTH)
        transaction.on_commit(on_commit)


post_save.connect(start_get_cases, sender=al_project)

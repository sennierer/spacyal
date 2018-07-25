import django_tables2 as tables
from django_tables2.utils import A
from .models import al_project


class AlProjectsTable(tables.Table):
    name = tables.LinkColumn('spacyal:al_project-detail', args=[A('pk')])

    class Meta:
        model = al_project
        fields = ['name', 'al_strategy', 'ner_type', 'num_retrain']
        # add class="paleblue" to <table> tag
        attrs = {"class": "table table-hover table-striped table-condensed"}

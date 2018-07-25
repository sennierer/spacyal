from django.views.generic.detail import DetailView
from django_tables2 import SingleTableView
from .models import al_project
from .tables import AlProjectsTable


class AlProject(DetailView):
    template_name = 'spacyal/al_project.html'
    model = al_project


class AlProjectList(SingleTableView):
    template_name = 'spacyal/al_project_list.html'
    table_class = AlProjectsTable
    model = al_project

    def get_queryset(self, **kwargs):
        qs = super(AlProjectList, self).get_queryset()
        if self.request.user.is_superuser:
            return qs
        else:
            return qs.filter(users_allowed=self.request.user)

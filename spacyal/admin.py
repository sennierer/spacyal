from django import forms
from django.contrib import admin

from .models import al_project, case


class ALProjectForm(forms.ModelForm):

    class Meta:
        model = al_project
        exclude = ['last_cases_list']


class ALProjectAdmin(admin.ModelAdmin):
    form = ALProjectForm


admin.site.register(al_project, ALProjectAdmin)
admin.site.register(case)

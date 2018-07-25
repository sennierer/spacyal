from django.urls import path
from . import views

app_name = 'spacyal'

urlpatterns = [
    path('al_project/<int:pk>/', views.AlProject.as_view(),
         name='al_project-detail'),
    path('al_project/list/', views.AlProjectList.as_view(),
         name='al_project-list'),
]

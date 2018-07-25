from django.urls import path
from . import api_views

app_name = 'spacyal_api'

urlpatterns = [
    path('retrievecases/', api_views.RetrieveCasesView.as_view(),
         name='retrievecases'),
    path('progress_model/', api_views.GetProgressModelView.as_view(),
         name='progress_model'),
    path('download_model/', api_views.DownloadModelView.as_view(),
         name='download_model'),
    path('download_cases/', api_views.DownloadCasesView.as_view(),
         name='download_cases'),
]

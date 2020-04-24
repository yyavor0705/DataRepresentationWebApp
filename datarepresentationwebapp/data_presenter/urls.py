from django.urls import path

from . import views


app_name = "data_presenter"
urlpatterns = [
    path('', views.MainPage.as_view(), name="data_presentation"),
    path('run', views.OriginalDataView.as_view(), name="run"),
    path('calibrate', views.CalibrateDataView.as_view(), name="calibrate"),
    path('postproc', views.PostprocView.as_view(), name="postproc"),
    path('report', views.ReportView.as_view(), name="report"),
    path('originaltemplate', views.OriginalTemplateDownloadView.as_view(), name="original-template"),
    path('calibratetemplate', views.CalibrateTemplateDownloadView.as_view(), name="calibrate-template"),
]
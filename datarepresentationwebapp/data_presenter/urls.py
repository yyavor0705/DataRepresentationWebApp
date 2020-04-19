from django.urls import path

from . import views


app_name = "data_presenter"
urlpatterns = [
    path('', views.MainPage.as_view(), name="data_presentation"),
    path('run', views.OriginalDataView.as_view(), name="run"),
    path('calibrate', views.CalibrateDataView.as_view(), name="calibrate"),
    path('re-run', views.ReRunView.as_view(), name="run-rerun"),
    path('report', views.ReportView.as_view(), name="report"),
    path('template', views.download, name="template"),
]
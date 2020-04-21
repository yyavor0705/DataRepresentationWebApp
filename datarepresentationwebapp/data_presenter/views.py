import os
import base64
import pandas as pd
from django.views import View
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import render, HttpResponse, Http404

from prototype.model_func import Model


class MainPage(LoginRequiredMixin, View):

    def get(self, request):
        file_data = request.session.get("file_data")
        if not file_data:
            file_data = "1234"
            request.session["file_data"] = file_data

        return render(request=request, template_name="data_presentation.html")


class OriginalDataView(LoginRequiredMixin, View):

    def post(self, request):
        sample_number = request.POST.get("sample_number")
        parameter_a = request.POST.get("parameter-a")
        parameter_b_1 = bool(request.POST.get("parameter-b-1"))
        parameter_b_2 = bool(request.POST.get("parameter-b-2"))
        parameter_b_3 = bool(request.POST.get("parameter-b-3"))
        request.session["sample_number"] = sample_number
        xlsx_file = request.FILES['orig-data-file']
        df = pd.read_excel(xlsx_file)
        request.session['orig_json_df'] = df.to_json()
        sample_number = int(sample_number)
        df, image_bytes = Model.run(df, sample_number)
        base64_image = base64.b64encode(image_bytes.getvalue())
        return HttpResponse(content=base64_image, content_type='image/png')


class CalibrateDataView(LoginRequiredMixin, View):

    def post(self, request):
        sample_number = request.session.get('sample_number')
        orig_json_df = request.session.get('orig_json_df')
        calibrate_json_df = request.session.get('calibrate-data-file')
        if orig_json_df:
            df = pd.read_json(orig_json_df)
        else:
            xlsx_file = request.FILES['calibrate-data-file']
            df = pd.read_excel(xlsx_file)
            request.session['calibrate_json_df'] = df.to_json()
        sample_number = int(sample_number)
        df, image_bytes = Model.calibrate(df, sample_number)
        base64_image = base64.b64encode(image_bytes.getvalue())
        return HttpResponse(content=base64_image, content_type='image/png')


class ReportView(LoginRequiredMixin, View):

    def get(self, request):
        image_bytes = Model.report()
        base64_image = base64.b64encode(image_bytes.getvalue())
        return HttpResponse(content=base64_image, content_type='image/png')


class ReRunView(LoginRequiredMixin, View):

    def get(self, request):
        sample_number = request.session.get('sample_number')
        orig_json_df = request.session.get('orig_json_df')
        df = pd.read_json(orig_json_df)
        sample_number = int(sample_number)
        df, image_bytes = Model.calibrate(df, sample_number)
        base64_image = base64.b64encode(image_bytes.getvalue())
        return HttpResponse(content=base64_image, content_type='image/png')


@login_required
def download(request):
    file_path = os.path.join(settings.DOWNLOAD_FILES_ROOT, "template.xlsx")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
    raise Http404

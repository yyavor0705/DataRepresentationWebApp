import os
import io
import pickle
import base64
import pandas as pd
from django.views import View
from django.conf import settings
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import render, HttpResponse, Http404

from .models import DataModelPickle
from .ReportObjects.ReportsCollection import ReportsCollection
from prototype_update.carbonation import Carbonation_Model, load_df_R_ACC


class DataModelKeys:
    ORIGINAL_MODEL = "orig_model"
    CALIBRATE_MODEL = "calibrated_model"


class Wrapper:
    pass


class MainPage(LoginRequiredMixin, View):

    def get(self, request):
        return render(request=request, template_name="data_presentation.html")


class OriginalDataView(LoginRequiredMixin, View):

    def _get_pars(self, df_pars, choose, cement_type):
        pars = Wrapper()  # empty class to store raw parameters

        pars.cover_mean = df_pars.cover_mean.values[
            0]  # mm .values[0] returns a number rather than an array; number works in model.run()
        pars.cover_std = df_pars.cover_std.values[0]
        pars.RH_real = df_pars.RH_real.values[0]
        pars.t_c = df_pars.RH_real.values[0]
        pars.x_c = df_pars.x_c.values[0]  # m
        pars.ToW = df_pars.ToW.values[0]
        pars.p_SR = df_pars.p_SR.values[0]

        pars.option = Wrapper()  # empty sub class to store a sub group of raw parameters

        pars.option.choose = choose  # from boolean check in the UI
        pars.option.cement_type = cement_type  # from drop down select in the UI
        # CEM_I_42.5_R
        # CEM_I_42.5_R+FA
        # CEM_I_42.5_R+SF
        # CEM_III/B_42.5
        pars.option.wc_eqv = 0.6
        pars.option.df_R_ACC = load_df_R_ACC()  # load a df defined in carbonation.py
        pars.option.plot = True
        return pars

    def post(self, request):
        years = float(request.POST.get("year"))
        request.session["years"] = years

        cement_type = request.POST.get("parameter-a")
        choose = bool(request.POST.get("parameter-b"))

        xlsx_file = request.FILES['orig-data-file']
        df = pd.read_excel(xlsx_file)

        pars = self._get_pars(df, choose, cement_type)

        carb_model = Carbonation_Model(pars)
        carb_model.run(years)

        file = io.BytesIO()
        pickle.dump(carb_model, file)
        file.seek(0)

        origin_data_model_pickle, _ = DataModelPickle.objects.get_or_create(
            name=DataModelKeys.ORIGINAL_MODEL,
            user=request.user
        )
        origin_data_model_pickle.pickle = file.getvalue()
        origin_data_model_pickle.save()
        return HttpResponse()


class PostprocView(LoginRequiredMixin, View):

    def get(self, request):
        model_pickle = DataModelPickle.objects.get(user=request.user, name=DataModelKeys.ORIGINAL_MODEL)
        orig_model_bytes = model_pickle.pickle

        file = io.BytesIO(orig_model_bytes)
        orig_model = pickle.load(file)
        image_bytes, report_text = orig_model.postproc(True)

        base64_image = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
        content = {
            "image": base64_image,
            "text": report_text
        }

        file = io.BytesIO()
        pickle.dump(orig_model, file)
        file.seek(0)

        origin_data_model_pickle, _ = DataModelPickle.objects.get_or_create(
            name=DataModelKeys.ORIGINAL_MODEL,
            user=request.user
        )
        origin_data_model_pickle.pickle = file.getvalue()
        origin_data_model_pickle.save()

        return JsonResponse(data=content)


class CalibrateDataView(LoginRequiredMixin, View):

    def post(self, request):
        years = request.session.get("years")
        xlsx_file = request.FILES['calibrate-data-file']
        df = pd.read_excel(xlsx_file)

        model_pickle = DataModelPickle.objects.get(user=request.user, name=DataModelKeys.ORIGINAL_MODEL)
        orig_model_bytes = model_pickle.pickle

        file = io.BytesIO(orig_model_bytes)
        orig_model = pickle.load(file)

        calibrated_model, image_bytes, text_to_report = orig_model.calibrate(years, df.values, print_out=True)

        file = io.BytesIO()
        pickle.dump(calibrated_model, file)
        file.seek(0)

        calibrated_db_object, _ = DataModelPickle.objects.get_or_create(
            name=DataModelKeys.CALIBRATE_MODEL,
            user=request.user,
        )
        calibrated_db_object.pickle = file.getvalue()
        calibrated_db_object.save()

        base64_image = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
        content = {
            "image": base64_image,
            "text": text_to_report
        }

        return JsonResponse(data=content)


class ReportView(LoginRequiredMixin, View):

    def get(self, request):

        calibrated_pickle = DataModelPickle.objects.get(user=request.user, name=DataModelKeys.CALIBRATE_MODEL)
        orig_model_bytes = calibrated_pickle.pickle

        file = io.BytesIO(orig_model_bytes)
        calibrated_model = pickle.load(file)

        image_bytes = calibrated_model.report()

        pdf = ReportsCollection.prepare_report(image_bytes)
        response = HttpResponse(base64.b64encode(pdf), content_type='application/pdf')
        return response


class DataTemplateDownloadViewBase(LoginRequiredMixin, View):
    template_name = ""

    def get(self, request):
        file_path = os.path.join(settings.DOWNLOAD_FILES_ROOT, self.template_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fh:
                response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
                response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
                return response
        raise Http404


class OriginalTemplateDownloadView(DataTemplateDownloadViewBase):
    template_name = "template.xlsx"


class CalibrateTemplateDownloadView(DataTemplateDownloadViewBase):
    template_name = "calibration_template.xlsx"

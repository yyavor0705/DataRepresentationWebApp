$(document).ready(function () {
        $(".custom-file-input").on("change", function () {
            let fileName = $(this).val().split("\\").pop();
            $(this).siblings(".custom-file-label").addClass("selected").html(fileName);
        });
        $("#original-data-form").submit(Run);
        $("#calibrate-data-form").submit(getCalibratePlot);
        $(".plot-image").click(plotImageClick);
        sampleNumberSliderListen();
        $("#report-button").click(createReportPDF);
    }
);

function setBtnLoading(btn) {
    btn.attr("disabled", true);
    btn.append(spinEml);
}

function stopBtnLoading(btn) {
    btn.attr("disabled", false);
    btn.children("span").remove();
}

let spinEml = $("<span class=\"spinner-border spinner-border-sm\" role=\"status\" aria-hidden=\"true\"></span>");

function Run(event) {
    event.preventDefault();
    let postprocBtn = $('#postproc-btn');
    postprocBtn.attr("disabled", true);
    let url = $(this).attr('action');
    let method = $(this).attr('method');
    formRequestBase(url, this, method, function (data) {
        postprocBtn.click(PostProc);
        postprocBtn.removeAttr("disabled");
    })
}

function PostProc(event) {
    let postProcBtn = $(this);
    setBtnLoading(postProcBtn);
    event.preventDefault();
    let url = $(this).attr('data-url');
    $.ajax({
        type: "get",
        url: url,
        success: function (data) {
            let plotImg = $('#orig-plot');
            plotImg.attr("src", "data:image/png;base64," + data.image);
            let textHolder = $('#orig-plot-text');
            textHolder.empty();
            textHolder.append("<pre>" + data.text.join("<br />") + "</pre>");
            plotImg.parent().removeClass("col-md-7");
        }
    }).done(function () {
        stopBtnLoading(postProcBtn);
    });

}

function getCalibratePlot(event) {
    event.preventDefault();
    let url = $(this).attr('action');
    let method = $(this).attr('method');
    formRequestBase(url, this, method, function (data) {
        $('#calibrated-plot').attr("src", "data:image/png;base64," + data.image);
        let textHolder = $('#calibrate-plot-text');
        textHolder.empty();
        textHolder.append("<pre>" + data.text.join("<br />") + "</pre>");
    })
}

function formRequestBase(url, form, method, callback) {
    let form_data = {};
    if (form) {
        form_data = new FormData($(form)[0]);
    }
    $.ajax({
        type: method,
        url: url,
        enctype: 'multipart/form-data',
        data: form_data,
        processData: false,
        contentType: false,
        cache: false,
        timeout: 600000,
        success: function (data) {
            callback(data)
        }
    });
}

function plotImageClick(event) {
    let modalDialog = $("#plot-modal");
    let modalPlotImage = $("#modal-plot-image");
    let selectedImageSrc = $(this).attr("src");
    $(modalPlotImage).attr("src", selectedImageSrc);
    modalDialog.modal();

    let elem = document.getElementById('modal-plot-image');
    let panzoom = Panzoom(elem, {
        canvas: true,
        minScale: 1,
        step: 0.5,
        cursor: 'all-scroll',
    });
    elem.addEventListener('wheel', panzoom.zoomWithWheel);

    modalDialog.on('hide.bs.modal', function () {
        panzoom.reset();
    })
}

function sampleNumberSliderListen() {
    let yearInput = $("#year-input");
    let valueDisplay = $("#year-input-display");
    valueDisplay.text(yearInput.val());
    yearInput.on("input", function (e) {
        valueDisplay.text(yearInput.val());
    })
}

function createReportPDF(e) {
    let reportBtn = $(this);
    setBtnLoading(reportBtn);
    let holder = $("#pdf-holder");
    let url = $(this).attr('data-url');
    $.ajax({
        type: "get",
        url: url,
        success: function (data) {
            let dataUrl = 'data:application/pdf;base64, ' + data;
            holder.attr('src', dataUrl);
            $("#pdf-modal").modal();
        }
    }).done(function () {
        stopBtnLoading(reportBtn);
    });

}


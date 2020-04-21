$(document).ready(function () {
        $(".custom-file-input").on("change", function () {
            let fileName = $(this).val().split("\\").pop();
            $(this).siblings(".custom-file-label").addClass("selected").html(fileName);
        });
        $("#original-data-form").submit(getOrigPlot);
        $("#calibrate-data-form").submit(getCalibratePlot);
        $(".plot-image").click(plotImageClick);
        $("button[data-plot-id]").click(formLessButtonPress);
        sampleNumberSliderListen();

    }
);

function getOrigPlot(event) {
    event.preventDefault();
    let url = $(this).attr('action');
    let method = $(this).attr('method');
    getPlotBase(url, this, method, function (data) {
        $('#orig-plot').attr("src", "data:image/png;base64," + data);
    })
}

function getCalibratePlot(event) {
    event.preventDefault();
    let url = $(this).attr('action');
    let method = $(this).attr('method');
    getPlotBase(url, this, method, function (data) {
        $('#calibrated-plot').attr("src", "data:image/png;base64," + data);
    })
}

function getPlotBase(url, form, method, callback) {
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

function formLessButtonPress(event) {
    let plotImage = $($(this).attr("data-plot-id"));
    let url = $(this).attr("data-url");
    console.log(url);
    getPlotBase(url, null, "get", function(data){
        console.log(data);
        plotImage.attr("src", "data:image/png;base64," + data);
    })
}

function sampleNumberSliderListen(){
    let sampleInput = $("#sample-number-input");
    let valueDisplay = $("#sample-number-display");
    valueDisplay.text(sampleInput.val());
    sampleInput.on("input", function(e) {
        valueDisplay.text(sampleInput.val());
    })
}



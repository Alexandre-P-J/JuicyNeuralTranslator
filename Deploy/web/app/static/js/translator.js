let translation_timer = null;
let translation_uuid_timer = null;
let file_uuid_timer = null;
let last_translation = null;
const translation_req_delay = 500;
const text_uuid_req_delay = 300;
const file_uuid_req_delay = 5000;

let lang_req = 0;
let trans_req = 0;

let translation_uuid = "";
let file_uuid = "";


// Ensure the correct state
window.addEventListener("load", function () {
    $("#correction-button").hide();
    $("#out-text").prop("disabled", true);
    $("#out-text").val("");
    $("#in-text").val("");
    update_target_languages();
});

$(function () {
    $("#input-method-text").click(function () {
        $("#input-method-text").prop("disabled", true);
        $("#input-method-files").prop("disabled", false);
        $("#in-file").hide();
        $("#out-file").hide();
        $("#in-text").css('display', 'flex');
        $("#out-text").css('display', 'flex');
    });
});

$(function () {
    $("#input-method-files").click(function () {
        $("#input-method-text").prop("disabled", false);
        $("#input-method-files").prop("disabled", true);
        $("#in-text").hide();
        $("#out-text").hide();
        $("#in-file").css('display', 'flex');
        $("#out-file").css('display', 'flex');
        clearTimeout(translation_timer);
        $("#in-text").val("");
        $("#out-text").val("");
    });
});

//
// LEFT SPLIT MANAGEMENT
//
$(function () {
    $("#in-text").on("input", function () {
        clearTimeout(translation_timer);
        $("#correction-button").hide();
        $("#out-text").prop("disabled", true);
        if ($("#in-text").val() === "") {
            $("#out-text").val("");
        } else {
            translation_timer = setTimeout(text_translate_request, translation_req_delay);
        }
    });
});

$(function () {
    $("#in-file-button").click(function () {
        $("#in-file-input").click();
    });
});

$(function () {
    $("#in-file-input").change(function () {
        let file = this.files[0]
        let formData = new FormData();
        formData.append("file", file);
        formData.append("from_lang", $("#lang-selector1").val());
        formData.append("to_lang", $("#lang-selector2").val());
        $.ajax({
            url: '/upload',
            data: formData,
            type: 'POST',
            cache: false,
            contentType: false,
            processData: false,
            success: function (response) {
                $("#in-file-err").val("");
                data = JSON.parse(response);
                if (data["success"]) {
                    file_uuid = data["result"];
                    file_uuid_timer = clearTimeout(file_uuid_timer);
                    file_uuid_timer = setTimeout(file_translate_uuid_request, file_uuid_req_delay);
                    $("#out-file-button").hide();
                    $("#out-file-spinner").show();
                }
                else if (data["filename_err"]) {
                    $("#in-file-err").text("Only .txt, .docx and .pdf files are allowed");
                }
            },
            error: function (jqXHR, error) {
                if (jqXHR.status === 413) {
                    $("#in-file-err").text("File exceeds maximum allowed side of 16MB");
                }
                else {
                    $("#in-file-err").text("Upload error, try again later");
                    console.log(error);
                }
            }
        });
    });
});

$(function () {
    $("#lang-selector1").change(function () {
        clearTimeout(translation_timer);
        $("#correction-button").hide();
        $("#out-text").prop("disabled", true);
        $("#out-text").val("");
        update_target_languages();
    });
});

$(function () {
    $("#lang-selector2").change(function () {
        $("#correction-button").hide();
        $("#out-text").prop("disabled", true);
        if ($("#in-text").val() !== "") {
            clearTimeout(translation_timer);
            text_translate_request();
        }
    });
});

function update_target_languages() {
    $.ajax({
        url: "/languages",
        data: { "from_lang": $("#lang-selector1").val() },
        type: "POST",
        query_id: ++lang_req,
        success: function (response) {
            data = JSON.parse(response);
            if (this.query_id >= lang_req) {
                let langs = data["languages"];
                langs.sort();
                $("#lang-selector2").empty();
                langs.forEach(element => {
                    $("#lang-selector2").append(`<option value="${element}">${element}</option>`)
                });
                if ($("#in-text").val() !== "") {
                    clearTimeout(translation_timer);
                    text_translate_request();
                }
            }
        },
        error: function (error) {
            console.log(error);
        },
    });
}

function text_translate_request() {
    $.ajax({
        url: "/text_translate",
        data: {
            "in-text": $("#in-text").val(),
            "from_lang": $("#lang-selector1").val(), "to_lang": $("#lang-selector2").val()
        },
        type: "POST",
        query_id: ++lang_req,
        success: function (response) {
            data = JSON.parse(response);
            if ((this.query_id >= trans_req) && data["success"]) {
                clearTimeout(translation_uuid_timer);
                translation_uuid = data["result"];
                translation_uuid_timer = setTimeout(text_translate_uuid_request, text_uuid_req_delay);
            }
        },
        error: function (error) {
            console.log(error);
        },
    });
}


function text_translate_uuid_request() {
    $.ajax({
        url: "/text_translate",
        data: {
            "task_id": translation_uuid,
        },
        type: "POST",
        success: function (response) {
            data = JSON.parse(response);
            if (data["ready"] && data["success"]) {
                last_translation = data["result"];
                $("#out-text").val(last_translation);
                $("#out-text").prop("disabled", false);
                $("#correction-button").hide();
            }
            else if (!data["ready"]) {
                clearTimeout(translation_uuid_timer);
                translation_uuid_timer = setTimeout(text_translate_uuid_request, text_uuid_req_delay);
            }
            else {
                console.log("ERROR");
            }
        },
        error: function (error) {
            console.log(error);
        },
    });
}


//
// RIGHT SPLIT MANAGEMENT
//
$(function () {
    $("#out-text").on("input", function () {
        if ($("#out-text").val() != last_translation && !!last_translation) {
            $("#correction-button").show();
        } else {
            $("#correction-button").hide();
        }
    });
});


function file_translate_uuid_request() {
    $.ajax({
        url: "/upload",
        data: {
            "task_id": file_uuid,
        },
        type: "POST",
        success: function (response) {
            data = JSON.parse(response);
            if (data["ready"] && data["success"]) {
                link = data["result"];
                clearTimeout(file_uuid_timer);
                $("#out-file-button").attr("href", link);
                $("#out-file-spinner").hide();
                $("#out-file-button").show();
            }
            else if (!data["ready"]) {
                clearTimeout(file_uuid_timer);
                file_uuid_timer = setTimeout(file_translate_uuid_request, file_uuid_req_delay);
            }
            else {
                console.log("FileError");
            }
        },
        error: function (error) {
            console.log(error);
        },
    });
}


$(function () {
    $("#correction-button").click(function () {
        $("#correction-button").hide();
        correction_request();
    });
});

function correction_request() {
    $.ajax({
        url: "/text_correct",
        data: {
            "in-text": $("#in-text").val(),
            "out-text": last_translation,
            "correct-text": $("#out-text").val(),
            "from_lang": $("#lang-selector1").val(),
            "to_lang": $("#lang-selector2").val()
        },
        type: "POST",
        success: function (_response) {
            last_translation = $("#out-text").val();
        },
        error: function (error) {
            console.log(error);
            $("#correction-button").show();
        },
    });
}
from flask import Flask, request, render_template, make_response, send_from_directory
import os
from helpers import upscaler, makePhoneLike , denoise_and_delay, applyGainCompression, applyGrayscale, colorInvert, voiceEnhancement

app = Flask(__name__, static_folder="static",instance_relative_config=True)
_UPLOADED_ = 0
_FILE_NAME_ = ""
_CONFIGS_ = []

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "static")

@app.route("/")
def hello_world():
    return render_template('project_template.html')


@app.route("/post/", methods=["POST"])
def uploadedVideo():
    global _UPLOADED_, _FILE_NAME_
    if _UPLOADED_ == 1:
        mr = make_response(render_template("project_template.html"), 404)
        mr.headers["res"] = "You've already uploaded a file!"
        return mr
    else:
        request.files["file"].save(request.files["file"].filename)
        _UPLOADED_ = 1
        _FILE_NAME_ = request.files["file"].filename
        print(f"File uploaded: {_FILE_NAME_}")#
    return render_template("project_template.html")



@app.route("/delete/", methods=["DELETE"])
def deletedVideo():
    global _UPLOADED_, _FILE_NAME_
    if _UPLOADED_ == 1:
        _UPLOADED_ = 0
        os.remove(_FILE_NAME_)
        _FILE_NAME_ = ""        
        mr = make_response(render_template("project_template.html"), 204)
        mr.headers["res"] = "file deleted!"
        return mr
    else: 
        return render_template('project_template.html')

@app.route("/configurefilter/", methods=["POST"])
def saveConfiguration():
    global _CONFIGS_
    _CONFIGS_.clear()
    for l in request.get_json():
        _CONFIGS_.append([l["name"], {v["name"]: v["value"] for v in l["props"]}]) ## For easier use!
    return render_template('project_template.html')
    

@app.route("/applyfilter/", methods=["GET"])
def applyFilter():
    global _CONFIGS_, _FILE_NAME_
    if not _CONFIGS_ or not _FILE_NAME_:
        mr = make_response(render_template("project_template.html"), 403)
        mr.headers["res"] = "Missing file or config!"
        return mr   
    else:
        configSize = len(_CONFIGS_)
        for (i, (k, v)) in enumerate(_CONFIGS_):
            prevFileName = _FILE_NAME_
            if i == (configSize - 1):
                _FILE_NAME_ = os.path.join(UPLOAD_FOLDER, f"result.{_FILE_NAME_.split('.')[-1]}")
            else:
                _FILE_NAME_ = os.path.join(UPLOAD_FOLDER, f"temp{i}.{_FILE_NAME_.split('.')[-1]}")
            if k == "phone":
                makePhoneLike(int(v["phoneFilterOrder"]), float(v["phoneSideGain"]), prevFileName, _FILE_NAME_)
            elif k == "upscale":
                upscaler(int(v["upscaleTargetWidth"]), int(v["upscaleTargetHeight"]), prevFileName, _FILE_NAME_)
            elif k == "denoiseDelay":
                denoise_and_delay(_FILE_NAME_ , int (v["noisePower"]) , int(v["delay"]) , int(v["delayGain"]) )
            elif k == "grayscale":
                applyGrayscale(prevFileName,_FILE_NAME_)
            elif k == "gainCompressor":
                applyGainCompression(float(v["gainCompressorThreshold"]), float(v["limiterThreshold"]), prevFileName, _FILE_NAME_)
            elif k == "voiceEnhancement":
                voiceEnhancement(int(v["preemphasisAlpha"]), int(v["highPassFilter"]), prevFileName, _FILE_NAME_)
            elif k == "colorinvert":
                colorInvert(prevFileName, _FILE_NAME_)
            os.remove(prevFileName)


        return render_template("project_template.html")
    
@app.route("/stream/", methods=["GET"])
def stream():
    return send_from_directory(UPLOAD_FOLDER,
                               f"result.{_FILE_NAME_.split('.')[-1]}", as_attachment=True)  
        
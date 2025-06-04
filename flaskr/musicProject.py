from flask import Flask, request, render_template, make_response, send_from_directory
import os

from helpers import upscaler, makePhoneLike , denoise_and_delay, applyGrayscale, colorInvert, voiceEnhancement,  pathMaker, makeCarLike
app = Flask(__name__, static_folder="static", instance_relative_config=True)


_UPLOADED_ = 0


"""

Once a file is uploaded, we save the name of that file in these 2 variables.
_FILE_NAME_ is then changed accordingly as we start to apply our filters

"""
_FILE_NAME_ = ""
_INITIAL_FILE_NAME_ = ""

# Filter configs are stored in here
_CONFIGS_ = []

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "static")


# Project template is served at "/"
@app.route("/")
def landingPage():
    return render_template('project_template.html')


"""
Upload: it's not possible to upload a file if one has already been uploaded. 
"""
@app.route("/post/", methods=["POST"])
def uploadedVideo():
    global _UPLOADED_, _FILE_NAME_, _INITIAL_FILE_NAME_
    if _UPLOADED_ == 1:
        # Return 403 status code to indicate forbidden operation
        mr = make_response(render_template("project_template.html"), 403)
        mr.headers["res"] = "You've already uploaded a file!"
        return mr
    else:
        # saving the file in our folder
        request.files["file"].save(request.files["file"].filename)
        _UPLOADED_ = 1
        # assigning file names! 
        _FILE_NAME_ = request.files["file"].filename
        _INITIAL_FILE_NAME_ = _FILE_NAME_
    return render_template("project_template.html")


"""
Delete: Delete file from folder if there's been an upload! 
"""
@app.route("/delete/", methods=["DELETE"])
def deletedVideo():
    global _UPLOADED_, _FILE_NAME_, _INITIAL_FILE_NAME_
    if _UPLOADED_ == 1:
        _UPLOADED_ = 0
        # removing the original uploaded file
        os.remove(_INITIAL_FILE_NAME_)
        _FILE_NAME_ = ""
        _INITIAL_FILE_NAME_ = ""        
        # 204 to indicate the success of an op but no content to return
        mr = make_response(render_template("project_template.html"), 204)
        mr.headers["res"] = "file deleted!"
        return mr
    else: 
        return render_template('project_template.html')

"""
Configure Filter: appends the desired filters to the CONFIGS list. 
"""
@app.route("/configurefilter/", methods=["POST"])
def saveConfiguration():
    global _CONFIGS_, _FILE_NAME_
    # We want to apply the filters to the initial file
    if _INITIAL_FILE_NAME_:  
        _FILE_NAME_ = _INITIAL_FILE_NAME_
    # clearing all previous filters! 
    _CONFIGS_.clear()
    for l in request.get_json():
        # Coupling between filter name and parameters -- using dict comprehension
        _CONFIGS_.append([l["name"], {v["name"]: v["value"] for v in l["props"]}]) 
    return render_template('project_template.html')
    

@app.route("/applyfilter/", methods=["GET"])
def applyFilter():
    global _CONFIGS_, _FILE_NAME_, _INITIAL_FILE_NAME_
    # should not be able to apply a filter if nothing has been uploaded! 
    if not _CONFIGS_ or not _FILE_NAME_:
        mr = make_response(render_template("project_template.html"), 403)
        mr.headers["res"] = "Missing file or config!"
        return mr   
    else:
        configSize = len(_CONFIGS_)
        # Going through the config couplings and applying each filter
        for (i, (k, v)) in enumerate(_CONFIGS_):
            """ 
            We cannot read and write to a file at the same time, so we're forced 
            to save some intermediary files 
            """  
            prevFileName = _FILE_NAME_
            if i == (configSize - 1):
                # The resulting file is saved as result. + format of file in the static folder
                _FILE_NAME_ = pathMaker("result", _FILE_NAME_)
            else:
                _FILE_NAME_ = pathMaker(f"temp{i}", _FILE_NAME_)
            if k == "phone":
                makePhoneLike(int(v["phoneFilterOrder"]), float(v["phoneSideGain"]), prevFileName, _FILE_NAME_)
            elif k == "upscale":
                upscaler(int(v["upscaleTargetWidth"]), int(v["upscaleTargetHeight"]), prevFileName, _FILE_NAME_)
            elif k == "denoiseDelay":
                denoise_and_delay( float (v["noisePower"]) , int(v["delay"]) , int(v["delayGain"]) , prevFileName , _FILE_NAME_)
            elif k == "grayscale":
                applyGrayscale(prevFileName,_FILE_NAME_)
            elif k == "car":
                makeCarLike(float(v["carSideGain"]), int(v["carFilterOrder"]), prevFileName, _FILE_NAME_)
            elif k == "voiceEnhancement":
                voiceEnhancement(int(v["preemphasisAlpha"]), int(v["highPassFilter"]), prevFileName, _FILE_NAME_)
            elif k == "colorinvert":
                colorInvert(prevFileName, _FILE_NAME_)
            elif k == "frameInterpolate":
                frameInterpolation(float (v["frameInterpolateTargetFps"]) , prevFileName , _FILE_NAME_)
            """ 
            we don't want to remove the original file as we want to be able to
            clean -> config -> apply
            """ 
            if i:
                os.remove(prevFileName)
        return render_template("project_template.html")


"""
Serving the filtered or unfiltered file
"""
@app.route("/stream/", methods=["GET"])
def stream():
    if not _FILE_NAME_:
        return  make_response(render_template("project_template.html"), 404)
    return send_from_directory(UPLOAD_FOLDER,
                               f"result.{_FILE_NAME_.split('.')[-1]}", as_attachment=True)  
        
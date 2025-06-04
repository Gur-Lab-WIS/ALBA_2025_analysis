"""python functions to write and execute imagej macros"""

import os
import subprocess
from pathlib import Path
from time import sleep
from os.path import join as pjoin
from .io import rtsv
from .paths import LS, dblashes, flat_dir, phead
from .utils import headtail


IMAGEJ_PATH = r"C:\TOOLS\ImageJ_Java16\ij.jar"

def make_proc_macro(BC_cropped, save: bool=True, close: bool=True, mac_path = None):
    """
    makes imageJ macro that performs -log(x) and optionally saves filtered image
    args:
        BC_cropped : path to directory containing cropped and transformed scans
        save : whether to save the result
        close : whether to close imagej after the operation
        mac_path : where to save the macro file
    returns:
        path to macro file
    """
    ejfilter = pjoin(precede(BC_cropped), "samplename_Edge_jump_filtery.tif")
    BCname = os.path.split(BC_cropped)[1]
    ejname = os.path.split(ejfilter)[1]
    npath = pjoin(precede(BC_cropped), "BC_cropped_filt")
    if save:
        print(npath)
        if not Path(npath).is_dir():
            os.mkdir(npath)
    macro = rf"""run("Image Sequence...", "open=[{dblashes(LS(BC_cropped, dpath = 1)[0])}] sort");
run("TIFF Virtual Stack...", "open='{dblashes(ejfilter)}'");
selectWindow("{ejname}");
selectWindow("{BCname}");
run("Log", "stack");
run("Multiply...", "value=-1 stack");
imageCalculator("Multiply create stack", "{BCname}", "{ejname}");
selectWindow("Result of {BCname}");"""
    if save:
        macro += "\n"
        macro += f"""run("Image Sequence... ", "format=TIFF name=BC_cropped_filt use save='{dblashes(npath)}'");"""
    if close:
        macro += "\n"
        macro += f"""close();
selectWindow("{BCname}");
close();
selectWindow("{ejname}");
close();"""
    mac_path = mac_path or pjoin(precede(BC_cropped), "proc.ijm")
    with open(mac_path, "w") as mac:
        mac.write(macro)
    return Path(mac_path)


def make_proc_nofilt_macro(BC_cropped, save=0, projection="Median", mac_path = None):
    """
    same as make_proc_macro, but without multiplying by filter
    args:
        BC_cropped : path to directory containing cropped and transformed scans
        save : whether to save the result
        close : whether to close imagej after the operation
        mac_path : where to save the macro file
        projection : method to project the result, optional
    returns:
        path to macro file
    """
    BCname = os.path.split(BC_cropped)[1]
    npath = pjoin(precede(BC_cropped), "BC_cropped_filt")
    if save:
        print(npath)
        if not Path(npath).is_dir():
            os.mkdir(npath)
    macro = rf"""run("Image Sequence...", "open=[{dblashes(LS(BC_cropped, dpath = 1)[0])}] sort");
selectWindow("{BCname}");
run("Log", "stack");
run("Multiply...", "value=-1 stack");"""
    if save:
        macro += "\n"
        macro += f"""run("Image Sequence... ", "format=TIFF name=BC_cropped_filt use save='{dblashes(npath)}'");"""

    if projection:
        macro += f"""\nrun("Z Project...", "projection={projection}");"""
    mac_path = mac_path or pjoin(precede(BC_cropped), "proc.ijm")
    with open(mac_path, "w") as mac:
        mac.write(macro)
    return Path(mac_path)

def run_ijm(mac_path, imagej_path=IMAGEJ_PATH):
    """
    runs imageJ macros with subprocess
    args:
        mac_path : path to macro file
        imagej_path : path to imagej executable
    returns:
        subprocess result object
    """
    return subprocess.Popen(
        ["Java", "-jar", imagej_path, "-macro", mac_path], start_new_session=True
    )

def get_roiz(roidir, mac_path = None):
    """
    input is dir with RoiSet.zip in it, output is z values of all the rois.
    args:
        roidir : path to dir containing RoiSet.zip (output from imagej)
        mac_path : where to save the macro file
    returns:
        values of all the rois in list of lists
    """
    bcfilt = os.path.split(roidir.split("proc")[0])[0]
    bcfilt = pjoin(bcfilt, "cropped", "BC_cropped_filt")
    roisetp = pjoin(roidir, "RoiSet")
    macro = f"""run("Image Sequence...", "open=[{dblashes(LS(bcfilt, dpath = 1)[0])}] sort");
    run("Set Measurements...", "mean redirect=None decimal=3");
    run("ROI Manager...");
    roiManager("Open", "{dblashes(roisetp)}.zip");
    roiManager("Deselect");
    roiManager("Multi Measur
    e");
    saveAs("Results", "{dblashes(roisetp)}.txt");
    run("Close");
    close();
    roiManager("Deselect");
    roiManager("Delete");"""
    mac_path = mac_path or pjoin(roidir, "roi_extract_macro.ijm")
    with open(mac_path, "w") as file:
        file.write(macro)
    run_ijm(mac_path)
    sleep(20)
    file = rtsv(roisetp + ".txt")
    return file

def show_rois(dirr, roidir, mac_path = './see_rois.ijm'):
    """
    input is dir with RoiSet.zip in it, output is none, imagej opens with the rois.
    args:
        dirr : path to top level folder of scan, containing hdf5 files
        roidir : directory above rois to display
        mac_path : where to save the macro file
    """
    filt = dblashes(pjoin(dirr, "cropped", "samplename_Edge_jump_grays.tif"))
    rois = filtl(headtail("", ".roi"), flat_dir(roidir))
    roiset = filtl(lambda x: headtail("RoiSet", ".zip")(phead(x)), flat_dir(roidir))
    roistr = "\n".join([f'roiManager("Open", "{dblashes(x)}");' for x in rois])
    roisetstr = "\n".join([f'roiManager("Open", "{dblashes(x)}");' for x in roiset])
    macro = f"""run("TIFF Virtual Stack...", "open=[{dblashes(filt)}]");
run("ROI Manager...");"""
    macro += roistr
    macro += roisetstr
    with open(mac_path, "w") as file:
        file.write(macro)
    run_ijm(mac_path)
    return mac_path

from numpy.lib.recfunctions import drop_fields
import pandas
from datetime import datetime
from pydl.pydlutils.yanny import yanny
import os
import glob
from astropy.io import fits
import numpy
from astropy.time import Time, TimeDelta
from astropy import units as u


CONFIG_BASE_PATH = "/uufs/chpc.utah.edu/common/home/sdss50/software/git/sdss/sdsscore/main"
DATA_BASE_PATH = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data"


def parseConfSummary(ff):
    print("parsing sum file", ff)
    yf = yanny(ff)
    ft = yf["FIBERMAP"]
    magArr = ft["mag"]
    ft = drop_fields(ft, 'mag')
    df = pandas.DataFrame.from_records(ft)
    df["umag"] = magArr[:, 0]
    df["gmag"] = magArr[:, 1]
    df["rmag"] = magArr[:, 2]
    df["imag"] = magArr[:, 3]
    df["zmag"] = magArr[:, 4]
    df["holeId"] = df["holeId"].str.decode("utf-8")
    df["fiberType"] = df["fiberType"].str.decode("utf-8")
    df["category"] = df["category"].str.decode("utf-8")
    df["firstcarton"] = df["firstcarton"].str.decode("utf-8")

    #add headers
    df["configuration_id"] = int(yf["configuration_id"])
    df["design_id"] = int(yf["design_id"])
    df["field_id"] = int(yf["field_id"])
    df["epoch"] = float(yf["epoch"])
    obsTime = datetime.strptime(yf["obstime"], "%a %b %d %H:%M:%S %Y")
    df["obsTime"] = obsTime
    df["mjd"] = int(yf["MJD"])
    df["observatory"] = yf["observatory"]
    df["raCen"] = float(yf["raCen"])
    df["decCen"] = float(yf["decCen"])
    df["filename"] = ff
    try:
        df["fvc_image_path"] = yf["fvc_image_path"]
    except:
        df["fvc_image_path"] = None

    if "focal_scale" in yf.pairs():
        df["focal_scale"] = float(yf["focal_scale"])
    else:
        df["focal_scale"] = 1

    # import pdb; pdb.set_trace()
    # add an easier flag for sky fibers
    isSky = []
    for c in df.firstcarton.to_numpy():
        if "sky" in c or "skies" in c:
            isSky.append(True)
        else:
            isSky.append(False)

    df["isSky"] = isSky

    # add an easier flag for science fibers
    ot = df.on_target.to_numpy(dtype=bool)
    iv = df.valid.to_numpy(dtype=bool)
    aa = df.assigned.to_numpy(dtype=bool)
    dc = df.decollided.to_numpy(dtype=bool)

    df["activeFiber"] = ot & iv & aa & ~dc

    # last check if this is an "F" file
    if "confSummaryF" in ff:
        df["fvc"] = True
    else:
        df["fvc"] = False

    # figure out if its dithered or not
    if "parent_configuration" in yf.pairs():
        df["parent_configuration"] = int(yf["parent_configuration"])
        # df["is_dithered"] = True
        df["dither_radius"] = float(yf["dither_radius"])
    else:
        df["parent_configuration"] = -999
        # df["is_dithered"] = False
        df["dither_radius"] = -999

    # check for apogee or boss assignments
    _df = df[df.fiberType == "BOSS"]
    if 1 in _df.assigned.to_numpy():
        df["bossAssigned"] = True
    else:
        df["bossAssigned"] = False

    _df = df[df.fiberType == "APOGEE"]
    if 1 in _df.assigned.to_numpy():
        df["apogeeAssigned"] = True
    else:
        df["apogeeAssigned"] = False

    return df


class ConfSumm(object):
    def __init__(self, configID):
        if configID > 10000000:
            self.site = "lco"
        else:
            self.site = "apo"
        self.configID = configID
        confPath, confFPath = self._getConfPaths()
        assert os.path.exists(confPath)
        assert os.path.exists(confFPath)
        self.confExpect = parseConfSummary(confPath)
        self.confMeas = parseConfSummary(confFPath)
        self.mjd = int(self.confMeas.mjd.to_numpy()[0])

        self._getGimgExps()
        self._getApExps()
        self._getBossExps()

    def _getConfPaths(self):
        confStr = ("%i"%self.configID).zfill(6)
        confSubDir = confStr[:-2] + "XX"
        conf = CONFIG_BASE_PATH + "/%s/summary_files/%s/confSummary-%i.par"%(self.site, confSubDir, self.configID)
        confF = CONFIG_BASE_PATH + "/%s/summary_files/%s/confSummaryF-%i.par"%(self.site, confSubDir, self.configID)
        return conf, confF

    def _getGimgExps(self):
        # only look from gcam3
        if self.site == "apo":
            gcamNum = "gfa3n"
        else:
            gcamNum = "gfa3s"

        gimgGlob = DATA_BASE_PATH + "/gcam/%s/%i/proc-gimg-%s-*.fits"%(self.site,self.mjd,gcamNum)
        gimgFiles = sorted(glob.glob(gimgGlob))
        self.gimgNum = []
        self.gimgFile = []
        self.gimgExpStart = []
        self.gimgExpEnd = []
        self.gimgExpTime = []
        for file in gimgFiles:
            ff = fits.open(file)
            if ff[1].header["CONFIGID"] == self.configID:
                gimgNum = int(file.split("-")[-1].split(".fits")[0])
                expStart = Time(ff[1].header["DATE-OBS"], format="iso", scale="tai")
                expTime = ff[1].header["EXPTIME"]
                expEnd = expStart + TimeDelta(expTime*u.s)

                self.gimgNum.append(gimgNum)
                self.gimgFile.append(file)
                self.gimgExpStart.append(expStart)
                self.gimgExpTime.append(expTime)
                self.gimgExpEnd.append(expEnd)

    def _getApExps(self):
        # find all apogee exposures with this configid
        apGlob = DATA_BASE_PATH + "/apogee/%s/%i/apR-a*.apz"%(self.site,self.mjd)
        apFiles = sorted(glob.glob(apGlob))

        self.apNum = []
        self.apFile = []
        self.apExpStart = []
        self.apExpEnd = []
        self.apExpTime = []
        self.apDitherFile = []
        for file in apFiles:
            ff = fits.open(file)
            if ff[1].header["IMAGETYP"] != "Object":
                continue
            if ff[1].header["CONFIGID"] == self.configID:
                apNum = int(file.split("-")[-1].split(".apz")[0])
                dateOBS = ff[1].header["DATE-OBS"].replace("T", " ")
                expStart = Time(dateOBS, format="iso", scale="tai")
                expTime = ff[1].header["EXPTIME"]
                expEnd = expStart + TimeDelta(expTime*u.s)

                dithGlob = DATA_BASE_PATH + "/apogee/quickred/%s/%i/dither/ditherAPOGEE-%i-*.fits"%(self.site, self.mjd, apNum)
                dithFile = glob.glob(dithGlob)
                if len(dithFile) == 1:
                    self.apNum.append(apNum)
                    self.apFile.append(file)
                    self.apExpStart.append(expStart)
                    self.apExpTime.append(expTime)
                    self.apExpEnd.append(expEnd)
                    self.apDitherFile.append(dithFile[0])


    def _getBossExps(self):

        bossGlob = DATA_BASE_PATH + "/boss/spectro/%s/%i/sdR*.fit.gz"%(self.site,self.mjd)
        bossFiles = sorted(glob.glob(bossGlob))

        self.bossNum = []
        self.bossFile = []
        self.bossColor = []
        self.bossExpStart = []
        self.bossExpEnd = []
        self.bossExpTime = []
        self.bossDitherFile = []

        for file in bossFiles:
            ff = fits.open(file)
            if ff[0].header["FLAVOR"] != "science":
                continue
            if ff[0].header["CONFID"] == self.configID:
                bossNum = int(file.split("-")[-1].split(".fit.gz")[0])
                bossNumPad = ("%i"%bossNum).zfill(8)
                dateOBS = ff[0].header["DATE-OBS"].replace("T", " ")
                expStart = Time(dateOBS, format="iso", scale="tai")
                expTime = ff[0].header["EXPTIME"]
                expEnd = expStart + TimeDelta(expTime*u.s)
                if "sdR-b1-" in file:
                    bossColor = "blue"
                    bossColorStr = "b1"
                else:
                    bossColor = "red"
                    bossColorStr = "r1"


                dithGlob = DATA_BASE_PATH + "/boss/sos/%s/%i/dither/ditherBOSS-%s-%s-*.fits"%(self.site, self.mjd, bossNumPad, bossColorStr)
                dithFile = glob.glob(dithGlob)
                if len(dithFile) == 1:
                    self.bossNum.append(bossNum)
                    self.bossFile.append(file)
                    self.bossColor.append(bossColor)
                    self.bossExpStart.append(expStart)
                    self.bossExpEnd.append(expEnd)
                    self.bossExpTime.append(expTime)
                    self.bossDitherFile.append(dithFile[0])




if __name__ == "__main__":

    confLCO = 10000305
    confAPO1 = 7917
    confAPO2 = 5241 # apogee targets, https://data.sdss5.org/sas/sdsswork/sandbox/commiss/dither.html
    confAPO3 = 5168 # boss targets, https://data.sdss5.org/sas/sdsswork/sandbox/commiss/dither.html

    lco = ConfSumm(confLCO)
    apo1 = ConfSumm(confAPO1)
    apo2 = ConfSumm(confAPO2)
    apo3 = ConfSumm(confAPO3)

    import pdb; pdb.set_trace()




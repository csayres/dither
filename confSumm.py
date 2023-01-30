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
from coordio.utils import fitsTableToPandas, wokxy2radec, radec2wokxy
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from functools import partial

from procGimg import GuideBundle

CONFIG_BASE_PATH = "/uufs/chpc.utah.edu/common/home/sdss50/software/git/sdss/sdsscore/main"
DATA_BASE_PATH = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data"

def processGuideBundle(imageNum, site, mjd, fitPointing):
    gb = GuideBundle(site, mjd, imageNum, fitPointing)

    matches = gb.matches.copy()
    matches["fluxRatio"] = matches.fluxNorm_meas / matches.fluxNorm_expect
    matches["raFit"] = gb.raCenFit
    matches["decFit"] = gb.decCenFit
    matches["paFit"] = gb.paFit
    matches["scaleFit"] = gb.scaleFit

    return matches

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
    df["pa"] = float(yf["pa"])
    df["filename"] = ff
    coord_epoch = df.coord_epoch.to_numpy()
    _tt = Time(coord_epoch, format="jyear")
    df["coord_epoch_jd"] = _tt.jd
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

    df["activeFiber"] = ot & iv & aa

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


class SciExp(object):
    def __init__(
        self, site, fiberType, mjd, sciImgNum, expStart, expTime,
        gimgNums, ditherFile, confMeas,
        quick=False, fitPointing=False
    ):
        if quick:
            # use only 3 random guide images
            gimgNums = numpy.random.choice(gimgNums, size=3)
        self.site = site.lower()
        self.fiberType = fiberType.lower()
        self.mjd = mjd
        self.sciImgNum = sciImgNum
        self.gimgNums = gimgNums
        self.confMeas = confMeas.copy()

        _processGuideBundle = partial(processGuideBundle, mjd=mjd, site=site, fitPointing=fitPointing)
        p = Pool(25)
        matches = p.map(_processGuideBundle, gimgNums)
        matches = pandas.concat(matches)
        p.close()

        # self.guideBundles = [GuideBundle(site,mjd,imgNum) for imgNum in gimgNums]

        # pointing parameters from gimgs
        # write gfa matches to csv
        matches["sciImgNum"] = sciImgNum
        matches.to_csv("gfa_%i.csv"%sciImgNum)

        self.raFit = numpy.median(matches.raFit)
        self.decFit = numpy.median(matches.decFit)
        self.paFit = numpy.median(matches.paFit)
        self.scaleFit = numpy.median(matches.scaleFit)
        # optical performance from gimgs
        matches = matches[matches.cpeak > 500]
        matches = matches[matches.cpeak < 50000]
        matches = matches.groupby(["source_id", "gfaID"]).median().reset_index()
        matches["fluxRatio"] = matches.fluxNorm_meas / matches.fluxNorm_expect


        self.dateObs = expStart + TimeDelta(expTime/2*u.s) # midpoint of spectrograph exposure
        self.dateObsJD = self.dateObs.jd

        xWokStar = []
        yWokStar = []
        for ii, row in confMeas.iterrows():
            xWok, yWok, fieldWarn, HA, PA = radec2wokxy(
                [float(row.racat)], [float(row.deccat)], float(row.coord_epoch_jd), self.fiberType.capitalize(),
                self.raFit, self.decFit, self.paFit,
                self.site.upper(), self.dateObsJD, focalScale=self.scaleFit,
                pmra=float(row.pmra), pmdec=float(row.pmdec)
            )
            xWokStar.append(xWok[0])
            yWokStar.append(yWok[0])

        self.confMeas["xWokStar"] = xWokStar
        self.confMeas["yWokStar"] = yWokStar
        self.confMeas["raFit"] = self.raFit
        self.confMeas["decFit"] = self.decFit
        self.confMeas["paFit"] = self.paFit
        self.confMeas["scaleFit"] = self.scaleFit
        self.confMeas["dateObsJD"] = self.dateObsJD
        self.confMeas["sigmaGFA"] = numpy.sqrt(numpy.median(matches.x2/2+matches.y2/2))*13.5/1000 # mm in focal plane
        self.confMeas["fluxRatioGFA"] = numpy.median(matches.fluxRatio)
        self.confMeas["xWokFiber"] = self.confMeas["xwok"] # FVC measured
        self.confMeas["yWokFiber"] = self.confMeas["ywok"] # FVC measured
        self.confMeas["fiberType"] = self.fiberType
        self.confMeas["sciImgNum"] = self.sciImgNum

        ff = fits.open(ditherFile)
        ditherFlux = fitsTableToPandas(ff[1].data)
        ditherFlux["fiberId"] = ditherFlux.fiber
        if self.fiberType == "apogee":
            magCol = "hmag"
        else:
            magCol = "flux_g"

        fluxNorm_expect = 10**(-ditherFlux[magCol]/2.5) # plus a constant zeropoint
        ditherFlux["flux_expect"] = fluxNorm_expect

        self.confMeas = self.confMeas.merge(ditherFlux, on="fiberId").reset_index()
        keepColumns = ["positionerId", "fiberId", "fiberType", "spectroflux", "spectroflux_ivar", magCol, "flux_expect", "xWokStar", "yWokStar", "xWokFiber", "yWokFiber"]
        keepColumns += ["raFit", "decFit", "paFit", "scaleFit", "dateObsJD", "sigmaGFA", "fluxRatioGFA"]
        keepColumns += ["sciImgNum", "alpha", "beta", "dateObsJD"]
        self.confMeas = self.confMeas[keepColumns]

        # print("sigmaGFA", self.confMeas.sigmaGFA.to_numpy()[0])



class Configuration(object):
    def __init__(self, configID, color="red", fitPointing=True):
        """color ignored for apogee, corresponds to red or blue boss chip
        """
        assert color in ["blue", "red"]

        if configID > 10000000:
            self.site = "lco"
        else:
            self.site = "apo"
        self.fitPointing = fitPointing
        self.configID = configID
        self.color = color.lower()
        confPath, confFPath = self._getConfPaths()
        assert os.path.exists(confPath)
        assert os.path.exists(confFPath)
        self.confMeas = parseConfSummary(confFPath)
        self.mjd = int(self.confMeas.mjd.to_numpy()[0])
        print("conf lens", len(self.confMeas))

        self.gimgNum = []
        self.gimgFile = []
        self.gimgExpStart = []
        self.gimgExpEnd = []
        self.gimgExpTime = []

        self.apNum = []
        self.apFile = []
        self.apExpStart = []
        self.apExpEnd = []
        self.apExpTime = []
        self.apDitherFile = []

        self.bossNum = []
        self.bossFile = []
        self.bossExpStart = []
        self.bossExpEnd = []
        self.bossExpTime = []
        self.bossDitherFile = []

        self._getGimgExps()
        self._getApExps()
        self._getBossExps()

        if len(self.apNum) > 0:
            self.fiberType = "apogee"
            assert len(self.bossNum) == 0, "found both ap and boss exposures"
            # keep only apogee assigned fibers
            self.confMeasAssigned = self.confMeas[
                (self.confMeas.fiberType == "APOGEE") & \
                (self.confMeas.activeFiber == True) & \
                (self.confMeas.isSky == False) & \
                (self.confMeas.fiberId != -999)
            ]

        elif len(self.bossNum) > 0:
            self.fiberType = "boss"
            assert len(self.apNum) == 0, "found both ap and boss exposures"
            # keep only apogee assigned fibers
            self.confMeasAssigned = self.confMeas[
                (self.confMeas.fiberType == "BOSS") & \
                (self.confMeas.activeFiber == True) & \
                (self.confMeas.isSky == False) & \
                (self.confMeas.fiberId != -999)
            ]

        else:
            raise RuntimeError("No Boss or Ap exposures found")

        if len(self.confMeasAssigned) > 0:
            self.sciExps = self.bundleSciExps() # writes a csv for every exposure
        else:
            print("found no assigned")
            self.sciExps = None


    def bundleSciExps(self):
        if self.fiberType == "apogee":
            attrPre = "ap"
        else:
            attrPre = "boss"
        Num = getattr(self,"%sNum"%attrPre)
        File = getattr(self,"%sFile"%attrPre)
        ExpStart = getattr(self,"%sExpStart"%attrPre)
        ExpEnd = getattr(self,"%sExpEnd"%attrPre)
        ExpTime = getattr(self,"%sExpTime"%attrPre)
        DitherFile = getattr(self,"%sDitherFile"%attrPre)

        sciExps = []
        for n,f,es,ee,et,df in zip(Num,File,ExpStart,ExpEnd,ExpTime,DitherFile):
            # find what gimgs go with this science image
            print("on image n",n)
            gimgExpNums = []
            for gimgNum, gimgStart, gimgEnd in zip(self.gimgNum, self.gimgExpStart, self.gimgExpEnd):
                if gimgStart < es:
                    continue
                if gimgEnd > ee:
                    continue
                gimgExpNums.append(gimgNum)

            sciExp = SciExp(site=self.site, fiberType=self.fiberType,
                             mjd=self.mjd, sciImgNum=n, expStart=es, expTime=et, gimgNums=gimgExpNums,
                             ditherFile=df,
                             confMeas=self.confMeasAssigned, fitPointing=self.fitPointing)
            print("sigmaGFA", sciExp.confMeas.sigmaGFA.to_numpy()[0])
            dframe = sciExp.confMeas.copy()
            dframe["mjd"] = self.mjd
            dframe["configID"] = self.configID
            dframe.to_csv("dither_%i_%i.csv"%(self.configID, n), index=False)
            sciExps.append(sciExp)

        return sciExps


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
        if self.site == "apo":
            apStr = "apR"
        else:
            apStr = "asR"
        apGlob = DATA_BASE_PATH + "/apogee/%s/%i/%s-a*.apz"%(self.site,self.mjd,apStr)
        apFiles = sorted(glob.glob(apGlob))


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

        for file in bossFiles:
            ff = fits.open(file)
            if ff[0].header["FLAVOR"] != "science":
                continue
            if self.color == "blue" and "sdR-r1-" in file:
                continue
            if self.color == "blue" and "sdR-r2-" in file:
                continue
            if self.color == "red" and "sdR-b1-" in file:
                continue
            if self.color == "red" and "sdR-b2-" in file:
                continue
            if ff[0].header["CONFID"] == self.configID:
                bossNum = int(file.split("-")[-1].split(".fit.gz")[0])
                bossNumPad = ("%i"%bossNum).zfill(8)
                dateOBS = ff[0].header["DATE-OBS"].replace("T", " ")
                expStart = Time(dateOBS, format="iso", scale="tai")
                expTime = ff[0].header["EXPTIME"]
                expEnd = expStart + TimeDelta(expTime*u.s)
                if "sdR-b1-" in file:
                    bossColorStr = "b1"
                elif "sdR-r1-" in file:
                    bossColorStr = "r1"
                elif "sdR-b2-" in file:
                    bossColorStr = "b2"
                else:
                    bossColorStr = "r2"


                dithGlob = DATA_BASE_PATH + "/boss/sos/%s/%i/dither/ditherBOSS-%s-%s-*.fits"%(self.site, self.mjd, bossNumPad, bossColorStr)
                dithFile = glob.glob(dithGlob)
                if len(dithFile) == 1:
                    self.bossNum.append(bossNum)
                    self.bossFile.append(file)
                    self.bossExpStart.append(expStart)
                    self.bossExpEnd.append(expEnd)
                    self.bossExpTime.append(expTime)
                    self.bossDitherFile.append(dithFile[0])




if __name__ == "__main__":

    # tstart = time.time()
    # confLCO1 = 10000207 #apogee
    # lco = Configuration(confLCO1)
    # print("one config took", time.time()-tstart)


    # confLCO2 = 10000275 #boss
    confAPO2 = 5951
    confAPO3 = 5316

    # lco1 = Configuration(confLCO1)
    # lco2 = Configuration(confLCO2)

    # tstart = time.time()
    # apo2 = Configuration(confAPO2)
    # print("apo boss one config took", time.time()-tstart)

    tstart = time.time()
    apo3 = Configuration(confAPO3)
    print("apo boss one config took", time.time()-tstart)

    # import pdb; pdb.set_trace()




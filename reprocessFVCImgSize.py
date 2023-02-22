import glob
from astropy.io import fits
from multiprocessing import Pool
import functools

from coordio.transforms import FVCTransformAPO, FVCTransformLCO
from coordio.utils import fitsTableToPandas
from coordio.defaults import calibration

dataPath = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/fcam/"
outPath = "/uufs/chpc.utah.edu/common/home/u0449727/work/"

MJDStart = 59950
MJDEnd = 60000


def getImgList(site):
    _mjdStart = MJDStart
    imgList = []
    for mjd in range(MJDStart,MJDEnd):
        p = dataPath + "%s/%i/proc-fimg*.fits"%(site,mjd)
        files = glob.glob(p)
        if len(files) > 0:
            imgList.extend(files)
    return imgList


def doOne(file, site, nudgeAdjust):
    if site=="apo":
        FVCTransform = FVCTransformAPO
    else:
        FVCTransform = FVCTransformLCO

    imgNum = int(file.split("-")[-1].strip(".fits"))
    mjd = int(file.split("/")[-2])
    pt = calibration.positionerTable.loc[site.upper()].reset_index()
    wc = calibration.wokCoords.loc[site.upper()].reset_index()
    fc = calibration.fiducialCoords.loc[site.upper()].reset_index()
    ff = fits.open(file)
    print("imagesize", ff[1].data.shape)
    ipa = ff[1].header["IPA"]
    pc = fitsTableToPandas(ff["POSANGLES"].data)
    ft = FVCTransform(
        ff[1].data,
        pc,
        ipa,
        positionerTable=pt,
        wokCoords=wc,
        fiducialCoords=fc,
        nudgeAdjust=nudgeAdjust
    )
    ft.extractCentroids()
    print("got centroids", len(ft.centroids))
    ft.fit()
    print("got positioner table")
    print(len(ft.positionerTableMeas))

    ptm = ft.positionerTableMeas
    fcm = ft.fiducialCoordsMeas

    ptm["mjd"] = mjd
    ptm["imgNum"] = imgNum
    ptm["adjusted"] = nudgeAdjust
    ptm["site"] = site

    fcm["mjd"] = mjd
    fcm["imgNum"] = imgNum
    fcm["adjusted"] = nudgeAdjust
    fcm["site"] = site

    fpath = outPath + "fvcResize/" + "ptm-%s-%i-%i-%s.csv"%(site,mjd,imgNum,str(nudgeAdjust))
    ptm.to_csv(fpath)

    fpath = outPath + "fvcResize/" + "fcm-%s-%i-%i-%s.csv"%(site,mjd,imgNum,str(nudgeAdjust))
    fcm.to_csv(fpath)

    ff.close()

for site in ["apo", "lco"]:
    files = sorted(getImgList(site))
    files = files[:3]
    for nudgeAdjust in [True, False]:
        # p = Pool(5)
        _func = functools.partial(doOne, site=site, nudgeAdjust=nudgeAdjust)
        # p.map(_func, files)

        for file in files:
            print("processing file", file)
            _func(file)




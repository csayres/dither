import glob
from astropy.io import fits
from multiprocessing import Pool
import pdb; pdb.set_trace()

from coordio.transforms import FVCTransformAPO, FVCTransformLCO
from coordio.utils import fitsTableToPandas
from coordio.defaults import calibration

dataPath = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/fcam/"
outPath = "/uufs/chpc.utah.edu/common/home/u0449727/work/"

MJDStart = 59900
MJDEnd = 60000


def getImgList(site):
    _mjdStart = MJDStart
    imgList = []
    for mjd in range(MJDStart,MJDEnd):
        p = dataPath + "/lco/%i/proc-fimg*.fits"%mjd
        files = glob.glob(p)
        if len(files) > 0:
            imgList.extend(files)
    return imgList

def doOne(file, site):
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
    ipa = ff[1].header["IPA"]
    pc = fitsTableToPandas(ff["POSANGLES"].data)
    ft = FVCTransform(
        ff[1].data,
        pc,
        ipa,
        positionerTable=pt,
        wokCoords=wc,
        fiducialCoords=fc
    )
    ft.extractCentroids()
    fc.fit()
    print(fc.positionerTableMeas)

for site in ["apo", "lco"]:
    files = sorted(getImgList(site))
    files = files[:10]
    for file in files:
        print("processing file", file)
        doOne(file, site)



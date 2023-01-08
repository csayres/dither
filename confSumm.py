from numpy.lib.recfunctions import drop_fields
import pandas
import datetime
from pydl.pydlutils.yanny import yanny

BASE_PATH = "/uufs/chpc.utah.edu/common/home/sdss50/software/git/sdss/sdsscore/main"


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
    def __init__(self, site, configID):
        self.site = site.lower()
        self.configID = configID



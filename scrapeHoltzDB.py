from commiss import db
import pandas
import os

# confMJD = [ [ 5144, 59704],
#             [ 5146, 59704],
#             [ 5148, 59704],
#             [ 5165, 59705],
#             [ 5167, 59705],
#             [ 5169, 59705],
#             [ 5259, 59710],
#             [ 5286, 59712],
#             [ 5287, 59712],
#             [ 5301, 59713]
# ]
def getAPO():
    configs = [[5316,59714], [5146, 59704], [5167,59705], [5192,59706], [5287,59712], [5300,59713], [5315,59714], [5260,59710], [5288,59712], [5148,59704]]
    configs += [[5169,59705], [5194,59706], [5302,59713], [5301,59713], [5319,59714], [5286,59712], [5144,59704], [5165,59705], [5189,59706], [5259,59710]]

    dfList = []
    for confid, mjd in configs:
        out = db.select(mjd=mjd)
        keep = out["config_id"] == confid
        out = out[keep]
        keep = out["camera"] == "r1"
        out = out[keep]

        dChernoRA = out["cherno_offset_ra"]
        dChernoDec = out["cherno_offset_dec"]
        dRA = out["offset_ra"]
        dDec = out["offset_dec"]
        spectroflux = out["spectroflux"]
        mag = out["mag_r"]
        sciImgNum = out["exp_no"]
        fiberID = out["fiber"]
        alpha = out["alpha"]
        beta = out["beta"]
        xfocal = out["xfocal"]
        yfocal = out["yfocal"]
        ra = out["ra"]
        dec = out["dec"]

        dfList.append(pandas.DataFrame(
            {
                "configurationId": confid,
                "mjd": mjd,
                "fiberID": fiberID,
                "dChernoRA": dChernoRA,
                "dChernoDec": dChernoDec,
                "dRA": dRA,
                "dDec": dDec,
                "spectroflux": spectroflux,
                "mag_r": mag,
                "sciImgNum": sciImgNum,
                "alpha": alpha,
                "beta": beta,
                "xfocal": xfocal,
                "yfocal": yfocal,
                "ra": ra,
                "dec": dec
            }

        ))

    df = pandas.concat(dfList)
    df.to_csv("holtzScrapeAPO.csv", index=False)


def getLCO():
    mjdStart = 59820 - 1
    mjdEnd = 59966
    dfList = []
    while True:
        mjdStart += 1
        if mjdStart > mjdEnd:
            break
        out = db.select(mjd=mjdStart)
        if len(out) == 0:
            continue
        keep = out["observatory"] == "lco"
        out = out[keep]
        print("got", mjdStart)
        dChernoRA = out["cherno_offset_ra"]
        dChernoDec = out["cherno_offset_dec"]
        dRA = out["offset_ra"]
        dDec = out["offset_dec"]
        spectroflux = out["spectroflux"]
        mag = out["mag_r"]
        sciImgNum = out["exp_no"]
        fiberID = out["fiber"]
        alpha = out["alpha"]
        beta = out["beta"]
        xfocal = out["xfocal"]
        yfocal = out["yfocal"]
        ra = out["ra"]
        dec = out["dec"]

        dfList.append(pandas.DataFrame(
            {
                "configurationId": out["config_id"],
                "camera": out["camera"],
                "site": out["observatory"],
                "mjd": mjdStart,
                "fiberID": fiberID,
                "dChernoRA": dChernoRA,
                "dChernoDec": dChernoDec,
                "dRA": dRA,
                "dDec": dDec,
                "spectroflux": spectroflux,
                "mag_r": mag,
                "sciImgNum": sciImgNum,
                "alpha": alpha,
                "beta": beta,
                "xfocal": xfocal,
                "yfocal": yfocal,
                "ra": ra,
                "dec": dec
            }

        ))

    df = pandas.concat(dfList)
    df.to_csv("holtzScrapeLCO.csv", index=False)


def procAllLCO():
    os.nice(10)
    df = pandas.read_csv("holtzScrapeLCO.csv")
    configIDs = list(set(df.configurationId))
    from confSumm import Configuration
    for configID in configIDs:
        Configuration(configID)


if __name__ == "__main__":
    procAllLCO()














import pandas
from peewee import PostgresqlDatabase
import time

db = PostgresqlDatabase("sdss5db", user="sdss_user", host="operations.sdss.org")

def getDitherDesigns():
    db.connect()
    sqltxt = "select distinct obs.visit.visit_pk, visit.observatory, obs.dither.mjd, obs.visit.field_id, obs.exposure.config_id, obs.exposure.design_id from obs.dither join obs.visit on obs.dither.visit_pk=obs.visit.visit_pk join obs.exposure on obs.visit.visit_pk=obs.exposure.visit_pk order by obs.dither.mjd, obs.exposure.config_id"
    header = ["visit_pk", "observatory", "mjd", "field_id", "config_id", "design_id"]
    dd = {}
    for head in header:
        dd[head] = []
    cc = db.execute_sql(sqltxt)
    for row in cc.fetchall():
        for head, data in zip(header, row):
            dd[head].append(data)

    df = pandas.DataFrame(dd)
    db.close()
    return df

def getDitherSolns(visit_pk):
    db.connect()
    header = ["visit_pk", "mjd", "camera", "fiber", "xfocal", "yfocal", "xwok", "ywok", "alpha", "beta", "fit_xfiboff", "fit_yfiboff", "fit_dx_res", "fit_dy_res", "fit_dx_global", "fit_dy_global"]
    strHeader = (", ").join(header)
    sqltxt = "select %s from obs.dither where obs.dither.visit_pk=%i"%(strHeader, visit_pk)
    dd = {}
    for head in header:
        dd[head]=[]
    cc = db.execute_sql(sqltxt)
    for row in cc.fetchall():
        for head, data in zip(header, row):
            dd[head].append(data)
    df = pandas.DataFrame(dd)
    db.close()
    return df

t1 = time.time()
df = getDitherDesigns()
print("took", time.time()-t1)
df = df[df.observatory=="apo"]
df = df[df.mjd > 60226]
df = df[df.design_id != 35896]
visit_pks = df["visit_pk"].to_numpy()
t1 = time.time()
dfList = [getDitherSolns(x) for x in visit_pks]
print("took", time.time()-t1)
df2 = pandas.concat(dfList)
df2 = df2.merge(df, on="visit_pk")
df2.to_csv("recentDithers.csv", index=False)
import pdb; pdb.set_trace()

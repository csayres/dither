import matplotlib.pyplot as plt
import pandas
import seaborn as sns
import numpy
from skimage.transform import EuclideanTransform
from coordio.defaults import calibration

gfaCoords = calibration.gfaCoords.reset_index()


def plotMeta():
    df = pandas.read_csv("gimgMeta.csv")
    # sns.histplot(x="nStars", hue="solved", element="step", stat="density", data=df)
    # plt.show()

    # find mjd imgNum with all cameras solving
    _df = df[(df.mjd==59843) & (df.offra==0) & (df.offdec==0)]
    # import pdb; pdb.set_trace()
    _dfg = _df.groupby(["configid", "imgNum"]).prod()
    _dfg = _dfg[_dfg.solved==1]

    configID = 10000205

    _df = df[(df.mjd==59843) & (df.offra==0) & (df.offdec==0) & (df.configid==configID)]


    # mjd 59843 imgNum 22
    import pdb; pdb.set_trace()

def fitNewGFACoords(csvfile, write=True):
    df = pandas.read_csv(csvfile)
    xWok = []
    yWok = []
    ix = []
    iy = []
    jx = []
    jy = []

    for gfaID in range(1,7):
        gfaRow = gfaCoords[gfaCoords.id==gfaID]
        b = gfaRow[["xWok", "yWok"]].to_numpy().squeeze()
        iHat = gfaRow[["ix", "iy"]].to_numpy().squeeze()
        jHat = gfaRow[["jx", "jy"]].to_numpy().squeeze()

        _df=df[df.gfaID==gfaID]
        x = _df.xWok_expect.to_numpy()
        y = _df.yWok_expect.to_numpy()
        dx = _df.dxFit.to_numpy()
        dy = _df.dyFit.to_numpy()

        xFit = x - dx
        yFit = y - dy

        dr = numpy.sqrt(dx**2+dy**2)

        plt.figure(figsize=(5,5))
        plt.quiver(x,y,dx,dy,angles="xy",units="xy",scale=0.05,width=0.1)
        plt.title("GFA %i"%gfaID)

        # plt.figure(figsize=(5,5))
        # plt.plot(x,y,'.k')
        # plt.title("GFA %i"%gfaID)

        plt.figure()
        plt.hist(dr)
        plt.title("GFA %i"%gfaID)

        xyExpect = numpy.array([x,y]).T - numpy.array([b]*len(x))
        xyFit = numpy.array([xFit,yFit]).T - numpy.array([b]*len(x))

        tf = EuclideanTransform()
        tf.estimate(xyFit,xyExpect)
        print("GFA", gfaID, tf.translation, numpy.degrees(tf.rotation))

        xyFit2 = tf(xyFit)
        dxy2 = xyExpect - xyFit2

        plt.figure(figsize=(5,5))
        plt.quiver(x,y,dxy2[:,0],dxy2[:,1],angles="xy",units="xy",scale=0.05,width=0.1)
        plt.title("fit GFA %i"%gfaID)

        plt.figure()
        plt.hist(numpy.linalg.norm(dxy2,axis=1))
        plt.title("fit GFA %i"%gfaID)

        cosRot = numpy.cos(tf.rotation)
        sinRot = numpy.sin(tf.rotation)

        rotMat = numpy.array([
            [cosRot, -sinRot],
            [sinRot, cosRot]
        ])

        b = b + tf.translation
        iHat = rotMat @ iHat
        jHat = rotMat @ jHat
        xWok.append(b[0])
        yWok.append(b[1])
        ix.append(iHat[0])
        iy.append(iHat[1])
        jx.append(jHat[0])
        jy.append(jHat[1])

    gfaCoords["xWok"] = numpy.array(xWok)
    gfaCoords["yWok"] = numpy.array(yWok)
    gfaCoords["ix"] = numpy.array(ix)
    gfaCoords["iy"] = numpy.array(iy)
    gfaCoords["jx"] = numpy.array(jx)
    gfaCoords["jy"] = numpy.array(jy)

    if write:
        gfaCoords.to_csv("gfaCoords_new.csv")
    plt.show()

if __name__ == "__main__":
    fitNewGFACoords("allSolved_orig.csv")
    fitNewGFACoords("allSolved_new.csv", write=False)

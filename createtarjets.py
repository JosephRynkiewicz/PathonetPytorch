import numpy as np
import json
from imageio import imread, imwrite
import argparse
import glob



parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--inputPath', '-i', required=True)
parser.add_argument("--GaussianSize", '-g',  type=int, default=9)
args=parser.parse_args()



def get_label(fichier):
    """print image labels"""
    with open(fichier) as fichier_label:
        label = json.load(fichier_label)
    return label


def bulles(listecentres, Sizeoutput=(256,256,3), dimbulle=9,sigma2=4):
    padding=(dimbulle//2)*2
    output=np.zeros((Sizeoutput[0]+padding,Sizeoutput[1]+padding,Sizeoutput[2]),dtype=np.int)
    x_arr, y_arr = np.mgrid[0:dimbulle, 0:dimbulle]
    center=(padding//2,padding//2)
    patch=np.ceil(np.exp(-1/(2*sigma2)*((x_arr-center[0])**2+(y_arr-center[0])**2))*255)
    #print("patch : ",patch)
    for point in listecentres:
        #if (point['y']==0 | point['x']==0):
        #    print("Zero trouvé!")
        #print("point :",point)       
        output[point['y']:(point['y']+dimbulle),point['x']:(point['x']+dimbulle),point['label_id']-1]=patch
    return output[padding//2:-padding//2,padding//2:-padding//2,:]



def preprocess():
    jsonFiles = glob.glob(args.inputPath+'//'+'*.json')
    print("jsonFiles :", jsonFiles)
    for f in  jsonFiles:
        listecentres=get_label(f)
        lbl=get_label(f)
        label=bulles(lbl,dimbulle=args.GaussianSize)
        name=f[:-4]+'npy'
        np.save(name,label)

preprocess()

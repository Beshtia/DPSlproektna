import tifffile as tiff
import os
import pandas as pd


def read_3band(img_id=None, train=True):
    """
    Citaj sliki od three_band papkata;
    :param img_id: ako sakas da procitas edna slika vnesi img_id na taa slika
    :param train: ako sakas da procitas mnozestvo na sliki togas odredi dali go sakas mnozestvoto za treniranje ili za
    predviduvanje
    :return vraka python recnik od sliki so nivniot img_id ili dokolku img_id ne e None togas vraka edna slika; Slikite
    se numpy.array() so dimenzii mxnx3:
    """
    df = pd.read_csv(os.path.join('train_wkt_v4.csv', 'train_wkt_v4.csv'))
    image_ids = df['ImageId'].unique()
    images = {}
    if not img_id:
        if train:
                for file in os.listdir('three_band'):
                    timg_id = file.strip('.tif')
                    if timg_id in image_ids:
                        images[timg_id] = (tiff.imread(os.path.join('three_band', file)).transpose([1, 2, 0]))
        else:
            for file in os.listdir('three_band'):
                timg_id = file.strip('.tif')
                if timg_id not in image_ids:
                    images[timg_id] = (tiff.imread(os.path.join('three_band', file)).transpose([1, 2, 0]))
    else:
        return tiff.imread(os.path.join('three_band', img_id + '.tif')).transpose([1, 2, 0])

    return images


def read_polygons():
    df = pd.read_csv(os.path.join('train_wkt_v4.csv', 'train_wkt_v4.csv'))
    return df
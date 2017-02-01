import tifffile as tiff
import os
import pandas as pd
import shapely.wkt
import numpy as np


def get_timg_ids():
    """
    :return image_ids na train mnozestvoto:
    """
    df = pd.read_csv(os.path.join('train_wkt_v4.csv', 'train_wkt_v4.csv'))
    image_ids = df['ImageId'].unique()
    return image_ids


def read_3band(img_id=None, train=True):
    """
    Citaj sliki od three_band papkata;
    :param img_id: ako sakas da procitas edna slika vnesi img_id na taa slika
    :param train: ako sakas da procitas mnozestvo na sliki togas odredi dali go sakas mnozestvoto za treniranje ili za
    predviduvanje
    :return vraka pandas DataFrame so kolono=[ImageId, Image] kade vo Image se smesteni slikite so dimenzii mxnx3:
    """
    timg_ids = get_timg_ids()
    image_ids = []
    images = []
    if not img_id:
        if train:
            for file in os.listdir('three_band'):
                timg_id = file.strip('.tif')
                if timg_id in timg_ids:
                    image_ids.append(timg_id)
                    images.append(tiff.imread(os.path.join('three_band', file)).transpose([1, 2, 0]))
        else:
            for file in os.listdir('three_band'):
                timg_id = file.strip('.tif')
                if timg_id not in timg_ids:
                    image_ids.append(timg_id)
                    images.append(tiff.imread(os.path.join('three_band', file)).transpose([1, 2, 0]))
    else:
        return tiff.imread(os.path.join('three_band', img_id + '.tif')).transpose([1, 2, 0])

    df = pd.DataFrame.from_dict({'ImageId': image_ids, 'Image': images})
    df = df[df.columns[::-1]]

    return df


def read_polygons(im_id=None):
    """
    :return Dataframe kako train_wkt_v4.csv:
    """
    df = pd.read_csv(os.path.join('train_wkt_v4.csv', 'train_wkt_v4.csv'))
    df['MultipolygonWKT'] = df['MultipolygonWKT'].apply(shapely.wkt.loads)
    df.set_value(201, 'MultipolygonWKT', df.iloc[201, 2].buffer(0))  # se poprava invalid poligon
    if im_id is not None:
        return df[df['ImageId'].str.strip() == im_id]['MultipolygonWKT'].tolist()
    return df


def read_grid_sizes():
    """
    :return DataFrame kako grid_sizes.csv:
    """
    df = pd.read_csv(os.path.join('grid_sizes.csv', 'grid_sizes.csv'),
                     header=0,
                     names=['ImageId', 'Xmax', 'Ymin'],)
    return df

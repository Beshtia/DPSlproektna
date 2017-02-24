import tifffile as tiff
import os
import pandas as pd
import shapely.wkt
import numpy as np
import re


def get_timg_ids():
    """
    :return image_ids na train mnozestvoto:
    """
    df = pd.read_csv(os.path.join('train_wkt_v4.csv', 'train_wkt_v4.csv'))
    image_ids = df['ImageId'].unique()
    return image_ids


def get_pimg_ids():
    """
    :return image_ids na predict mnozestvoto:
    """
    df = pd.read_csv(os.path.join('train_wkt_v4.csv', 'train_wkt_v4.csv'))
    timage_ids = df['ImageId'].unique()
    pimage_ids = []
    for file in os.listdir('three_band'):
        i_id = file.strip('.tif')
        if i_id not in timage_ids:
            pimage_ids.append(i_id)
    return pimage_ids


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
                    images.append(tiff.imread(os.path.join('three_band', file)).transpose([1, 2, 0]).astype(np.uint16))
        else:
            for file in os.listdir('three_band'):
                timg_id = file.strip('.tif')
                if timg_id not in timg_ids:
                    image_ids.append(timg_id)
                    images.append(tiff.imread(os.path.join('three_band', file)).transpose([1, 2, 0]).astype(np.uint16))
    else:
        return tiff.imread(os.path.join('three_band', img_id + '.tif')).transpose([1, 2, 0]).astype(np.uint16)

    df = pd.DataFrame.from_dict({'ImageId': image_ids, 'Image': images})
    df = df[df.columns[::-1]]

    return df


def read_16band(img_id=None, train=True):
    """
    Citaj sliki od sixteen_band papkata;
    :param img_id: ako sakas da procitas edna slika vnesi img_id na taa slika
    :param train: ako sakas da procitas mnozestvo na sliki togas odredi dali go sakas mnozestvoto za treniranje ili za
    predviduvanje
    :return vraka pandas DataFrame so kolono=[ImageId, Image] kade vo Image se smesteni slikite so dimenzii mxnx3:
    """
    timg_ids = get_timg_ids()
    df = {'ImageId': [], 'A': [], 'M': [], 'P': []}

    if not img_id:
        if train:
            for file in os.listdir('sixteen_band'):
                timg_id = re.sub(r'_.\.tif', '', file)
                channel = re.search('_([A-Z])\.', file).group(1)
                if timg_id in timg_ids:
                    if timg_id not in df['ImageId']:
                        df['ImageId'].append(timg_id)
                    image = tiff.imread(os.path.join('sixteen_band', file)).astype(np.uint16)
                    if channel in ['A', 'M']:
                        image = image.transpose([1, 2, 0])
                    df[channel].append(image)
        else:
            for file in os.listdir('sixteen_band'):
                timg_id = re.sub(r'_.\.tif', '', file)
                channel = re.search('_([A-Z])\.', file).group(1)
                if timg_id not in timg_ids:
                    if timg_id not in df['ImageId']:
                        df['ImageId'].append(timg_id)
                    image = tiff.imread(os.path.join('sixteen_band', file)).astype(np.uint16)
                    if channel in ['A', 'M']:
                        image = image.transpose([1, 2, 0])
                    df[channel].append(image)
    else:
        image_A = tiff.imread(os.path.join('sixteen_band', img_id + '_A.tif')).transpose([1, 2, 0]).astype(np.uint16)
        image_M = tiff.imread(os.path.join('sixteen_band', img_id + '_M.tif')).transpose([1, 2, 0]).astype(np.uint16)
        image_P = tiff.imread(os.path.join('sixteen_band', img_id + '_P.tif')).astype(np.uint16)
        df['ImageId'] = img_id
        df['A'].append(image_A)
        df['M'].append(image_M)
        df['P'].append(image_P)

    df = pd.DataFrame.from_dict(df)
    columns = df.columns
    df = df[np.roll(columns, -1)]

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

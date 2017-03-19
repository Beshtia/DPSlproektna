from reading import *
import numpy as np
import shapely.affinity
import cv2
import matplotlib.pyplot as plt
import random
import tifffile as tf
from skimage.transform import rotate


IM_SIZE = (3349, 3396)


def get_scalers(grids, im_size=IM_SIZE):
    """
    :param grids: grids_siizes.csv DataFrame
    :param im_size: golemina na slikata
    :return DataFrame kako grids so dopolnitelni koloni za skalirackite faktori x_ i y_:
    """
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    w_ = w*(w/(w + 1))
    h_ = h*(h/(h + 1))
    grids['x_'], grids['y_'] = w_/grids['Xmax'], h_/grids['Ymin']
    return grids


def imadjust(image, lower_percent=2, higher_percent=98):
    """
    Pravi linearno razvlekuvanje(image enhancement) na histogramot na slikata i ja sveduva na 8 biti po kanal so nivoa
    (0 - 255)=(output_min, output_max)
    :param image: Slika so razlicen broj na kanali
    :param lower_percent: Dolen percentil za odbiranje na input_min
    :param higher_percent: Goren percentil za odbiranj na input_max
    :return:
    """
    if len(image.shape) > 2:
        out = np.zeros_like(image)
        n_bands = image.shape[-1]
        for i in range(n_bands):
            a = 0  # output_min
            b = 255  # output_max
            c = np.percentile(image[:, :, i], lower_percent)  # input_min
            d = np.percentile(image[:, :, i], higher_percent)  # input_max
            t = a + (image[:, :, i] - c) * (b - a) / (d - c)
            t.clip(a, b, t)

            out[:, :, i] = t

        return out.astype(np.uint8)
    else:
        out = np.zeros_like(image)
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(image[:, :], lower_percent)  # input_min
        d = np.percentile(image[:, :], higher_percent)  # input_max
        t = a + (image[:, :] - c) * (b - a) / (d - c)
        t.clip(a, b, t)

        out[:, :] = t

        return out.astype(np.uint8)


def polygon_masks_1mask(im_id=None, im_size=IM_SIZE):
    """
    Kreira np.array() so dimenzii=im_size cii elementi pripagjaat na mnozestvoto vrednosti [0, 10] kade sekoja vrednost
    oznacuva klasata na koja pripagja pikselot, pr. (vodena povrsina, zelena povrsina, pat itn...)
    :param im_size: golemina na maskata:
    :param im_id: za selekcija na edna slika samo
    :return vrakja pandas.Dataframe() so koloni ImageId i Mask kade Mask e maskata za slikata so id=ImageId:
    """
    if im_id is None:
        wkt = read_polygons()
        grids = read_grid_sizes()
        grids = grids[grids['ImageId'].isin(wkt['ImageId'])]
        grids = get_scalers(grids, im_size).set_index('ImageId')
        masks = {'ImageId': [], 'Mask': []}

        for img_id in grids.index.tolist():
            x_, y_ = grids.loc[img_id].x_, grids.loc[img_id].y_
            mask = np.zeros(im_size, dtype=np.uint8)
            polys = wkt[wkt.ImageId == img_id]['MultipolygonWKT'].tolist()
            for i, mpoly in enumerate(polys):
                    pom = [shapely.affinity.scale(poly, xfact=x_, yfact=y_, origin=(0, 0, 0)) for poly in mpoly]
                    exteriors = [np.array(poly.exterior.coords).round().astype(np.int32) for poly in pom]
                    interiors = [np.array(pi.coords).round().astype(np.int32) for poly in pom for pi in poly.interiors]
                    cv2.fillPoly(mask, pts=exteriors, color=i+1)
                    cv2.fillPoly(mask, pts=interiors, color=0)
            masks['ImageId'].append(img_id)
            masks['Mask'].append(mask)
    else:
        grids = read_grid_sizes()
        grids = grids[grids['ImageId'].str.strip() == im_id]
        grids = get_scalers(grids, im_size).set_index('ImageId')
        x_, y_ = grids.loc[im_id].x_, grids.loc[im_id].y_
        mask = np.zeros(im_size, dtype=np.uint8)
        polys = read_polygons(im_id)
        for i, mpoly in enumerate(polys):
            pom = [shapely.affinity.scale(poly, xfact=x_, yfact=y_, origin=(0, 0, 0)) for poly in mpoly]
            exteriors = [np.array(poly.exterior.coords).round().astype(np.int32) for poly in pom]
            interiors = [np.array(pi.coords).round().astype(np.int32) for poly in pom for pi in poly.interiors]
            cv2.fillPoly(mask, pts=exteriors, color=i+1)
            cv2.fillPoly(mask, pts=interiors, color=0)
        return mask

    return pd.DataFrame.from_dict(masks)


def polygon_masks_10mask(im_id=None, im_size=IM_SIZE):
    """
    Kreira np.array() so dimenzii=im_size cii elementi pripagjaat na mnozestvoto vrednosti [0, 10] kade sekoja vrednost
    oznacuva klasata na koja pripagja pikselot, pr. (vodena povrsina, zelena povrsina, pat itn...)
    :param im_size: golemina na maskata:
    :param im_id: za selekcija na edna slika samo
    :return vrakja pandas.Dataframe() so koloni ImageId i Mask_i kade Mask_i e maskata za slikata
    so id=ImageId i klasa i, ako se raboti za edna slika odnosno im_id != None togas vrakja recnik od 10 maski
    """
    if im_id is None:
        wkt = read_polygons()
        grids = read_grid_sizes()
        grids = grids[grids['ImageId'].isin(wkt['ImageId'])]
        grids = get_scalers(grids, im_size).set_index('ImageId')
        masks = {'ImageId': []}
        for i in range(10):
            masks.update({'Mask_' + str(i+1): []})

        for img_id in grids.index.tolist():
            x_, y_ = grids.loc[img_id].x_, grids.loc[img_id].y_
            polys = wkt[wkt.ImageId == img_id]['MultipolygonWKT'].tolist()
            for i, mpoly in enumerate(polys):
                mask = np.zeros(im_size, dtype=np.uint8)
                pom = [shapely.affinity.scale(poly, xfact=x_, yfact=y_, origin=(0, 0, 0)) for poly in mpoly]
                exteriors = [np.array(poly.exterior.coords).round().astype(np.int32) for poly in pom]
                interiors = [np.array(pi.coords).round().astype(np.int32) for poly in pom for pi in poly.interiors]
                cv2.fillPoly(mask, pts=exteriors, color=1)
                cv2.fillPoly(mask, pts=interiors, color=0)
                masks['Mask_' + str(i+1)].append(mask)
            masks['ImageId'].append(img_id)
    else:
        grids = read_grid_sizes()
        grids = grids[grids['ImageId'].str.strip() == im_id]
        grids = get_scalers(grids, im_size).set_index('ImageId')
        x_, y_ = grids.loc[im_id].x_, grids.loc[im_id].y_
        masks = {}
        polys = read_polygons(im_id)
        for i, mpoly in enumerate(polys):
            mask = np.zeros(im_size, dtype=np.uint8)
            pom = [shapely.affinity.scale(poly, xfact=x_, yfact=y_, origin=(0, 0, 0)) for poly in mpoly]
            exteriors = [np.array(poly.exterior.coords).round().astype(np.int32) for poly in pom]
            interiors = [np.array(pi.coords).round().astype(np.int32) for poly in pom for pi in poly.interiors]
            cv2.fillPoly(mask, pts=exteriors, color=1)
            cv2.fillPoly(mask, pts=interiors, color=0)
            masks['Mask_' + str(i + 1)] = mask
        return masks

    return pd.DataFrame.from_dict(masks)


#   Deprecated
def get_net_train_data():
    ids = get_timg_ids()
    three_band_list = []
    masks_list = []
    a_list = []
    m_list = []
    for img_id in ids:
        three_band = cv2.resize(read_3band(img_id), (3584, 3584))
        sixteen_band = read_16band(img_id)
        a = cv2.resize(sixteen_band['A'], (224, 224))
        m = cv2.resize(sixteen_band['M'], (896, 896))
        masks = polygon_masks_10mask(img_id, (3584, 3584))  # ovde se menja goleminata na maskata
        pom = []
        for i in range(1, 11):
            pom.append(masks['Mask_' + str(i)])
        masks = np.moveaxis(np.array(pom), 0, -1)

        for i in range(7):
            for j in range(7):
                three_band_list.append(three_band[512*i: 512*i + 512, 512*j: 512*j + 512, :])
                masks_list.append(masks[512*i: 512*i + 512, 512*j: 512*j + 512, :])
                m_list.append(m[128 * i: 128 * i + 128, 128 * j: 128 * j + 128, :])
                a_list.append(a[32 * i: 32 * i + 32, 32 * j: 32 * j + 32, :])

    return np.array(three_band_list), np.array(m_list), np.array(a_list), np.array(masks_list)


def save_train_data(data_size=45e3):
    """
    Generates augmented images from the original train_set and saves the in folder  train_data
    Img format of saving 'ImageID_Channel_i_j_rotation(0, 90, 180, 270)_flip(0, lr, ud)'
    :param data_size: Determines size of the augmented-fragmented train set.
    :return:
    """
    if not os.path.exists('train_data'):
        os.makedirs('train_data')
    ids = get_timg_ids()

    def dont_flip(image):
        return image

    for img_id in ids:
        #   Read channels
        three_band = cv2.resize(read_3band(img_id), (3584, 3584))
        sixteen_band = read_16band(img_id)
        a = cv2.resize(sixteen_band['A'], (224, 224))
        m = cv2.resize(sixteen_band['M'], (896, 896))
        masks = polygon_masks_10mask(img_id, (3584, 3584))  # ovde se menja goleminata na maskata
        pom = []
        #   Stick all masks in one image
        for i in range(1, 11):
            pom.append(masks['Mask_' + str(i)])
        masks = np.moveaxis(np.array(pom), 0, -1)

        #   Find max coordinates for image extraction
        three_band_max = three_band.shape[0] - 512

        for _ in range(int(data_size/300)):
            three_band_i, three_band_j = random.randint(0, three_band_max), random.randint(0, three_band_max)
            for img, channel, i, j, size in [(three_band, '_3band_', three_band_i, three_band_j, 512),
                                             (m, '_m_', int(three_band_i/4), int(three_band_j/4), 128),
                                             (a, '_a_', int(three_band_i/16), int(three_band_j/16), 32),
                                             (masks, '_mask_', three_band_i, three_band_j, 512)]:
                for angle in [0, 90, 180, 270]:
                    for flip, direction in [(dont_flip, '0'), (np.fliplr, 'lr'), (np.flipud, 'ud')]:
                        tiff.imsave(os.path.join('train_data',
                                                 str(img_id) +
                                                 channel +
                                                 str(three_band_i) + '_' +
                                                 str(three_band_j) + '_' +
                                                 str(angle) + '_'
                                                 + direction + '.tiff'),
                                    flip(rotate(img[i:i + size, j:j + size, :], angle)))

from reading import *
import numpy as np
import shapely.affinity
import cv2


IM_SIZE = (3349, 3396)


def get_scalers(grids, im_size= IM_SIZE):
    """

    :param grids: grids_siizes.csv DataFrame
    :param im_size: golemina na slikata
    :return DataFrame so dopolnitelni koloni za skalirackite faktori x_ i y_:
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


def polygon_masks():
    wkt = read_polygons()
    grids = read_grid_sizes()
    grids = grids[grids['ImageId'].isin(wkt['ImageId'])]
    grids = get_scalers(grids).set_index('ImageId')
    masks = {'ImageId': [], 'Mask': []}

    for id in grids.index.tolist():
        x_, y_ = grids.loc[id].x_, grids.loc[id].y_
        mask = np.zeros(IM_SIZE, dtype=np.uint8)
        polys = wkt[wkt.ImageId == id]['MultipolygonWKT'].tolist()
        for mpoly, i in zip(polys, range(len(polys))):
                pom = [shapely.affinity.scale(poly, xfact=x_, yfact=y_, origin=(0, 0, 0)) for poly in mpoly]
                exteriors = [np.array(poly.exterior.coords).round().astype(np.int32) for poly in pom]
                interiors = [np.array(pi.coords).round().astype(np.int32) for poly in pom for pi in poly.interiors]
                cv2.fillPoly(mask, exteriors, i+1)
                cv2.fillPoly(mask, interiors, 0)
        masks['ImageId'].append(id)
        masks['Mask'].append(mask)
    return pd.DataFrame.from_dict(masks)
from reading import *
import numpy as np


def get_scalers(x_max, y_min, im_size=(3349, 3396)):
    """
    Se dobivaat skaliracki faktori za koordinatite na poligonite za da mozat da se konvertiraat vo pikseli
    :param x_max: Se zima od grid_sizes posebno za sekoja slika
    :param y_min: Se zima od grid_sizes posebno za sekoja slika
    :param im_size: golemina na slikata so koja se raboti
    :return:
    """
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    w_ = w*(w/(w + 1))
    h_ = h*(h/(h + 1))
    return w_/x_max, h_/y_min


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

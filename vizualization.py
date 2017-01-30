from matplotlib.patches import PathPatch
from matplotlib.path import Path
from descartes.patch import PolygonPatch
from matplotlib.colors import LinearSegmentedColormap as LSC
import matplotlib.pyplot as plt
from processing import *
from reading import *
from matplotlib.patches import Patch


CLASSES = {0: 'not a class',
           1: 'Bldg',
           2: 'Struct',
           3: 'Road',
           4: 'Track',
           5: 'Trees',
           6: 'Crops',
           7: 'Fast H20',
           8: 'Slow H20',
           9: 'Truck',
           10: 'Car'}
CL = [[1., 1., 1.],
      [1., 0., 0.],
      [0, 0, 0],
      [0.5, 0.5, 0.5],
      [1., 0.6, 0.2],
      [0., 0.39, 0.],
      [1., 1., 0.],
      [0.2, 1., 1.],
      [0., 0., 1.],
      [1., 0.6, 1.],
      [0.4, 0., 0.4]]
ZORDER = {1: 8,
          2: 9,
          3: 3,
          4: 2,
          5: 10,
          6: 1,
          7: 4,
          8: 7,
          9: 6,
          10: 5}
C_MAP = LSC.from_list('ime', CL, 11)


def show_img(im_id, title=True):
    image = read_3band(im_id)
    image = imadjust(image)
    plt.figure(1)
    plt.imshow(image)
    plt.axis('off')
    if title:
        plt.title(im_id)
    return


def show_overlapping_mask(im_id, title=True):
    mask = polygon_masks(im_id)
    plt.figure(2)
    plt.imshow(mask, cmap=C_MAP)
    plt.axis('off')
    if title:
        plt.title(im_id)
    cb = plt.colorbar()
    cb.set_ticks(np.arange(0, 11, 1) + 0.5)
    cb.set_ticklabels(list(CLASSES.values()))
    return


def show_nonoverlapping_mask(im_id, title=True, leg=True):
    polys = read_polygons(im_id)
    fig, ax = plt.subplots(1, 1)
    legend = []
    for i, mpoly in enumerate(polys, 1):
        legend.append(Patch(color=CL[i], label=CLASSES[i]))
        if mpoly.is_empty:
            continue
        for poly in mpoly:
            patch = PolygonPatch(poly, color=CL[i], lw=0, alpha=0.7, zorder=ZORDER[i])
            ax.add_patch(patch)
    ax.relim()
    ax.autoscale_view()
    if title:
        ax.set_title(im_id)
    ax.axis('off')

    if leg:
        plt.legend(handles=legend,
                   bbox_to_anchor=(0.9, 1.),
                   bbox_transform=plt.gcf().transFigure,
                   ncol=5,
                   fontsize='x-small',
                   framealpha=0.2)
    return


def save_imgs(f_name='sliki'):
    img_ids = get_timg_ids()
    newpath = os.path.join('sliki', '3band')
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for img_id in img_ids:
        show_img(img_id, False)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join('f_name', '3band', img_id + '.png'), bbox_inches='tight', pad_inches=0, dpi='figure')
        plt.clf()
        plt.close()

        show_nonoverlapping_mask(img_id, False, False)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join('f_name', '3band', img_id + '_mask' + '.png'), bbox_inches='tight', pad_inches=0, dpi='figure')
        plt.clf()
        plt.close()
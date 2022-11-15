import numpy as np
import matplotlib.pyplot as plt

import skimage.io as skio
import skimage.filters as skfilt
import skimage.exposure as skexp
import skimage.restoration as skres
import skimage.util as skuti
import skimage.feature as skfeat
import skimage.segmentation as skseg
import skimage.measure as skmeas
import skimage.color as skcol
import skimage.morphology as skmorph
from scipy import ndimage

def fill_holes(original, n=1, thresh=0.5):
    """
    Function for filling holes in a mask. Locates pixels of class 1 and evaluates
    how many pixels within a surrounding box are of class 2. If enough
    pixels within the surrounding box are class 2 pixels, the original class 1
    pixel is changed into a class 2 pixel.

    Arguments:
        original (array with bools): Original mask.
        n (int, optional): Parameter determining size of box to be scanned for
                           class 2 pixels. Defaults to 1, giving a box area of
                           3x3 pixels.
        thresh (float): Fraction of box area that needs to be class 2 for
                        original class 1 pixel to switch class.
    Returns:
        filled (array with bools): Original mask with holes filled.
    """
    x = original.shape[0]
    y = original.shape[1]

    box_area = (2*n+1)**2 - 1             # area of box to be checked [pixels]

    filled = original
    class2 = 0                            # number of class 2 pixels in box
    # find class 1 pixels
    for i in range(n, x-n):
        for j in range(n, y-n):
            if not original[i,j]:         # check if pixel is class 1
                # check box elements for class 2 pixels
                for k in range(i-n, i+n):
                    for l in range(j-n, j+n):
                        if original[k,l]:
                            class2 +=1
                # fill hole if enough box pixels are class 2
                if class2/box_area >= thresh:
                    filled[i,j] = True
                class2 = 0

    return filled

def pre_process(image, filter=None, equalize=None, size=5, sigma=3):
    """
    Function for preprocessing an image for segmentation.
    Arguments:
        image (array): Image to preprocess.
        filter (str, optional): Which filter to apply, if any. Defaults to None.
                                Available filters: Gaussian, Blur
        size (int, optional): Parameter for adjusting the Blur filter.
                              Defaults to 5.
        sigma (int, optional): Parameter for adjusting the Gaussian filter.
                               Defaults to 3.
        equalize (str, optional): What histogram equalizing method to use, if
                                  any. Available methods: Regular, CLAHE, None
                                  Defaults to None.
    Returns:
        image (array): Filtered image.
    """

    if equalize is "Regular":
            image = (skexp.equalize_hist(image))**2
    elif equalize is "CLAHE":
            image = skexp.equalize_adapthist(image)

    if filter == "Gaussian":
        image = skfilt.gaussian(image, sigma=sigma, truncate=2)

    elif filter == "Blur":
        image = ndimage.uniform_filter(image, size=size)





    return image

def nuclei_per_cell(cells, nuclei, mn_thresh=400):
    """
    Function for finding nuclei in cells.
    Arguments:
        cells (array): Labeled array of found cells.
        nuclei (array): Labeled array of found nuclei.
        mn_thresh (int, optional): Size threshold for separating nuclei from
                                   micronuclei (in pixels).
                                   Defaults to 400.
    Returns:
        n_nuclei (list): Number of nuclei per found cell.
        n_micronuclei (list): Number of micronuclei per found cell.
    """
    cells_max = np.max(cells)                      # total number of cells found

    """
    Part 1: Assign nuclei to cells based on their location and determine the
    size of each nucleus found.
    """

    nuclei_per_cell = np.zeros(cells_max)          # number of nuclei per cell

    nuclei_in_cell = []                            # ID of nuclei in each cell
    size_nuclei_in_cell = []                       # size of each nucleus

    for i in range(1, cells_max+1):                # loop over all cells found
        cell_loc = np.argwhere(cells==i)           # find indexes with label i

        if cell_loc.any():
            nuclei_id = []
            nuclei_size = []
            for j in range(cell_loc.shape[0]):
                a = nuclei[cell_loc[j][0], cell_loc[j][1]]
                if a != 0 and a not in nuclei_id:
                    nuclei_id.append(a)
                    nuclei_size.append((nuclei==a).sum())
            nuclei_per_cell[i-1] = len(nuclei_id)
            nuclei_in_cell.append(nuclei_id)
            size_nuclei_in_cell.append(nuclei_size)
            nuclei_id = []
            nuclei_size = []
        else:
            # remove leftover objects that have been removed during segmentation
            nuclei_per_cell[i-1] = None

    nuclei_per_cell = nuclei_per_cell[~np.isnan(nuclei_per_cell)]

    """
    Part 2: Find binucleated cells and determine how many micronuclei are
    contained within each binucleated cell.
    """

    n_binucleated = 0
    mn_per_binucleated = []

    n_nuclei = []                               # each element represents a cell
    n_micronuclei =[]

    for i in range(len(nuclei_in_cell)):
        tot_nuclei = len(nuclei_in_cell[i])

        # find number of (normal sized) nuclei in each cell
        n_nuclei.append(sum([True if k > mn_thresh else False for k in
                        size_nuclei_in_cell[i]]))

        # find number of micronuclei in each cell
        n_micronuclei.append(sum([True if k < mn_thresh else False for k in
                             size_nuclei_in_cell[i]]))

    return n_nuclei, n_micronuclei

class segmentation_v1:
    """
    Class for finding and counting objects in an image.
    """
    def __init__(self, image):
        """
        Arguments:
            image (array): Image for which to find blobs.
        """
        self.image = image
        self.mask = None
        self.EDT = None
        self.markers = None
        self.objects = None

        # set default filling parameters
        self.n_bg = 3
        self.n_fg = 3
        self.thresh_bg = 0.5
        self.thresh_fg = 0.5

    def __call__(self, title=None, show=True, cmap="Greens_r"):
        """
        Function for displaying the image.
        Arguments:
            title (str, optional): Title to use. If None, "Original image" is
                                   set as title.
            show (bool, optional): Whether to show image. Defaults to true.
            cmap (str, optional): Which colormap to use. Defaults to "Greens_r".
        """

        plt.imshow(self.image, cmap=cmap)
        plt.axis("off")
        if title is None:
            plt.title("Original image", fontsize=15)
        else:
            plt.title(title, fontsize=15)
        if show:
            plt.show()

    def set_fill_params(self, n_bg=3, n_fg=3, thresh_bg=0.5, thresh_fg=0.5):
        """
        Function for changing the default parameters for filling in the
        segmentation mask.
        Arguments:
            n_bg (int, optional): Parameter determining size of area to be
                                  searched for background pixels around each
                                  foreground pixel.
                                  Defaults to 3.
            n_fg (int, optional): Parameter determining size of area to be
                                  searched for foreground pixels around each
                                  background pixel.
                                  Defaults to 3.
            thresh_bg (float, optional): Surrounding area needed to be
                                         background for a foreground pixel to be
                                         changed into background.
                                         Defaults to 0.5 (half of the
                                         surrounding area).
            thresh_fg (float, optional): Surrounding area needed to be
                                         foreground for a background pixel to be
                                         changed into foreground.
                                         Defaults to 0.5 (half of the
                                         surrounding area).
        """
        self.n_bg = n_bg
        self.n_fg = n_fg
        self.thresh_bg = thresh_bg
        self.thresh_fg = thresh_fg

    def find_blobs(self, fill_mask=True, fill_bg=True, min_dist=35, min_size=50, thresh="Otsu"):
        """
        Function for finding blobs in the image using Otsu thresholding and
        watershed.
        Arguments:
            fill_mask (bool, optional): Whether to fill the mask using
                                        fill_holes.
                                        Defaults to True.
            fill_bg (bool, optional): Whether to fill the background using
                                      fill_holes.
                                      Defaults to True.
            min_dist (int, optional): Minimun distance between seed points.
                                      Defaults to 35.
            min_size (int, optional): Minimum size of objects to find.
                                      Defaults to 50.
        """
        # find threshold
        if thresh == "Otsu":
            thresh = skfilt.threshold_otsu(self.image)
        elif thresh == "Multi":
            thresh = skfilt.threshold_multiotsu(self.image)[-1]

        # separate foreground and background
        self.mask = self.image >= thresh

        # beautify mask
        if fill_bg:
            # find foreground pixels and turn them into background if enough
            # surrounding pixels are background pixels

            # fill foreground pixels in background area
            self.mask = np.invert(fill_holes(np.invert(self.mask), n=self.n_bg, thresh=self.thresh_bg))

        if fill_mask:
            # fill background pixels in foreground area
            self.mask = fill_holes(self.mask, n=self.n_fg, thresh=self.thresh_fg)

        # compute Euclidean distance transform within mask
        self.EDT = ndimage.distance_transform_edt(self.mask)

        # find maxima of EDT
        maxima = skfeat.peak_local_max(self.EDT, min_distance=min_dist)

        # create markers
        maxima_mask = np.zeros(self.EDT.shape, dtype=bool)
        maxima_mask[tuple(maxima.T)] = True
        self.markers = skmeas.label(maxima_mask)

        # perform watershed
        self.objects = skseg.watershed(-self.EDT, markers=self.markers, mask=self.mask)

        # remove objects touching the edge of the image
        self.objects = skseg.clear_border(self.objects)

        # remove objects smaller than min_size
        self.objects = skmorph.remove_small_objects(self.objects, min_size=min_size)

    def get_labels(self):
        """
        Function for retrieving the labeled array with blobs.
        Returns:
            objects (array): Labeled array of found objects.
        Raises:
            AttributeError: If the segmentation has not yet been performed,
                            meaning that no blobs have been found.
        """
        if self.objects is None:
            raise AttributeError("No blobs have been found yet.")
        else:
            return self.objects

    def get_mask(self):
        """
        Function for retrieving the Otsu mask of the image.
        Returns:
            mask (array): Otsu mask.
        Raises:
            AttributeError: If the segmentation has not yet been performed,
                            meaning that no mask has been made.
        """
        if self.mask is None:
            raise AttributeError("No mask has been made.")
        else:
            return self.mask

    def get_EDT(self):
        """
        Function for retrieving the Euclidean distance transform within the
        mask.
        Returns:
            EDT (array): Euclidean distance transform.
        Raises:
            AttributeError: If the segmentation has not yet been performed,
                            meaning that the EDT has not been calculated.
        """
        if self.EDT is None:
            raise AttributeError("The EDT has not been calculated.")
        else:
            return self.EDT

    def get_markers(self):
        """
        Function for retrieving the watershed starting points.
        Returns:
            markers (array): Coordinates of starting points.
        Raises:
            AttributeError: If the segmentation has not yet been performed,
                            meaning that the markers have not yet been found.
        """
        if self.markers is None:
            raise AttributeError("No markers have been made.")
        else:
            return self.markers

    def blobs(self, title="Found objects", show=True, overlay=True):
        """
        Function for plotting the found blobs.
        Arguments:
            title (str, optional): Plot title. Defaults to "Found objects".
            show (bool, optional): Whether to show plot. Defaults to True.
            overlay (bool, optional): Whether to show the blobs over the
                                      original image. Defaults to True.
        Raises:
            AttributeError: If no segmentation has been performed.
        """
        if self.objects is None:
            raise AttributeError("No segmentation has been performed.")

        if overlay:
            plt.imshow(self.image, cmap="binary_r")
            plt.imshow(skcol.label2rgb(self.objects, bg_label=0), alpha=0.6)
        else:
            plt.imshow(skcol.label2rgb(self.objects, bg_label=0))
        plt.axis("off")
        plt.title(title, fontsize=15)
        plt.show()

    def count_blobs(self):
        """
        Function for counting the number of objects found.
        Returns:
            count (int): Number of objects.
        Raises:
            AttributeError: If no segmentation has been performed.
        """
        if self.objects is None:
            raise AttributeError("No segmentation has been performed.")

        # find labeled elements in objects array
        elems = np.unique(self.objects)

        # remove background label (0)
        elems = np.delete(elems, np.where(elems==0))

        # count objects
        count = len(elems)

        return count





if __name__ == "__main__":
    test_image = skio.imread("example_cells.tif")
    test_image = pre_process(test_image, equalize=True)

    find_blobs = segmentation_v1(test_image)

    find_blobs.set_fill_params(n_bg=1, n_fg=1)
    find_blobs.find_blobs()
    find_blobs.blobs()

    print("Number of cells found:",find_blobs.count_blobs())

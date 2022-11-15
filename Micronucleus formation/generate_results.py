import numpy as np
import matplotlib.pyplot as plt
import os

import skimage.io as skio
import skimage.color as skcol

from program_v1 import pre_process, segmentation_v1, nuclei_per_cell


def generate_results(rad_date, cell_line, rad_type, data_folder,
                     fill_param_cells=[3, 3, 0.5, 0.5],
                     fill_param_nuclei=[3, 3, 0.5, 0.5],
                     blob_param_cells=[15, 1000], blob_param_nuclei=[8, 50],
                     equalize=["Regular", None], filters=["Blur", "Blur"],
                     filter_params=[5,5], filename=None, mn_thresh=200, thresh=["Otsu", "Otsu"]):
    """
    Function for generating a .csv-file containing the results from an
    experiment.
    Arguments:
        rad_date (str): Date of irradiation. Should also be the name of the
                        folder containing samples.
        cell_line (str): Cell line used for experiment.
        rad_type (str): Type of radiation used during experiment. Can be
                        "proton" or "xray"
        data_folder (str): Location of data. The data folder should contain
                           a folder for each experiment, which again should
                           contain a folder for each sample.
        fill_param_cells (list, optional): Parameters for filling the mask in
                                           the cell images. Should be on the
                                           form [n_bg, n_fg, thresh_bg,
                                           thresh_fg].
                                           Defaults to [3, 3, 0.5, 0.5].
        fill_param_nuclei (list, optional): Parameters for filling the mask in
                                            the nuclei images. Should be on the
                                            form [n_bg, n_fg, thresh_bg,
                                            thresh_fg].
                                            Defaults to [3, 3, 0.5, 0.5].
        blob_param_cells (list, optional): Parameters for finding cells in the
                                           images. Should be on the form
                                           [min_dist, min_size].
                                           Defaults to [15, 1000].
        blob_param_nuclei (list, optional): Parameters for finding nuclei in the
                                            images. Should be on the form
                                            [min_dist, min_size].
                                            Defaults to [8, 50].
        equalize (list, optional): Parameters for histogram equalization. Should
                                   be on the form [param cells, param nuclei].
                                   Defaults to ["Regular", None]
        filters (list, optional): Filter to use for images. Should be on the form
                                  [filter cells, filter nuclei].
                                  Defaults to ["Blur", "Blur"].
        filter_params (list, optional): Parameters adjusting selected filters.
                                        Should be on the form [parameter for
                                        cells, parameter for nuclei].
                                        Defaults to [5,5].
        filename (str, optional): Name of results file. If None, the irradiation
                                  date is used.
                                  Defaults to None.

    Returns:
        (None)

    Raises:
        ValueError: If the radiation type specified is not proton or xray.

    """

    # create .csv file
    if filename is None:
        file = open(f"{rad_date}_results.csv", "w")
    else:
        file = open(f"{filename}.csv", "w")

    # make head row of .csv file
    if rad_type == "proton":
        file.write("irradiation date,cell line,radiation type,sample_id,dose," +
                   "position,image_nr,nuclei,micronuclei")
    elif rad_type == "xray":
        file.write("irradiation date,cell line,radiation type,sample_id,dose," +
                   "image_nr,nuclei,micronuclei")
    else:
        raise ValueError("Raditation type must be proton or xray.")

    # perform analysis for each sample
    for sample_id in os.listdir(f"./{data_folder}/{rad_date}/"):
        # each folder in rad_date folder corresponds to one sample
        if not os.path.isfile(sample_id):
            print("Currently analysing sample", sample_id)

            # read sample details from folder name
            dose = sample_id.split("gy")[0]
            if rad_type == "proton":
                position = sample_id.split("-")[1].split("p")[1]
                sample_nr = sample_id.split("-")[2]
            elif rad_type == "xray":
                sample_nr = sample_id.split("-")[1]

            # read contents of sample folder (images)
            contents = os.listdir(f"./MN_raw_data/{rad_date}/{sample_id}")

            # ensure that only .tif images are classified as images
            # separate cell and nuclei images
            cell_images = []
            nuclei_images = []
            n_img = 0
            for j in range(len(contents)):
                filename = contents[j]
                if filename.split(".")[1] == "tif":
                    image_nr = int(filename.split("_")[0])
                    channel_nr = int(filename.split("_")[1].split(".")[0].split("ch")[1])

                    # sort cell and nuclei images into separate lists
                    # note: channel 0 should be cells and channel 1 should
                    # be nuclei
                    if channel_nr == 0:
                        cell_images.append([image_nr, filename])
                    elif channel_nr == 1:
                        nuclei_images.append([image_nr, filename])

                    n_img += 1

            for k in range(len(cell_images)):
                for l in range(len(nuclei_images)):
                    # find images with same imageID
                    if cell_images[k][0] == nuclei_images[l][0]:
                        # read images
                        cell_img = skio.imread(f"./{data_folder}/{rad_date}"+
                                   f"/{sample_id}/{cell_images[k][1]}")
                        nuc_img = skio.imread(f"./{data_folder}/{rad_date}"+
                                  f"/{sample_id}/{nuclei_images[l][1]}")

                        # pre-process images
                        cell_img = pre_process(cell_img, filter=filters[0],
                                               size=filter_params[0],
                                               equalize=equalize[0])
                        nuc_img = pre_process(nuc_img, filter=filters[1],
                                              size=filter_params[1],
                                              equalize=equalize[1])

                        # find cells
                        v1_cells = segmentation_v1(cell_img)
                        v1_cells.set_fill_params(n_bg=fill_param_cells[0],
                                                 n_fg=fill_param_cells[1],
                                                 thresh_bg=fill_param_cells[2],
                                                 thresh_fg=fill_param_cells[3])
                        v1_cells.find_blobs(min_dist=blob_param_cells[0],
                                            min_size=blob_param_cells[1], thresh=thresh[0])
                        cells = v1_cells.get_labels()

                        # find nuclei
                        v1_nuclei = segmentation_v1(nuc_img)
                        v1_nuclei.set_fill_params(n_bg=fill_param_nuclei[0],
                                                  n_fg=fill_param_nuclei[1],
                                                  thresh_bg=fill_param_nuclei[2],
                                                  thresh_fg=fill_param_nuclei[3])
                        v1_nuclei.find_blobs(min_dist=blob_param_nuclei[0],
                                             min_size=blob_param_nuclei[1], thresh=thresh[1])
                        nuclei = v1_nuclei.get_labels()

                        # find number of nuclei and micronuclei in each cell
                        n, mn = nuclei_per_cell(cells, nuclei, mn_thresh=mn_thresh)

                        for x, y in zip(n, mn):
                            # print results for each cell to .csv file
                            if rad_type == "proton":
                                file.write(f"\n{rad_date},{cell_line},"+
                                           f"{rad_type},{sample_nr},{dose},{position},"+
                                           f"{cell_images[k][0]},{x},{y}")
                            elif rad_type == "xray":
                                file.write(f"\n{rad_date},{cell_line},"+
                                           f"{rad_type},{sample_nr},{dose},"+
                                           f"{cell_images[k][0]},{x},{y}")
    file.close()


if __name__ == "__main__":
    # example of using the function on the data from 15MAR2022

    # define experiment details
    rad_date = "15MAR2022"                      # date of irradiation
    cell_line = "A549"                          # cell line used for experiment
    rad_type = "proton"                         # type of irradiation

    # parameters determined from 8 first control images
    fill_param_cells = [3, 3, 0.5, 0.5]       # n_bg, n_fg, thresh_bg, thresh_fg
    fill_param_nuclei = [3, 3, 0.5, 0.5]      # n_bg, n_fg, thresh_bg, thresh_fg
    blob_param_cells = [15, 1000]             # min_dist, min_size
    blob_param_nuclei = [8, 50]               # min_dist, min_size

    generate_results(rad_date=rad_date, cell_line=cell_line, rad_type=rad_type,
                     data_folder="MN_raw_data",
                     fill_param_cells=fill_param_cells,
                     fill_param_nuclei=fill_param_nuclei,
                     blob_param_cells=blob_param_cells,
                     blob_param_nuclei=blob_param_nuclei,
                     filename="test_generate_results")

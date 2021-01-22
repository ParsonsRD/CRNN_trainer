from ctapipe.reco.hillas_intersection import HillasIntersection
import copy
from tqdm import tqdm
from ctapipe.image import tailcuts_clean, dilate
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
import numpy as np
import astropy.units as u

__all__ = ["hillas_parameterisation", "reconstruct_event", "perform_reconstruction"]


# First up is our function to perform the image cleaning and Hillas parameterisation
def hillas_parameterisation(image_event, geometry, tel_x, tel_y,
                            picture_thresh=10, boundary_thresh=5,
                            intensity_cut=80, local_distance=3):

    tel_x_dict, tel_y_dict, hillas_parameter_dict = {}, {}, {}

    # Make a copy of the geometry class (we need to change it a bit)
    geometryh = copy.deepcopy(geometry)
    geometryh.pix_x = 1 * geometry.pix_x.value * u.deg
    geometryh.pix_y = -1 * geometry.pix_y.value * u.deg

    t = 0
    # Loop over all our images in this event
    for image in image_event:
        image_shape = image.shape
        if np.sum(image) == 0:
            image[:, :] = 0
            continue

        # Clean the images using split-level cleaning
        mask = tailcuts_clean(geometry, image.ravel(),
                              picture_thresh=picture_thresh,
                              boundary_thresh=boundary_thresh).reshape(image_shape)
        image_clean = np.zeros(image_shape)
        image_clean[mask] = image[mask]

        if np.sum(image_clean) == 0:
            image[:, :] = 0
        else:
            for i in range(4):
                mask = dilate(geometry, mask.ravel()).reshape(image_shape)
            image[np.invert(mask)] = 0

        # Make Hillas parameters and make some simple cuts on them
        try:
            hill = hillas_parameters(geometryh, image_clean.ravel())

            centroid_dist = np.sqrt(hill.x ** 2 + hill.y ** 2)
            # Cut on intensity on distance from camera centre
            if hill.intensity > intensity_cut and centroid_dist < local_distance * u.deg and hill.width > 0 *u.deg:
                tel_x_dict[t] = tel_y[t] * -1
                tel_y_dict[t] = tel_x[t]
                hillas_parameter_dict[t] = hill
            else:
                image[:, :] = 0

        # Skip if we can't make our Hillas parameters
        except HillasParameterizationError:
            t = t
            image[:, :] = 0

        t += 1

    return hillas_parameter_dict, tel_x_dict, tel_y_dict


# This is a more general function to perform the event reconstruction
def reconstruct_event(image_event, geometry, tel_x, tel_y, hillas_intersector,
                      min_tels=2, intensity_cut=80, local_distance=3,
                      picture_thresh=10, boundary_thresh=5):

    # Run our last function to perform Hillas parameterisation
    hillas_parameter_dict, tel_x_dict, tel_y_dict = hillas_parameterisation(image_event, geometry, tel_x, tel_y,
                                                                            intensity_cut=intensity_cut,
                                                                            local_distance=local_distance,
                                                                            picture_thresh=picture_thresh,
                                                                            boundary_thresh=boundary_thresh)

    # If we have enough telescopes perform reconstruction
    if len(hillas_parameter_dict) > min_tels - 1:
        # Perform reconstruction in both ground and nominal system
        nominal_x, nominal_y, _, _ = hillas_intersector.reconstruct_nominal(hillas_parameter_dict)
        ground_x, ground_y, _, _ = hillas_intersector.reconstruct_tilted(hillas_parameter_dict,
                                                                         tel_x_dict, tel_y_dict)
        if not np.isnan(nominal_x):
            hillas_parameters_event = []

            # Loop over all good telescopes and fill up the Hillas parameter output
            for tel in range(len(tel_x)):
                try:
                    hill = hillas_parameter_dict[tel]

                    # Impact distance
                    r = np.sqrt((tel_x[tel].value - (ground_y * -1)) ** 2 + (tel_y[tel].value - ground_x) ** 2)
                    x_cent, y_cent = hill.x.to(u.rad).value, hill.y.to(u.rad).value
                    # Displacement of CoG from reconstructed position
                    disp = np.sqrt((nominal_x - x_cent) ** 2 +
                                   (nominal_y - y_cent) ** 2)

                    # Fill our output that we can use for rejection
                    hillas_parameters_event.append([np.log10(hill.intensity),
                                                    hill.width.to(u.deg).value,
                                                    hill.length.to(u.deg).value,
                                                    np.rad2deg(disp),
                                                    np.rad2deg(np.sqrt(x_cent ** 2 + y_cent ** 2)),
                                                    np.log10(r)])
                except:
                    hillas_parameters_event.append([0., 0., 0., 0., 0., 0.])
            return nominal_x, nominal_y, ground_y * -1, ground_x, hillas_parameters_event
    # Don't return anything if not enough telescopes
    return None, None, None, None, None


# Finally package everything up to perform reconstruction
def perform_reconstruction(images, geometry, tel_x, tel_y,
                           min_tels=2, intensity_cut=80, local_distance=3,
                           picture_thresh=10, boundary_thresh=5):

    # Hillas intersection object
    hillas_intersector = HillasIntersection()

    selected = []
    reconstructed_parameters, hillas_parameters = [], []

    # Loop over all of our events an perform reconstruction
    for event in images:
        try:
            nominal_x, nominal_y, core_x, core_y, hillas = reconstruct_event(event, geometry,
                                                                             tel_x, tel_y,
                                                                             hillas_intersector,
                                                                             min_tels=min_tels,
                                                                             intensity_cut=intensity_cut,
                                                                             local_distance=local_distance,
                                                                             picture_thresh=picture_thresh,
                                                                             boundary_thresh=boundary_thresh)

            if nominal_x is not None and np.rad2deg(np.sqrt(nominal_x ** 2 + nominal_y ** 2)) < 3:
                selected.append(True)
                reconstructed_parameters.append([nominal_x, nominal_y, core_x, core_y])
                hillas_parameters.append(hillas)
            else:
                selected.append(False)
        except ZeroDivisionError:
            selected.append(False)

    return np.array(reconstructed_parameters), np.array(hillas_parameters), np.array(selected)

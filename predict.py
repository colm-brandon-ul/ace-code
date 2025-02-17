from pathlib import Path
from PIL import Image, TiffImagePlugin
from typing import Tuple, Union
import os
import numpy as np
import model
import functools
from scipy.ndimage import gaussian_filter


@functools.lru_cache(maxsize=None)
def apply_rounding(tmin, rounding_scheme):
    if rounding_scheme == 'default':
        return int(tmin)
    elif rounding_scheme == 'nearest':
        return round(tmin)
    elif rounding_scheme == 'stochastic':
        return np.random.choice([np.floor(tmin), np.ceil(tmin)])
    else:
        raise ValueError(f'Invalid rounding scheme: {rounding_scheme}')


def contrast_function_8bit(img, min,max, rounding_scheme = 'default'):
    assert type(img) == Image.Image or type(img) == TiffImagePlugin.TiffImageFile, f'Type {type(img)}, is not a valid input. Please convert image to a PIL.Image and retry'
    assert type(min) == int, 'min must be of type int'
    assert type(max) == int, 'max must be of type float'
    
    output_min, output_max = 0,255
    
    # update this to use stochastic rounding and other options
    def rescale_pixel(px):
        if px <= min:
            return output_min
        elif px > max:
            return output_max
        else:
            return apply_rounding(((px - min) / max) * output_max, rounding_scheme)
    
    vect_func = np.vectorize(rescale_pixel)
    
    return Image.fromarray(np.uint8(vect_func(np.array(img))))


def fastACE(img: Union[Image.Image,TiffImagePlugin.TiffImageFile], 
            remote_model: str,
            filter_sigma: float = 1.0,
            apply_filter: bool = False, 
            rounding_scheme: str = 'default') -> Tuple[Image.Image, Tuple[int,int]]:
    assert type(img) == TiffImagePlugin.TiffImageFile or type(img) == Image.Image, f'Please use a PIL.TiffImagePlugin.TiffImageFile, not {type(img)}'

    # check if the model exists
    if not os.path.exists(Path(__file__).parent / 'models' / 'ace_clf.pkl'):
        model.get_model(remote_model)

    # load the model
    clf = model.load_model()

    # Get the histogram for the image
    im_hist = np.array(img.histogram())
    # Now normalize this histogram so it's agmnostic of the Image size
    im_hist = im_hist / im_hist.sum()
    # pass the normalized histogram through the classifer, reshaping as there's only 1 sample being passed in
    tmin, tmax = clf.predict(im_hist.reshape(1,-1))[0]
    # Using the predicted thresholds, apply the contrast function and return the adjusted Image 
    
    # apply different rounding schemes to the thresholds

    _tmin = apply_rounding(tmin, rounding_scheme)
    _tmax = apply_rounding(tmax, rounding_scheme)


    return contrast_function_8bit(img,_tmin,_tmax), (_tmin, _tmax)
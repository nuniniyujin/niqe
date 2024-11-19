# Modification
* The origianl implementation is from : https://github.com/guptapraful/niqe
* There were big discrepancy between Matlab NIQE score and the original python impelemtation :
* Extracted clean image's mean & cov from matalb (data/clean_image_parameters.mat) and minor modification

| Image                     | Python version | Matlab (NIQE) |
|---------------------------|----------------|---------------|
| bikes.bmp                  | 3.298          | 3.231         |
| bikes_distorted.bmp        | 8.057          | 8.037         |
| parrots.bmp                | 3.778          | 3.789         |
| parrots_distorted.bmp      | 5.607          | 5.613         |


# niqe
NIQE (Completely blind image quality assessment) for grayscale images using skvideo's NIQE.

## for quality of test images
python niqe.py 


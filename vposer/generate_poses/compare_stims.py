import glob
import os
import imagehash
from PIL import Image, ImageDraw
from human_body_prior.tools.omni_tools import makepath


def compare_stims(original_dir: str, reconstructed_dir: str, out_dir: str) -> None:
    """
    Compare original and reconstructed images. First compare the average hash
    difference and print a warning if it varies too much, and second save a new
    image, showing both images side by side.

    :param original_dir: path of original images
    :param reconstructed_dir: path of reconstructed images
    :param str out_dir: path for new (side by side) png images
    """
    orig_files = glob.glob(os.path.join(original_dir, f'*'))

    for f in orig_files:
        f_recon = os.path.basename(f)
        indx = f_recon.find(f'scale')
        if indx != 0:
            f_start = f_recon[:indx]
            f_recon = f_start + f'Ipol_0.png'

        im1 = Image.open(f)
        im2 = Image.open(os.path.join(reconstructed_dir, f_recon))
        hash1 = imagehash.average_hash(im1)
        hash2 = imagehash.average_hash(im2)
        cutoff = 5  # maximum bits that could be different between the hashes.

        if hash1 - hash2 > cutoff:
            print(f'{f_recon} has a hash score ({hash1 - hash2}) higher than 5!')

        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        dst_draw = ImageDraw.Draw(dst)
        dst_draw.text((im1.width / 2 - 30, 20), 'Original', fill='black')
        dst_draw.text((im1.width + im2.width / 2 - 30, 20), 'Reconstructed', fill='black')
        dst_draw.text((20, 20), f'Hash Difference: {hash1 - hash2}', fill='red')

        dst.save(os.path.join(out_dir, f_recon))


if __name__ == "__main__":
    _out_dir = makepath(os.path.join(f'../data/', f'compare_stimuli'))
    old_stim_dir = f'../data/Stim_images/'
    new_stim_dir = f'/home/maxalex/Git/vposer/data/evaluations/VAEparams/png/'
    compare_stims(old_stim_dir, new_stim_dir, _out_dir)

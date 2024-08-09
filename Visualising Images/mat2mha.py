from scipy.io import loadmat
import SimpleITK as sitk
import numpy as np
import os
import glob

# # MAT filename
# Matfname = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Noise Statistics Project\Outputs\VERDICT outputs\No VASC VERDICT\Short Scheme v1\MLP\MATLAB\Rice\T2_10000\sigma_0.025\shV5_20240617\fIC.mat"


folder = r"C:\Users\adam\OneDrive - University College London\UCL PhD\PhD Year 1\Projects\Short VERDICT Project\Outputs\VERDICT outputs\No VASC VERDICT\Original ex905003000\MLP\MATLAB\Rice\T2_10000\sigma_0.025\INN_306"

Matfnames = glob.glob(f'{folder}/*.mat')
print(Matfnames)

for Matfname in Matfnames:
    # Get folder path and img name
    folder, imgname = os.path.split(Matfname) 
    imgname = imgname[0:-4] # Removing .mat

    if imgname == 'fIC':
        clip = True
    else:
        clip = False

    # Get image array
    img = (loadmat(Matfname))[imgname]

    # remove infinities
    img[img == np.inf] = 0

    # Remove nan
    img[np.isnan(img)] = 0

    # Change image orientation
    img = np.moveaxis( img , -1, 0)  

    if clip:
        img[img<0]=0;
        img[img>1]=1;



    # Save as mha
    sitk.WriteImage(sitk.GetImageFromArray(img), f'{folder}/{imgname}.mha')

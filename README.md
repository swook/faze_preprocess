# Preprocessing script for Faze (and ST-ED)

This is a repository for the code used to generate input data for the ICCV 2019 oral paper: *Few-Shot Adaptive Gaze Estimation*. The main code repository can be found at https://github.com/NVlabs/few_shot_gaze

The procedure outlined here is a prerequisite for running the training or evaluation code in the [main repository](https://github.com/NVlabs/few_shot_gaze).

Please cite the FAZE paper when using this pipeline in your research to generate eye-strip inputs:

    @inproceedings{Park2019ICCV,
      author    = {Seonwook Park and Shalini De Mello and Pavlo Molchanov and Umar Iqbal and Otmar Hilliges and Jan Kautz},
      title     = {Few-Shot Adaptive Gaze Estimation},
      year      = {2019},
      booktitle = {International Conference on Computer Vision (ICCV)},
      location  = {Seoul, Korea}
    }

and please consider citing the ST-ED paper as well when using this pipeline to generate 128x128 face images for gaze redirection:

    @inproceedings{Zheng2020NeurIPS,
      author    = {Yufeng Zheng and Seonwook Park and Xucong Zhang and Shalini De Mello and Otmar Hilliges},
      title     = {Self-Learning Transformations for Improving Gaze and Head Redirection},
      year      = {2020},
      booktitle = {Neural Information Processing Systems (NeurIPS)}
    }

 ## Setup

 1. Download all required prerequisite files by running `bash grab_prerequisites.bash`.
 2. Edit the paths defined at the bottom of the main script to point to the actual data paths. More information on where to acquire the original datasets are provided below.
 2. Run the main script using `python3 create_hdf_files_for_faze.py` (for generating eye-strips for FAZE) or `python3 create_hdf_files_for_sted.py` (for generating faces for ST-ED).

 ## Datasets

 ### MPIIFaceGaze

 The [MPIIFaceGaze dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation/) is a subset of the original [MPIIGaze dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/) which includes a privacy-aware release of face images. This subset was used for evaluations of full-face gaze estimation methods.

 When using this dataset, the following works should be cited:

    @inproceedings{zhang2017s,
      title={It’s written all over your face: Full-face appearance-based gaze estimation},
      author={Zhang, Xucong and Sugano, Yusuke and Fritz, Mario and Bulling, Andreas},
      booktitle={Computer Vision and Pattern Recognition Workshops (CVPRW), 2017 IEEE Conference on},
      pages={2299--2308},
      year={2017},
      organization={IEEE}
    }

and the original:

    @inproceedings{zhang15_cvpr,
      Author = {Xucong Zhang and Yusuke Sugano and Mario Fritz and Bulling, Andreas},
      Title = {Appearance-based Gaze Estimation in the Wild},
      Booktitle = {Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      Year = {2015},
      Month = {June}
      Pages = {4511-4520}
    }

_* Please respect the original license of this dataset: CC-BY-NC-SA 4.0_

### GazeCapture

The [GazeCapture dataset](https://gazecapture.csail.mit.edu/) is a large-scale dataset consisting of over 1000 participants. In order to download it, a strict license must be agreed to, so please take care to read and understand it. When using this dataset, please cite the original work:

    @inproceedings{krafka2016eye,
      title={Eye tracking for everyone},
      author={Krafka, Kyle and Khosla, Aditya and Kellnhofer, Petr and Kannan, Harini and Bhandarkar, Suchendra and Matusik, Wojciech and Torralba, Antonio},
      booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages={2176--2184},
      year={2016}
    }

## Contact

For any inquiries regarding this repository, please contact [Seonwook Park](mailto:seon.wook@swook.net).

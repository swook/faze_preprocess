"""
Copyright 2019 ETH Zurich, Seonwook Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os

import cv2 as cv
import h5py
import numpy as np

face_model_3d_coordinates = None

normalized_camera = {
    'focal_length': 1300,
    'distance': 600,
    'size': (256, 64),
}

norm_camera_matrix = np.array(
    [
        [normalized_camera['focal_length'], 0, 0.5*normalized_camera['size'][0]],  # noqa
        [0, normalized_camera['focal_length'], 0.5*normalized_camera['size'][1]],  # noqa
        [0, 0, 1],
    ],
    dtype=np.float64,
)


class Undistorter:

    _map = None
    _previous_parameters = None

    def __call__(self, image, camera_matrix, distortion):
        h, w, _ = image.shape
        all_parameters = np.concatenate([camera_matrix.flatten(),
                                         distortion.flatten(),
                                         [h, w]])
        if (self._previous_parameters is None or
                not np.allclose(all_parameters, self._previous_parameters)):
            self._map = cv.initUndistortRectifyMap(
                camera_matrix, distortion, np.eye(3),
                None, (w, h), cv.CV_32FC1)

        # Apply
        return cv.remap(image, self._map[0], self._map[1], cv.INTER_LINEAR)


undistort = Undistorter()


def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2,
              color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx,
                                   eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv.LINE_AA, tipLength=0.2)
    return image_out


def vector_to_pitchyaw(vectors):
    """Convert given gaze vectors to yaw (theta) and pitch (phi) angles."""
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def data_normalization(dataset_path, group):

    num_entries = next(iter(group.values())).shape[0]

    for i in range(num_entries):
        data_normalization_entry(dataset_path, group, i)


def data_normalization_entry(dataset_path, group, i):

    # Form original camera matrix
    fx, fy, cx, cy = group['camera_parameters'][i, :]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                             dtype=np.float64)

    # Grab image
    image_path = '%s/%s' % (dataset_path,
                            group['file_name'][i].decode('utf-8'))
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    image = undistort(image, camera_matrix,
                      group['distortion_parameters'][i, :])
    image = image[:, :, ::-1]  # BGR to RGB

    # Calculate rotation matrix and euler angles
    rvec = group['head_pose'][i, :3].reshape(3, 1)
    tvec = group['head_pose'][i, 3:].reshape(3, 1)
    rotate_mat, _ = cv.Rodrigues(rvec)

    # Take mean face model landmarks and get transformed 3D positions
    landmarks_3d = np.matmul(rotate_mat, face_model_3d_coordinates.T).T
    landmarks_3d += tvec.T

    # Gaze-origin (g_o) and target (g_t)
    g_o = np.mean(landmarks_3d[10:12, :], axis=0)  # between 2 eyes
    g_o = g_o.reshape(3, 1)
    g_t = group['3d_gaze_target'][i, :].reshape(3, 1)
    g = g_t - g_o
    g /= np.linalg.norm(g)

    # Code below is an adaptation of code by Xucong Zhang
    # https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/revisiting-data-normalization-for-appearance-based-gaze-estimation/

    # actual distance between gaze origin and original camera
    distance = np.linalg.norm(g_o)
    z_scale = normalized_camera['distance'] / distance
    S = np.eye(3, dtype=np.float64)
    S[2, 2] = z_scale

    hRx = rotate_mat[:, 0]
    forward = (g_o / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    # transformation matrix
    W = np.dot(np.dot(norm_camera_matrix, S),
               np.dot(R, np.linalg.inv(camera_matrix)))

    ow, oh = normalized_camera['size']
    patch = cv.warpPerspective(image, W, (ow, oh))  # image normalization

    R = np.asmatrix(R)

    # Correct head pose
    head_mat = R * rotate_mat
    n_h = np.array([np.arcsin(head_mat[1, 2]),
                    np.arctan2(head_mat[0, 2], head_mat[2, 2])])

    # Correct gaze
    n_g = R * g
    n_g /= np.linalg.norm(n_g)
    n_g = vector_to_pitchyaw(-n_g.T).flatten()

    to_visualize = cv.equalizeHist(cv.cvtColor(patch, cv.COLOR_RGB2GRAY))
    to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.24 * oh), n_g,
                             length=80.0, thickness=1)
    to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.5 * oh), n_h,
                             length=80.0, thickness=3, color=(0, 0, 0))
    to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.5 * oh), n_h,
                             length=80.0, thickness=1, color=(255, 255, 255))
    cv.imshow('zhang', to_visualize)
    cv.waitKey(1)

    return {
        'patch': patch,
        'head_pose': n_h,
        'gaze_direction': n_g,
        'rotation_matrix': np.transpose(R),
        'gaze_origin': g_o,
        'gaze_target': g_t,
    }


if __name__ == '__main__':
    # Grab SFM coordinates and store
    face_model_fpath = './sfm_face_coordinates.npy'
    face_model_3d_coordinates = np.load(face_model_fpath)

    # Preprocess some datasets
    output_dir = './outputs/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    datasets = {
        'MPIIGaze': {
            'input-path': '/media/wookie/WookExt4/datasets/MPIIFaceGaze',
            'supplementary': './MPIIFaceGaze_supplementary.h5',
            'output-path': output_dir + '/MPIIGaze.h5',
        },
        # 'GazeCapture': {
        #     'input-path': '/media/wookie/WookExt4/datasets/GazeCapture',
        #     'supplementary': './GazeCapture_supplementary.h5',
        #     'output-path': output_dir + '/GazeCapture.h5',
        # },
    }
    for dataset_name, dataset_spec in datasets.items():
        # Perform the data normalization
        with h5py.File(dataset_spec['supplementary'], 'r') as f:
            for person_id, group in f.items():
                data_normalization(dataset_spec['input-path'], group)

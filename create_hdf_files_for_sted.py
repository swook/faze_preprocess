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
import eos
import h5py
import numpy as np

face_model_3d_coordinates = None

full_face_model_3d_coordinates = None

normalized_camera = {  # Face for ST-ED
    'focal_length': 500,
    'distance': 600,
    'size': (128, 128),
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

    def __call__(self, image, camera_matrix, distortion, is_gazecapture=False):
        h, w, _ = image.shape
        all_parameters = np.concatenate([camera_matrix.flatten(),
                                         distortion.flatten(),
                                         [h, w]])
        if (self._previous_parameters is None
                or len(self._previous_parameters) != len(all_parameters)
                or not np.allclose(all_parameters, self._previous_parameters)):
            print('Distortion map parameters updated.')
            self._map = cv.initUndistortRectifyMap(
                camera_matrix, distortion, R=None,
                newCameraMatrix=camera_matrix if is_gazecapture else None,
                size=(w, h), m1type=cv.CV_32FC1)
            print('fx: %.2f, fy: %.2f, cx: %.2f, cy: %.2f' % (
                    camera_matrix[0, 0], camera_matrix[1, 1],
                    camera_matrix[0, 2], camera_matrix[1, 2]))
            self._previous_parameters = np.copy(all_parameters)

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


def data_normalization(dataset_name, dataset_path, group, output_path):

    # Prepare methods to organize per-entry outputs
    to_write = {}
    def add(key, value):  # noqa
        if key not in to_write:
            to_write[key] = [value]
        else:
            to_write[key].append(value)

    # Iterate through group (person_id)
    num_entries = next(iter(group.values())).shape[0]
    for i in range(num_entries):
        # Perform data normalization
        processed_entry = data_normalization_entry(dataset_name, dataset_path,
                                                   group, i)
        if processed_entry is None:
            continue

        # Gather all of the person's data
        add('pixels', processed_entry['patch'])
        add('labels', np.concatenate([
            processed_entry['normalized_gaze_direction'],
            processed_entry['normalized_head_pose'],
            processed_entry['normalized_gaze_direction_left'],
            processed_entry['normalized_gaze_direction_right'],
        ]))

    if len(to_write) == 0:
        return

    # Cast to numpy arrays
    for key, values in to_write.items():
        to_write[key] = np.asarray(values)
        print('%s: ' % key, to_write[key].shape)

    # Write to HDF
    with h5py.File(output_path,
                   'a' if os.path.isfile(output_path) else 'w') as f:
        if person_id in f:
            del f[person_id]
        group = f.create_group(person_id)
        for key, values in to_write.items():
            group.create_dataset(
                key, data=values,
                chunks=(
                    tuple([1] + list(values.shape[1:]))
                    if isinstance(values, np.ndarray)
                    else None
                ),
                compression='lzf',
            )


def data_normalization_entry(dataset_name, dataset_path, group, i):

    # Form original camera matrix
    fx, fy, cx, cy = group['camera_parameters'][i, :]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                             dtype=np.float64)

    # Grab image
    distortion = group['distortion_parameters'][i, :]
    image_path = '%s/%s' % (dataset_path,
                            group['file_name'][i].decode('utf-8'))
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    image = undistort(image, camera_matrix, distortion,
                      is_gazecapture=(dataset_name == 'GazeCapture'))
    image = image[:, :, ::-1]  # BGR to RGB

    # Calculate rotation matrix and euler angles
    rvec = group['head_pose'][i, :3].reshape(3, 1)
    tvec = group['head_pose'][i, 3:].reshape(3, 1)
    rotate_mat, _ = cv.Rodrigues(rvec)

    # Project 3D face model points, and check if any are beyond image frame
    points_2d = cv.projectPoints(full_face_model_3d_coordinates, rvec, tvec,
                                 camera_matrix, distortion)[0].reshape(-1, 2)
    ih, iw, _ = image.shape
    if np.any(points_2d < 0.0) or np.any(points_2d[:, 0] > iw) \
            or np.any(points_2d[:, 1] > ih):
        tmp_image = np.copy(image[:, :, ::-1])
        for x, y in points_2d:
            cv.drawMarker(tmp_image, (int(x), int(y)), color=[0, 0, 255],
                          markerType=cv.MARKER_CROSS,
                          markerSize=2, thickness=1)
        print('%s skipped. Landmarks outside of frame!' % image_path)
        cv.imshow('failed', tmp_image)
        cv.waitKey(1)
        return

    # Take mean face model landmarks and get transformed 3D positions
    landmarks_3d = np.matmul(rotate_mat, face_model_3d_coordinates.T).T
    landmarks_3d += tvec.T

    # Gaze-origin (g_o) and target (g_t)
    g_o = np.mean(landmarks_3d[10:12, :], axis=0)  # between 2 eyes
    g_o = landmarks_3d[-1, :]  # Face
    g_o = g_o.reshape(3, 1)
    g_t = group['3d_gaze_target'][i, :].reshape(3, 1)
    g = g_t - g_o
    g /= np.linalg.norm(g)

    # Gaze origins and vectors for left/right eyes
    g_l_o = np.mean(landmarks_3d[9:11, :], axis=0).reshape(3, 1)
    g_r_o = np.mean(landmarks_3d[11:13, :], axis=0).reshape(3, 1)
    g_l = g_t - g_l_o
    g_r = g_t - g_r_o
    g_l /= np.linalg.norm(g_l)
    g_r /= np.linalg.norm(g_r)

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
    h = np.array([np.arcsin(rotate_mat[1, 2]),
                  np.arctan2(rotate_mat[0, 2], rotate_mat[2, 2])])
    head_mat = R * rotate_mat
    n_h = np.array([np.arcsin(head_mat[1, 2]),
                    np.arctan2(head_mat[0, 2], head_mat[2, 2])])

    # Correct gaze
    n_g = R * g
    n_g /= np.linalg.norm(n_g)
    n_g = vector_to_pitchyaw(-n_g.T).flatten()

    # Gaze for left/right eyes
    n_g_l = R * g_l
    n_g_r = R * g_r
    n_g_l /= np.linalg.norm(n_g_l)
    n_g_r /= np.linalg.norm(n_g_r)
    n_g_l = vector_to_pitchyaw(-n_g_l.T).flatten()
    n_g_r = vector_to_pitchyaw(-n_g_r.T).flatten()

    # Basic visualization for debugging purposes
    if i % 50 == 0:
        to_visualize = cv.equalizeHist(cv.cvtColor(patch, cv.COLOR_RGB2GRAY))
        to_visualize = draw_gaze(to_visualize, (0.25 * ow, 0.3 * oh), n_g_l,
                                 length=80.0, thickness=1)
        to_visualize = draw_gaze(to_visualize, (0.75 * ow, 0.3 * oh), n_g_r,
                                 length=80.0, thickness=1)
        to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.3 * oh), n_g,
                                 length=80.0, thickness=1)
        to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.5 * oh), n_h,
                                 length=40.0, thickness=3, color=(0, 0, 0))
        to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.5 * oh), n_h,
                                 length=40.0, thickness=1,
                                 color=(255, 255, 255))
        cv.imshow('normalized_patch', to_visualize)
        cv.waitKey(1)

    return {
        'patch': patch.astype(np.uint8),
        'gaze_direction': g.astype(np.float32),
        'gaze_origin': g_o.astype(np.float32),
        'gaze_target': g_t.astype(np.float32),
        'head_pose': h.astype(np.float32),
        'normalization_matrix': np.transpose(R).astype(np.float32),
        'normalized_gaze_direction': n_g.astype(np.float32),
        'normalized_gaze_direction_left': n_g_l.astype(np.float32),
        'normalized_gaze_direction_right': n_g_r.astype(np.float32),
        'normalized_head_pose': n_h.astype(np.float32),
    }


if __name__ == '__main__':
    # Grab SFM coordinates and store
    face_model_fpath = './sfm_face_coordinates.npy'
    face_model_3d_coordinates = np.load(face_model_fpath)

    # Grab all face coordinates
    sfm_model = eos.morphablemodel.load_model('./eos/sfm_shape_3448.bin')
    shape_model = sfm_model.get_shape_model()
    sfm_points = np.array([shape_model.get_mean_at_point(d)
                           for d in range(1, 3448)]).reshape(-1, 3)
    rotate_mat = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
    sfm_points = np.matmul(sfm_points, rotate_mat)
    between_eye_point = np.mean([sfm_points[181, :], sfm_points[614, :]],
                                axis=0)
    sfm_points -= between_eye_point.reshape(1, 3)
    full_face_model_3d_coordinates = sfm_points

    # Preprocess some datasets
    output_dir = './outputs_sted/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    datasets = {
        'MPIIGaze': {
            # Path to the MPIIFaceGaze dataset
            # Sub-folders names should consist of person IDs, for example:
            # p00, p01, p02, ...
            'input-path': '/media/wookie/WookExt4/datasets/MPIIFaceGaze',

            # A supplementary HDF file with preprocessing data,
            # as provided by us. See grab_prerequisites.bash
            'supplementary': './MPIIFaceGaze_supplementary.h5',

            # Desired output path for the produced HDF
            'output-path': output_dir + '/MPIIGaze.h5',
        },
        'GazeCapture': {
            # Path to the GazeCapture dataset
            # Sub-folders names should consist of person IDs, for example:
            # 00002, 00028, 00141, ...
            'input-path': '/media/wookie/WookExt4/datasets/GazeCapture',

            # A supplementary HDF file with preprocessing data,
            # as provided by us. See grab_prerequisites.bash
            'supplementary': './GazeCapture_supplementary.h5',

            # Desired output path for the produced HDF
            'output-path': output_dir + '/GazeCapture.h5',
        },
    }
    for dataset_name, dataset_spec in datasets.items():
        # Perform the data normalization
        with h5py.File(dataset_spec['supplementary'], 'r') as f:
            for person_id, group in f.items():
                print('')
                print('Processing %s/%s' % (dataset_name, person_id))
                data_normalization(dataset_name,
                                   dataset_spec['input-path'],
                                   group,
                                   dataset_spec['output-path'])

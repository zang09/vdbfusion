import numpy as np
import open3d as o3d


class OusterCalib:
    def __init__(self):
        #print('OusterCalib __init__')
        self.h = 128
        self.w = 1024
        self.n = 27.67
        self.range_unit = 0.001

        self.lidar_to_sensor_transform = np.array(
            [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 36.18, 0, 0, 0, 1]).reshape(
                (4, 4))

        self.altitude_table = np.array([
            45.92, 44.91, 44.21, 43.76, 42.93, 41.95, 41.25, 40.81, 39.97,
            39.02, 38.31, 37.85, 37.02, 36.08, 35.38, 34.9, 34.09, 33.16,
            32.46, 31.97, 31.16, 30.25, 29.56, 29.06, 28.25, 27.36, 26.67,
            26.16, 25.34, 24.48, 23.79, 23.26, 22.45, 21.62, 20.93, 20.39,
            19.58, 18.77, 18.09, 17.54, 16.72, 15.94, 15.26, 14.69, 13.88,
            13.11, 12.44, 11.85, 11.03, 10.3, 9.63, 9.04, 8.2, 7.49, 6.83,
            6.22, 5.37, 4.69, 4.04, 3.4, 2.55, 1.89, 1.24, 0.58, -0.27, -0.91,
            -1.56, -2.23, -3.09, -3.71, -4.36, -5.05, -5.91, -6.51, -7.17,
            -7.87, -8.74, -9.32, -9.98, -10.71, -11.57, -12.14, -12.79, -13.54,
            -14.41, -14.96, -15.62, -16.41, -17.26, -17.8, -18.46, -19.26,
            -20.12, -20.64, -21.31, -22.14, -22.98, -23.49, -24.16, -25.01,
            -25.87, -26.36, -27.03, -27.9, -28.76, -29.24, -29.91, -30.81,
            -31.67, -32.13, -32.82, -33.73, -34.61, -35.06, -35.75, -36.69,
            -37.56, -38, -38.7, -39.66, -40.54, -40.97, -41.68, -42.65, -43.57,
            -43.97, -44.68, -45.68
        ])

        self.azimuth_table = np.array([
            11.47, 4.15, -3.05, -10.13, 11, 3.95, -2.97, -9.83, 10.59, 3.8,
            -2.9, -9.55, 10.23, 3.65, -2.86, -9.31, 9.93, 3.52, -2.82, -9.12,
            9.66, 3.41, -2.79, -8.94, 9.42, 3.31, -2.77, -8.8, 9.21, 3.22,
            -2.74, -8.69, 9.03, 3.13, -2.73, -8.59, 8.88, 3.06, -2.73, -8.51,
            8.74, 3.01, -2.71, -8.45, 8.64, 2.95, -2.74, -8.4, 8.54, 2.9,
            -2.73, -8.38, 8.45, 2.86, -2.76, -8.37, 8.4, 2.82, -2.78, -8.36,
            8.36, 2.8, -2.79, -8.37, 8.34, 2.78, -2.81, -8.4, 8.32, 2.75,
            -2.86, -8.43, 8.32, 2.72, -2.89, -8.5, 8.34, 2.72, -2.93, -8.58,
            8.36, 2.71, -2.98, -8.67, 8.41, 2.72, -3.03, -8.79, 8.47, 2.72,
            -3.07, -8.9, 8.55, 2.73, -3.13, -9.06, 8.65, 2.74, -3.2, -9.21,
            8.77, 2.77, -3.28, -9.41, 8.92, 2.81, -3.39, -9.63, 9.1, 2.84,
            -3.48, -9.87, 9.3, 2.89, -3.6, -10.16, 9.53, 2.95, -3.73, -10.51,
            9.82, 3.04, -3.89, -10.89, 10.17, 3.12, -4.05, -11.33
        ])

        self.shift_table = np.array([
            65, 44, 23, 3, 63, 43, 24, 4, 62, 43, 24, 5, 61, 42, 24, 6, 60, 42,
            24, 6, 59, 42, 24, 7, 59, 41, 24, 7, 58, 41, 24, 7, 58, 41, 24, 8,
            57, 41, 24, 8, 57, 41, 24, 8, 57, 40, 24, 8, 56, 40, 24, 8, 56, 40,
            24, 8, 56, 40, 24, 8, 56, 40, 24, 8, 56, 40, 24, 8, 56, 40, 24, 8,
            56, 40, 24, 8, 56, 40, 24, 8, 56, 40, 24, 7, 56, 40, 23, 7, 56, 40,
            23, 7, 56, 40, 23, 6, 57, 40, 23, 6, 57, 40, 23, 5, 57, 40, 22, 5,
            58, 40, 22, 4, 58, 40, 22, 3, 59, 40, 21, 2, 60, 41, 21, 1, 61, 41,
            20, 0
        ])

        assert self.h == len(self.azimuth_table)
        assert self.h == len(self.altitude_table)

        self.lut_dir, self.lut_offset = self._xyz2lut()

    def unproject(self, range_im, factor=1):
        '''
        Unproject a range image in the shape of (H, W)
        '''
        h, w = range_im.shape
        assert h == self.h // factor
        assert w == self.w // factor

        range_im = np.expand_dims(range_im, axis=-1)
        xyz = range_im * self.lut_dir[::factor, ::factor] \
            + self.lut_offset[::factor, ::factor]

        return xyz

    def project(self, xyz, factor=1):
        '''
        \param xyz Input point cloud (N, 3).
        \param w Width of the range image. Default: 1024
        \param h Height of the range image. Default: 128c = Calib()
        \param altitude_table Table of the altitude per row. (128, ) array
        \param n Lidar origin to beam origin offset in mm
        \param lidar_to_sensor_transform Lidar to scan coordinate transformation (4, 4)
        \param shift_table Shift per row in pixels. (128, ) array
        '''
        altitude_table = np.deg2rad(self.altitude_table)
        alpha = np.deg2rad(360 / self.w)

        # Rigid transformation
        sensor_to_lidar_transform = np.linalg.inv(
            self.lidar_to_sensor_transform)
        R = sensor_to_lidar_transform[:3, :3]
        t = sensor_to_lidar_transform[:3, 3:]

        xyz = (xyz / self.range_unit) @ R.T + t.T

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        range_pt = np.linalg.norm(xyz, axis=1)

        # Azimuth
        u_hat = np.arctan2(y, x)
        u_hat[u_hat < 0] += 2 * np.pi
        u_hat = np.floor((2 * np.pi - u_hat) / alpha)
        u = u_hat

        # Altitude
        phi = np.arcsin(z / range_pt)
        altitude_v = np.array(
            list(map(lambda x: altitude_table[int(x)], np.arange(self.h))))
        altitude_v_grid, phi_grid = np.meshgrid(altitude_v, phi)
        v_hat = (np.abs(altitude_v_grid - phi_grid)).argmin(axis=1)
        v = np.round(v_hat).astype(int)
        v = np.minimum(np.maximum(v, 0), self.h - 1)

        # Un-shift uv
        # azimuth_table before normalization: max 11.47, min -11.33
        # azimuth_table after normalization: max 22.8, min 0
        # shift after normalization: max 65, min 0
        # => shift before normalization: max 32.70, min -32.30
        # We need to compensate 32.30 pixels

        # Update, shifting by v not working properly Jul-19, fallback
        # u -= (self.shift_table[np.arange(0, len(u)) // self.w] - 32.3)

        # Update, works when normals are fixed Jul-22
        u -= (self.shift_table[v] - 32.3)

        u = np.round(u).astype(int)
        u[u < 0] += self.w
        u[u >= self.w] -= self.w

        return u//factor, v//factor, z

    def shift(self, im, factor=1):
        shifted = np.zeros_like(im)
        h = shifted.shape[0]
        w = shifted.shape[1]

        assert h == self.h // factor
        assert w == self.w // factor

        for i in range(h):
            s = self.shift_table[i * factor] // factor
            shifted[i, s:] = im[i, :(w - s)]
            shifted[i, :s] = im[i, (w - s):]

        return shifted

    def _xyz2lut(self):
        '''
        Ref: https://github.com/ouster-lidar/ouster_example/blob/master/ouster_client/src/lidar_scan.cpp#L11
        \brief Generate two (H, W, 3) pixel-wise lookup tables for xyz.
        \param w Width of the range image. Default: 1024
        \param h Height of the range image. Default: 128
        \param azimuth_table Table of the azimuth offset per column. (128, ) array
        \param altitude_table Table of the altitude offset per column. (128, ) array
        \param n Lidar origin to beam origin offset in mm
        \param lidar_to_sensor_transform Lidar to scan coordinate transformation (4, 4)
        \return lut_dir Directional look up table (H, W, 3)
        \return lut_offset Offset look up table (H, W, 3)
        '''

        # column 0: 2pi ==> column w-1: 2pi - (w-1)/w*2pi = 2pi/w
        theta_encoder = np.linspace(2 * np.pi, 2 * np.pi / self.w, self.w)
        theta_encoder = np.tile(theta_encoder, (self.h, 1))

        # unroll azimuth table
        theta_azimuth = -np.deg2rad(self.azimuth_table)
        theta_azimuth = np.tile(np.expand_dims(theta_azimuth, axis=1),
                                (1, self.w))

        # unroll altitude table
        phi = np.deg2rad(self.altitude_table)
        phi = np.tile(np.expand_dims(phi, axis=1), (1, self.w))

        x_dir = np.cos(theta_encoder + theta_azimuth) * np.cos(phi)
        y_dir = np.sin(theta_encoder + theta_azimuth) * np.cos(phi)
        z_dir = np.sin(phi)
        lut_dir = np.stack((x_dir, y_dir, z_dir)).transpose((1, 2, 0))

        x_offset = self.n * (np.cos(theta_encoder) - x_dir)
        y_offset = self.n * (np.sin(theta_encoder) - y_dir)
        z_offset = self.n * (-z_dir)

        lut_offset = np.stack((x_offset, y_offset, z_offset)).transpose(
            (1, 2, 0))

        R = self.lidar_to_sensor_transform[:3, :3]
        t = self.lidar_to_sensor_transform[:3, 3:]

        lut_dir = (lut_dir.reshape((-1, 3)) @ R.T).reshape((self.h, self.w, 3))
        lut_offset += t.T

        return lut_dir * self.range_unit, lut_offset * self.range_unit


ouster_calib_singleton = None


def get_ouster_calib():
    global ouster_calib_singleton
    if not ouster_calib_singleton:
        ouster_calib_singleton = OusterCalib()
    return ouster_calib_singleton


def load_range(filename):
    if filename.endswith('png'):
        import cv2
        return cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(float)
    elif filename.endswith('csv'):
        return np.loadtxt(filename, delimiter=',')
    else:
        print('Unrecognized extension {}'.format(filename))


class OusterData:
    def __init__(self, filename=None):
        if filename is None:
            self.range_im = np.zeros((0, 0))
            self.xyz_im = np.zeros((0, 0, 3))
            self.xyz = np.zeros((0, 0, 3))

        else:
            self.range_im = load_range(filename)

            c = get_ouster_calib()
            assert c.h == self.range_im.shape[0]
            assert c.w == self.range_im.shape[1]

            self.xyz_im = c.unproject(self.range_im)
            self.xyz = self.xyz_im.reshape((-1, 3))

    def downsample(self, factor=2):
        output = OusterData()

        output.range_im = np.copy(self.range_im[::factor, ::factor])

        output.xyz_im = np.copy(self.xyz_im[::factor, ::factor])
        output.xyz = output.xyz_im.reshape((-1, 3))

        return output

    def to_o3d_pointcloud(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz)
        return pcd


def window_search(src, dst, u_src2dst, v_src2dst, half_window=1):
    mask = (src.range_im > 0).flatten()
    u_src2dst[~mask] = 0
    v_src2dst[~mask] = 0

    # Naive, can be improved via vectorization
    c = get_ouster_calib()

    # Vectorized us and vs
    us = []
    vs = []
    for u_offset in range(-half_window, half_window + 1):
        for v_offset in range(-half_window, half_window + 1):
            us.append((u_src2dst + u_offset) % c.w)
            vs.append(np.maximum(np.minimum(v_src2dst + v_offset, c.h - 1), 0))

    # Shape: (window_size^2, N)
    us = np.stack(us)
    vs = np.stack(vs)

    dst_xyz_nb = dst.xyz_im[vs, us]
    src_xyz = np.expand_dims(src.xyz, axis=0)
    diff = np.linalg.norm(src_xyz - dst_xyz_nb, axis=2)

    # Shape: (N, ), each entry in [0, window_size]
    nb_sel = np.argmin(diff, axis=0)
    ind = np.arange(0, diff.shape[1])

    u_updated = us[nb_sel, ind]
    v_updated = vs[nb_sel, ind]

    return u_updated, v_updated, mask

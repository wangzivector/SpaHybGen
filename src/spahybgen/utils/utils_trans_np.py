# Inherent from [VGN](https://github.com/ethz-asl/vgn) and 
# [On the Continuity of Rotation Representations in Neural Networks](https://github.com/papagina/RotationContinuity/tree/master) and
# [GenDexGrasp](https://github.com/tengyu-liu/GenDexGrasp)

import numpy as np
import scipy.spatial.transform as tf_tool
from spahybgen.utils.lie_groups.numpy import SO3


def normalize_vector(v):
        # batch*n
        batch=v.shape[0]
        v_mag = np.sqrt(np.power(v, 2).sum(axis=1)) # batch
        v_mag = np.maximum(v_mag, 1e-8)
        v_mag = np.tile(np.reshape(v_mag, (batch, 1)), (1, v.shape[1]))
        v = v / v_mag
        return v

# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
    out = np.stack((i.reshape((batch)), j.reshape((batch)), k.reshape((batch))), axis = 1)#batch*3
    return out


def quat2so3(rotations):
    _rotations_so3 = [SO3.log(SO3.from_quaternion(rot_single, ordering='xyzw')) for rot_single in rotations]
    return np.array(_rotations_so3).astype(np.float32)


def so32quat(rotations):
    rotations_quat = [SO3.exp(rot_single).to_quaternion(ordering='xyzw') for rot_single in rotations]
    return np.array(rotations_quat).astype(np.float32)


def quat2R6d(quaternion, re_R9d=False):
    ## To Rotation 6D details see: https://zhouyisjtu.github.io/project_rotation/rotation.html
    ## and code in: https://github.com/papagina/RotationContinuity/tree/master
    #   quaternion batch*4
    is_batch = True
    if len(quaternion.shape) == 1: 
        is_batch = False
        quaternion = np.expand_dims(quaternion, axis=0)

    batch=quaternion.shape[0]
    quat = normalize_vector(quaternion)
    qx = np.reshape(quat[...,0], (batch, 1))
    qy = np.reshape(quat[...,1], (batch, 1))
    qz = np.reshape(quat[...,2], (batch, 1))
    qw = np.reshape(quat[...,3], (batch, 1))
    # Unit quaternion rotation matrices computatation  
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw
    row0 = np.stack((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), axis=1) #batch*3
    row1 = np.stack((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), axis=1) #batch*3
    row2 = np.stack((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), axis=1) #batch*3
    matrix = np.stack([row0.reshape((batch, 3)), row1.reshape((batch, 3)), row2.reshape((batch, 3))], axis=2) #batch*3*3
    mat_R6d = np.hstack((matrix[:, 0, :].reshape(batch, 3), matrix[:, 1, :].reshape(batch, 3)))
    if re_R9d: return matrix.astype(np.float32)
    if not is_batch: mat_R6d = mat_R6d.squeeze()
    return mat_R6d.astype(np.float32)


#poses batch*6
def R6d2R9d(poses):
    is_batch = True
    if len(poses.shape) == 1: 
        is_batch = False
        poses = np.expand_dims(poses, axis=0)

    x_raw = poses[:,0:3] # batch*3
    y_raw = poses[:,3:6] # batch*3
        
    x = normalize_vector(x_raw) # batch*3
    z = cross_product(x,y_raw) # batch*3
    z = normalize_vector(z) # batch*3
    y = cross_product(z,x) # batch*3
        
    x = x.reshape((-1,3))
    y = y.reshape((-1,3))
    z = z.reshape((-1,3))
    matrix = np.stack((x,y,z), axis=2) #batch*3*3
    if not is_batch: matrix = matrix.squeeze()
    return matrix


def R6d2quat(R6d):
    matrix = R6d2R9d(R6d)
    return tf_tool.Rotation.from_matrix(matrix).as_quat()


def weighted_average_quaternions(quaternions, weights):
    """
    Source code for sksurgerycore.algorithms.averagequaternions
    https://scikit-surgerycore.readthedocs.io/en/stable/_modules/sksurgerycore/algorithms/averagequaternions.html

    Average multiple quaternions with specific weights

    :params quaternions: is a Nx4 numpy matrix and contains the quaternions
        to average in the rows.
        The quaternions are arranged as (w,x,y,z), with w being the scalar

    :params weights: The weight vector w must be of the same length as
        the number of rows in the

    :returns: the average quaternion of the input. Note that the signs
        of the output quaternion can be reversed, since q and -q
        describe the same orientation
    :raises: ValueError if all weights are zero
    """
    # Number of quaternions to average
    samples = quaternions.shape[0]
    mat_a = np.zeros(shape=(4, 4), dtype=np.float64)
    weight_sum = 0

    for i in range(0, samples):
        quat = quaternions[i, :]
        mat_a = weights[i] * np.outer(quat, quat) + mat_a
        weight_sum += weights[i]

    if weight_sum <= 0.0:
        raise ValueError("At least one weight must be greater than zero")

    # scale
    mat_a = (1.0/weight_sum) * mat_a

    # compute eigenvalues and -vectors
    eigen_values, eigen_vectors = np.linalg.eig(mat_a)

    # Sort by largest eigenvalue
    eigen_vectors = eigen_vectors[:, eigen_values.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    return np.real(np.ravel(eigen_vectors[:, 0]))


class Rotation(tf_tool.Rotation):
    @classmethod
    def identity(cls):
        return cls.from_quat([0.0, 0.0, 0.0, 1.0])


class Transform(object):
    """Rigid spatial transform between coordinate systems in 3D space.

    Attributes:
        rotation (scipy.spatial.transform.Rotation)
        translation (np.ndarray)
    """

    def __init__(self, rotation, translation):
        assert isinstance(rotation, tf_tool.Rotation)
        assert isinstance(translation, (np.ndarray, list))

        self.rotation = rotation
        self.translation = np.asarray(translation, np.double)

    def as_matrix(self):
        """Represent as a 4x4 matrix."""
        return np.vstack(
            (np.c_[self.rotation.as_matrix(), self.translation], [0.0, 0.0, 0.0, 1.0])
        )

    def to_dict(self):
        """Serialize Transform object into a dictionary."""
        return {
            "rotation": self.rotation.as_quat().tolist(),
            "translation": self.translation.tolist(),
        }

    def to_list(self):
        return np.r_[self.rotation.as_quat(), self.translation]

    def __mul__(self, other):
        """Compose this transform with another."""
        rotation = self.rotation * other.rotation
        translation = self.rotation.apply(other.translation) + self.translation
        return self.__class__(rotation, translation)

    def transform_point(self, point):
        return self.rotation.apply(point) + self.translation

    def transform_vector(self, vector):
        return self.rotation.apply(vector)

    def inverse(self):
        """Compute the inverse of this transform."""
        rotation = self.rotation.inv()
        translation = -rotation.apply(self.translation)
        return self.__class__(rotation, translation)

    @classmethod
    def from_matrix(cls, m):
        """Initialize from a 4x4 matrix."""
        rotation = Rotation.from_matrix(m[:3, :3])
        translation = m[:3, 3]
        return cls(rotation, translation)

    @classmethod
    def from_dict(cls, dictionary):
        rotation = Rotation.from_quat(dictionary["rotation"])
        translation = np.asarray(dictionary["translation"])
        return cls(rotation, translation)

    @classmethod
    def from_list(cls, list):
        rotation = Rotation.from_quat(list[:4])
        translation = list[4:]
        return cls(rotation, translation)
    
    @classmethod
    def from_list_transrotvet(cls, list):
        translation = list[:3]
        rotation = Rotation.from_rotvec(list[3:])
        return cls(rotation, translation)

    @classmethod
    def identity(cls):
        """Initialize with the identity transformation."""
        rotation = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
        translation = np.array([0.0, 0.0, 0.0])
        return cls(rotation, translation)

    @classmethod
    def look_at(cls, eye, center, up):
        """Initialize with a LookAt matrix.

        Returns:
            T_eye_ref, the transform from camera to the reference frame, w.r.t.
            which the input arguments were defined.
        """
        eye = np.asarray(eye)
        center = np.asarray(center)

        forward = center - eye
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        up = np.asarray(up) / np.linalg.norm(up)
        up = np.cross(right, forward)

        m = np.eye(4, 4)
        m[:3, 0] = right
        m[:3, 1] = -up
        m[:3, 2] = forward
        m[:3, 3] = eye

        return cls.from_matrix(m).inverse()

import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from reconstruct_clip import align_cameras,to_local_camera
from render_saved import between


class GeometryContract(unittest.TestCase):
    def test_similarity_camera_roundtrip(self):
        source=np.stack([np.eye(4),np.eye(4)])
        source[1,:3,:3]=Rotation.from_euler('y',40,degrees=True).as_matrix()
        source[1,:3,3]=[2,.1,.5]
        rotation=Rotation.from_euler('xyz',[10,20,30],degrees=True).as_matrix()
        translation=np.array([5.,3.,2.]); scale=2.3
        reference=source.copy()
        for i in range(2):
            reference[i,:3,:3]=rotation@source[i,:3,:3]
            reference[i,:3,3]=scale*rotation@source[i,:3,3]+translation
        s,r,t=align_cameras(source,reference)
        self.assertAlmostEqual(s,scale)
        for c,want in zip(reference,source):
            np.testing.assert_allclose(to_local_camera(c,s,r,t),want,atol=1e-6)

    def test_interpolated_camera_is_rigid(self):
        a=np.eye(4); b=np.eye(4)
        b[:3,:3]=Rotation.from_euler('y',70,degrees=True).as_matrix()
        b[:3,3]=[2,0,1]
        c=between(a,b)
        np.testing.assert_allclose(c[:3,:3].T@c[:3,:3],np.eye(3),atol=1e-6)
        self.assertAlmostEqual(np.linalg.det(c[:3,:3]),1,places=6)
        np.testing.assert_allclose(c[:3,3],[1,0,.5])


if __name__=='__main__': unittest.main()

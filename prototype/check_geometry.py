"""Independent static-feature epipolar check of predicted source cameras."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def sampson(F, p, q):
    p = np.c_[p, np.ones(len(p))]
    q = np.c_[q, np.ones(len(q))]
    fp, ftq = p @ F.T, q @ F
    return np.abs(np.sum(q * fp, axis=1)) / np.sqrt(
        fp[:, 0]**2 + fp[:, 1]**2 + ftq[:, 0]**2 + ftq[:, 1]**2 + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    a = parser.parse_args()
    report = json.loads((a.run/'report.json').read_text())
    n = len(report['frame_indices'])
    frame = report['frame_indices'][0]
    ims = [cv2.imread(str(a.run/f'input_cam{c}_{frame:05}.png')) for c in (1, 2)]
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.015)
    features = []
    for i, im in enumerate(ims):
        h,w = im.shape[:2]
        mask = np.ones((h,w),np.uint8)*255
        # Exclude dancing couple, seated audience and corner logos. These are
        # coarse regions used for this diagnostic only, not dancer segmentation.
        mask[int(h*0.52):] = 0
        mask[int(h*0.18):,int(w*(0.38 if i==0 else 0.54)):int(w*(0.73 if i==0 else 0.79))]=0
        features.append(sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),mask))
    matches = cv2.BFMatcher().knnMatch(features[0][1],features[1][1],k=2)
    good = [x for x,y in matches if x.distance < .75*y.distance]
    if len(good)<8:
        print('Insufficient static matches for a calibration check')
        return
    p=np.array([features[0][0][m.queryIdx].pt for m in good])
    q=np.array([features[1][0][m.trainIdx].pt for m in good])
    F,inliers=cv2.findFundamentalMat(p,q,cv2.USAC_MAGSAC,1.5,.999,10000)
    if F is None or inliers is None:
        raise RuntimeError('Feature geometry estimation failed')
    keep=inliers.ravel().astype(bool)
    cams=np.load(a.run/'cameras.npz')
    C0,C1=cams['extrinsic_c2w'][0,[0,n]]
    K0,K1=cams['intrinsic'][0,[0,n]]
    relative=np.linalg.inv(C1)@C0
    t=relative[:3,3]
    skew=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    predicted=np.linalg.inv(K1).T@skew@relative[:3,:3]@np.linalg.inv(K0)
    errors=sampson(predicted,p[keep],q[keep])
    result=dict(matches=len(good),ransac_inliers=int(keep.sum()),
                predicted_sampson_median_px=float(np.median(errors)),
                predicted_sampson_p90_px=float(np.percentile(errors,90)),
                fitted_sampson_median_px=float(np.median(sampson(F,p[keep],q[keep]))),
                status='inconclusive',
                limitation='Repeated wall panels give ambiguous correspondences (visually inspected). Planar degeneracy possible. These residuals cannot establish camera accuracy; not a metric or human-pose validation.')
    (a.run/'geometry_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    drawn=cv2.drawMatches(ims[0],features[0][0],ims[1],features[1][0],
                         [m for m,k in zip(good,keep) if k],None,flags=2)
    cv2.imwrite(str(a.run/'static_matches.jpg'),drawn)


if __name__=='__main__':
    main()


import cv2 as cv
import numpy as np

FOCAL = 700.0       # approximate focal length
BASELINE = 0.05     # 5 cm
STEP = 5             # process every 5th frame

orb = cv.ORB_create(nfeatures=1500)
matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

stereo = cv.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=7,
    P1=8 * 7**2,
    P2=32 * 7**2,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
)

camL  = cv.VideoCapture("/dev/video0", cv.CAP_V4L2)
camR = cv.VideoCapture("/dev/video2", cv.CAP_V4L2)

if not camL.isOpened() or not camR.isOpened():
    raise RuntimeError("Could not open cameras")

map3D = []
prevGray = None
frameNo = 0
position = np.eye(4, dtype=np.float32)

while True:
    okL, left = camL.read()
    okR, right = camR.read()

    if not okL or not okR:
        print(f"camera failed, left {okL} right {okR}")
        break

    frameNo += 1

    if frameNo % STEP:
        continue

    grayL = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(right, cv.COLOR_BGR2GRAY)

    # --------------------------------------------------
    # ORB: estimate camera movement
    # --------------------------------------------------

    kp, des = orb.detectAndCompute(grayL, None)

    if prevGray is not None and des is not None:
        kp2, des2 = orb.detectAndCompute(prevGray, None)

        if des2 is not None:
            matches = matcher.match(des2, des)
            matches = sorted(matches, key=lambda m: m.distance)

            if len(matches) >= 8:
                p1 = np.float32(
                    [kp2[m.queryIdx].pt for m in matches[:100]]
                )
                p2 = np.float32(
                    [kp[m.trainIdx].pt for m in matches[:100]]
                )

                E, mask = cv.findEssentialMat(
                    p2, p1, FOCAL,
                    (grayL.shape[1] / 2, grayL.shape[0] / 2),
                    cv.RANSAC, 0.999, 1.0
                )

                if E is not None:
                    _, R, t, _ = cv.recoverPose(
                        E, p2, p1,
                        focal=FOCAL,
                        pp=(grayL.shape[1] / 2,
                            grayL.shape[0] / 2)
                    )

                    T = np.eye(4, dtype=np.float32)
                    T[:3, :3] = R.astype(np.float32)
                    T[:3, 3] = t[:, 0].astype(np.float32)

                    position = position @ T

    prevGray = grayL.copy()

    # --------------------------------------------------
    # STEREO DEPTH
    # --------------------------------------------------

    disparity = stereo.compute(
        grayL, grayR
    ).astype(np.float32) / 16.0

    valid = disparity > 1.0

    depth = np.zeros_like(disparity)
    depth[valid] = FOCAL * BASELINE / disparity[valid]

    valid &= (depth > 0.1) & (depth < 5.0)

    # --------------------------------------------------
    # SAMPLE 3D POINTS
    # --------------------------------------------------

    ys, xs = np.where(valid)

    # Don't store every pixel — keeps the map manageable.
    ys = ys[::8]
    xs = xs[::8]

    z = depth[ys, xs]
    x = (xs - grayL.shape[1] / 2) * z / FOCAL
    y = (ys - grayL.shape[0] / 2) * z / FOCAL

    points = np.column_stack((x, y, z))

    # Transform into accumulated map coordinates.
    pointsH = np.column_stack(
        (points, np.ones(len(points)))
    )

    world = (position @ pointsH.T).T[:, :3]

    map3D.extend(world.tolist())

    # --------------------------------------------------
    # SIMPLE TOP-DOWN HEIGHTMAP
    # --------------------------------------------------

    if map3D:
        pts = np.asarray(map3D)

        # X/Y position becomes map coordinates.
        px = ((pts[:, 0] + 2) * 100).astype(int)
        py = ((pts[:, 2] + 2) * 100).astype(int)

        heightmap = np.zeros((400, 400), np.float32)

        good = (
            (px >= 0) & (px < 400) &
            (py >= 0) & (py < 400)
        )

        for xx, yy, h in zip(
            px[good], py[good], pts[good, 1]
        ):
            heightmap[yy, xx] = max(
                heightmap[yy, xx], h
            )

        view = cv.normalize(
            heightmap, None, 0, 255,
            cv.NORM_MINMAX
        ).astype(np.uint8)

        view = cv.applyColorMap(
            view, cv.COLORMAP_JET
        )

    else:
        view = np.zeros((400, 400, 3), np.uint8)

    # --------------------------------------------------
    # DISPLAY
    # --------------------------------------------------

    cv.putText(
        view,
        f"3D points: {len(map3D)}",
        (10, 25),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 255), 2
    )

    cv.imshow("Live Topography", view)
    cv.imshow("Left Camera", left)

    if cv.waitKey(1) & 0xFF == ord('l'):
        break

camL.release()
camR.release()
cv.destroyAllWindows()

# Save accumulated map.
np.save("terrain_points.npy", np.asarray(map3D))

print(f"Saved {len(map3D)} 3D points.")
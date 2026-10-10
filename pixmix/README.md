# PixMix

CPU reproduction of PixMix, the real-time diminished-reality method in Herling and Broll, ISMAR 2012. The implementation follows the expanded account in Jan Herling's thesis *Advanced Real-Time Manipulation of Video Streams*: fingerprint selection, planar homography tracking, coarse-to-fine pixel maps, keyframe coherence, contour lighting correction, and a short blend on the mask border.

The solver copies live source pixels into the hole. It does not blend patches inside the mask. On later frames the previous pixel map is forwarded by the contour homography, which is what makes the video path much cheaper than solving each frame from scratch.

## Build

Uses the system OpenCV 4 C++ libraries. No GPU is required.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/pixmix_tests
./build/pixmix --benchmark
```

## Run

```bash
./build/pixmix --camera 0 --width 640 --height 480
./build/pixmix --video clip.mp4
./build/pixmix --still photo.jpg --mask hole.png
```

Batch mode takes the selection polygon in source pixels, after any rotation:

```bash
./build/pixmix --still photo.jpg --polygon 540,1360,1480,1360,1480,2780,540,2780 \
    --max-side 960 --dilate 8 --output removed.png
./build/pixmix --video clip.mp4 --polygon 400,1481,2007,1481,2007,2430,400,2430 \
    --max-side 720 --dilate 10 --no-correct --output removed.mp4 --debug-output debug.mp4
```

`--dilate N` grows the selected hole so cast shadows go with the object. `--no-correct` keeps the homography-warped contour and skips the fingerprint boundary correction, which can pull the contour open on cluttered backgrounds. OpenCV already applies the phone rotation tag when it decodes; `--rotate` is only for files that lack it.

`--debug-output` writes a 2×2 panel per frame. The top left shows tracking: the hole in red, the warped contour in yellow, the corrected contour in green (red when lost), and RANSAC inliers in cyan and outliers in red. The top right is the output. The bottom left is the reference model the coherence term aims for. The bottom right is the offset map, where hue is the direction of f(p) − p; flat color regions are joined source blocks. The batch run also prints `mean_flicker`, the mean change inside the hole between the motion-compensated previous output and the current one, next to `background_flicker`, the same measure on a ring of real background.

Selection:

- Click once to freeze the current frame and drop a contour point.
- Draw a rough polygon that fully contains the object and a little of the surrounding background.
- `z` removes the last point, `c` clears the polygon, Enter starts removal, `q` quits.

While running:

- `r` clears the selection and freezes a new frame.
- `d` toggles a mask tint.
- `p` pauses.
- `q` quits.

The overlay reports FPS and the time spent tracking, building the reference, inpainting, and compositing. The published operating point is about 15–23 FPS at 640×480 on a 2010 laptop, for holes of roughly 13k to 60k pixels. `--benchmark` measures that workload on a synthetic planar sequence and prints the median frame time, excluding the slower first frame.

A steady Release run of that benchmark here was about 25 FPS for a 13k-pixel hole and about 20 FPS for a 60k-pixel hole, with the removed region within a few gray levels of the hidden texture. Running both sizes back to back can drop the larger one under 15 FPS while the CPU is busy.

## Assumptions

Several constants are not numeric in the thesis and live in `include/config.hh`:

- border weight `tau_a = 8`
- spatial clamp `tau_s = 2000` on the squared offset difference, in full-resolution pixels (thesis eq. 4.12); costs are in raw units, with appearance as a mean per-pixel 3-channel SSD
- still layers iterate until fewer than 1% of mappings change (at least 5, at most 20 passes)
- video frames refine only the full-resolution layer, starting from the forwarded map, with 3 passes
- video random-search radius of 12 pixels
- box-filter radii 8 and 2
- 4-pixel boundary blend
- fingerprint vote against color-cluster means, using the small-box RGB so a textured contour does not split into one cluster per sample

Appearance (eq. 4.13) leaves out the center sample and only counts samples whose source lies in S. A candidate whose source window covers less than half of the known target neighborhood is rejected; otherwise windows clipped by the image border would look free. Coherence (eq. 5.16) compares the candidate's source window with the reference model, center included, and is added to the appearance cost (eq. 5.17).

Selection runs once, at full resolution. Later frames only adjust the contour.

Keyframe reset uses a 45° threshold, but the rotation is the polar angle of the homography's 2×2 block. The thesis does not give camera intrinsics, so this is an image-plane approximation rather than a decomposed 3D rotation.

## Limits

The object has to stay still in the scene, and the background behind it needs to be nearly planar. A moving object or a thick silhouette that reveals unseen background will break the homography and the keyframe guide. Selection also fails when the object looks like the ring of background used as fingerprints. The live path does not use the paper's explicit area or line constraints.

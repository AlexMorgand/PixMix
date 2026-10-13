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

`tools/annotate.html` paints those masks in a browser. Open a folder of stills, draw the hole, and save. White pixels are removed. Chrome writes `photo_mask.png` next to the image; a second object on the same photo becomes `photo_mask_2.png`.

```bash
./build/pixmix --still photo.jpg --mask photo_mask.png --output photo_removed.png
```

Batch mode takes the selection polygon in source pixels, after any rotation:

```bash
./build/pixmix --still photo.jpg --polygon 540,1360,1480,1360,1480,2780,540,2780 \
    --max-side 960 --dilate 8 --output removed.png
./build/pixmix --video clip.mp4 --polygon 400,1481,2007,1481,2007,2430,400,2430 \
    --max-side 720 --dilate 10 --no-correct --output removed.mp4 --debug-output debug.mp4
```

`--dilate N` grows the selected hole so cast shadows go with the object. `--no-correct` keeps the homography-warped contour and skips the fingerprint boundary correction, which can pull the contour open on cluttered backgrounds. OpenCV already applies the phone rotation tag when it decodes; `--rotate` is only for files that lack it.

Solver experiments can override `--alpha`, `--tau-s`, `--tau-a`, and
`--locality`. The optional full-resolution line-pattern restart is off; the
pyramid keeps the upsampled mapping. Set `PIXMIX_PROFILE=1` to print phase
timings and candidate selection to stderr.

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
- spatial cost is the minimum clamped neighbor disagreement times `1/4` (thesis eq. 4.61), with `tau_s = 4000` in full-resolution pixels; appearance is the thesis eq. 4.13 sum divided by the neighborhood size and the number of valid samples
- border pixels, including hole pixels within one 5×5 patch of the source, use `tau_a`; deeper hole pixels stay at weight 1 (thesis eq. 4.18)
- every pyramid level uses a 5×5 color patch; the thesis timing setup used grayscale only on the finest layer, but its own color comparison shows that drops region boundaries
- the coarsest level follows the thesis 3-pixel border rule, but not below 48 pixels on a side, matching the published elephant pyramid so thin lines are still visible to initialization
- still layers iterate until fewer than 3% of mappings change (at least 5, at most 10 passes)
- still images initialize only the coarsest level, by the thesis section 4.4.3 single-pixel patch shrink, ordered by structure crossing the border
- each finer level upsamples a complete mapping and then refines all masked pixels; it does not restart an inward-growing border front
- still and video both refine the complete adaptive pyramid; each video level considers both the projected forwarded map and the upsampled coarser map, with 3 passes
- random search starts over the complete source image at every layer and halves to one pixel, as in thesis section 4.3; `--search-radius` is an explicit speed/quality override
- 5×5 appearance patches by default; `--patch-size` permits controlled comparison with larger odd sizes
- box-filter radii 8 and 2
- 4-pixel boundary blend
- fingerprint vote against color-cluster means, using the small-box RGB so a textured contour does not split into one cluster per sample

Appearance (eq. 4.13) leaves out the center sample and only counts samples whose source lies in S. Samples that lie in S, including the mask border, are read from the original frame and weighted by `tau_a`; samples already synthesized inside the hole keep weight 1. Every masked pixel is refined together from the upsampled mapping. A candidate whose source window covers less than half of the known target neighborhood is rejected; otherwise windows clipped by the image border would look free. Coherence (eq. 5.16) uses the same center-free neighborhood to compare the candidate's source window with the reference model and is added to the appearance cost (eq. 5.17).

The regression suite always runs the regular texture and line-pattern cases with five fixed seeds. This catches both quality regressions and solutions that only succeed because of a lucky random search.

`tools/evaluate_rord.py` evaluates the fixed five-image RORD subset (or all 50
with `--all`) and writes machine-readable MAE, runtime, hole size, seed, and
candidate choice. `--check` enforces the accepted quick-set quality and
performance budget in `tools/rord_quick_baseline.json`.

```bash
python3 tools/evaluate_rord.py --check
python3 tools/evaluate_rord.py --all --output data/rord50/removed_corrected \
    --json data/rord50/corrected.json
```

Selection runs once, at full resolution. Later frames only adjust the contour.

Keyframe reset uses a 45° threshold, but the rotation is the polar angle of the homography's 2×2 block. The thesis does not give camera intrinsics, so this is an image-plane approximation rather than a decomposed 3D rotation.

## Limits

The object has to stay still in the scene, and the background behind it needs to be nearly planar. A moving object or a thick silhouette that reveals unseen background will break the homography and the keyframe guide. Selection also fails when the object looks like the ring of background used as fingerprints. The live path does not use the paper's explicit area or line constraints.

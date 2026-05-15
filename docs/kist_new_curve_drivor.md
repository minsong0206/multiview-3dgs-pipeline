# HUGSIM Rendering → DrivOR Integration Debug Plan

## 1. Main Goal

The goal of this debugging process is to identify and fix the reason why the **HD score** becomes low after integrating **HUGSIM rendering results** with **DrivOR**, and ultimately improve the overall driving performance.

In particular, I want to understand whether the low score comes from:

- HUGSIM rendering quality
- semantic rendering errors
- ground / background point distribution
- DrivOR input conversion
- collision scoring logic
- ego / future trajectory box alignment
- coordinate-frame mismatch
- terrain slope or height variation

The final objective is not only to explain the low score, but also to modify the problematic part so that DrivOR produces better driving results.

---

## 2. Current Debugging Focus

Whenever I explain what I am currently checking, I should explicitly state **which part of the pipeline I am looking at**.

Current focus:

```text
HUGSIM rendering output
→ DrivOR input generation
→ DrivOR scoring process
→ HD score / PDMS / NC / TTC analysis
```

More specifically, I am currently checking the **collision-related scoring part**, especially why **NC** and **TTC** become very low.

---

## 3. Current Observation

The current observation is:

```text
The road slope gradually increases.
As the road height changes, ground points are not properly detected around the ego/future boxes.
Instead, background points are detected inside or near the boxes.
As a result, NC and TTC become very low.
This directly lowers PDMS / HD score.
```

This suggests that the issue may not be only from the rendered RGB image itself. Instead, the more direct cause may be related to how the collision boxes interact with the reconstructed scene points.

---

## 4. Current Hypothesis

The current hypothesis is:

> The low HD score is likely caused by collision scoring errors where terrain / vegetation / background points in `scene.ply` overlap with the ego or future trajectory boxes.

A more detailed chain is:

```text
Road slope or height variation exists
↓
Future ego boxes may reuse or incorrectly estimate the current ego box height / z position
↓
The boxes become vertically misaligned with the actual road surface
↓
The boxes may penetrate the ground or overlap with background / terrain / vegetation points
↓
The scoring code detects false collisions
↓
NC and TTC become very low
↓
PDMS / HD score becomes low
```

---

## 5. Key Question

The key question I want to verify is:

> Should the ego box or future trajectory boxes ever penetrate the ground?

My current intuition is:

```text
No. The ego box should be placed above the drivable surface.
If the box penetrates the ground, collision detection can incorrectly count terrain or background points as obstacles.
```

Therefore, if the ego box or future boxes are visually penetrating the road surface, this is a strong signal that the scoring/collision geometry may be wrong.

---

## 6. Important Distinction

At this stage, I should distinguish between the following possible causes:

### A. Rendering-side issue

Examples:

- wrong RGB rendering
- wrong semantic rendering
- ground semantic splatted incorrectly
- ground model geometry issue
- camera pose / COLMAP alignment problem

### B. DrivOR input-side issue

Examples:

- wrong image path
- wrong camera input
- wrong semantic or depth input
- wrong coordinate conversion
- wrong route / ego pose alignment

### C. Scoring-side issue

Examples:

- ego box height is wrong
- future boxes reuse current ego box height
- collision boxes are not adjusted to local road slope
- background / terrain / vegetation points are counted as collision objects
- `scene.ply` contains points that should be ignored during collision scoring

The current evidence seems to point more strongly toward **C. Scoring-side issue**, especially collision box height / z handling and scene point filtering.

---

## 7. Current Working Interpretation

The previous hypothesis that `merge_depth_ground.py` used the wrong physical axis is now weaker.

The updated interpretation is:

```text
In the original saved coordinate system:
world +X = right
world +Y = down
world +Z = forward
```

Therefore, `points_local[:, 1] = cam_height` likely means that ground points are placed at a fixed camera-relative downward height, which is reasonable if local `+Y` aligns with physical down.

So the current focus should move away from `merge_depth_ground.py` as the primary cause, and toward the collision scoring process.

---

## 8. Things to Check Next

### 8.1 Check ego/future box placement

Verify whether:

- the current ego box is placed correctly on the road surface
- future trajectory boxes are placed at the correct height
- future boxes reuse the current ego box z value
- boxes penetrate the ground on sloped or curved sections

### 8.2 Check collision point classes

For points detected inside ego/future boxes, check whether they are:

- vehicle
- pedestrian
- terrain
- vegetation
- road
- sidewalk
- background
- unknown

If most collision points are terrain / vegetation / background, then the collision may be a false positive.

### 8.3 Check whether ground/background points should be filtered

Verify whether collision scoring should ignore:

- road
- sidewalk
- terrain
- vegetation
- low-height ground points
- points below the ego box bottom plane

### 8.4 Check road slope handling

Verify whether the score code assumes a flat road surface.

If the road height changes but future boxes keep the same height, the boxes may become misaligned.

### 8.5 Compare visualization with score result

For low NC / TTC frames, visualize:

- ego box
- future boxes
- scene points inside the boxes
- semantic class of those points
- road surface height
- camera pose
- predicted trajectory

---

## 9. Debugging Log Format

Whenever I make progress, I should update this file using the following format.

```markdown
## Update YYYY-MM-DD

### Pipeline part checked

Example:
HUGSIM rendering / DrivOR input / collision scoring / semantic map / scene.ply

### What I checked

- ...

### Observation

- ...

### Interpretation

- ...

### Decision / Next step

- ...
```

---

## Update 2026-05-12

### Pipeline part checked

Collision scoring stage — `hug_sim.py` ego box z computation and `score_calculator.py` NC/TTC future waypoint z handling.

### What I checked

1. **`hug_sim.py` `ground_height()` return value** (lines 162–175)
   - `cam_poses` in `ground_param.pkl` are **camera center poses**, not vehicle body poses.
   - The function projects the query (u, 0, v) into the nearest camera's local frame (local Y=0 = camera center plane), then maps back to world space.
   - `uhv_world[1]` = world Y coordinate of the camera-center plane at (u, v) — this is NOT the road surface.
   - World +Y = physical DOWN (confirmed: camera local +Y · world +Y ≈ 0.999). Road surface is `cam_height=1.5m` further in world +Y direction (i.e., physically below camera).
   - Missing `+ cam_height` in the return value means ego_box_z is computed from camera-plane height, not road surface.
   - `plan.py ground_height()` (lines 71–85) already has `return uv_world[1] + cam_height` — this is the correct reference.

2. **Ego box z offset from road surface (numerical verification)**
   - At t=0.25: `hug_sim.py` returns world_Y ≈ −0.037 → `ego_box_z` = +0.037 (score_z = −world_Y ≈ +0.037).
   - Actual road surface near ego at t=0.25: ground.ply world_Y mean ≈ +1.308 → score_z ≈ −1.308.
   - Gap: ego_box_z (+0.037) − road score_z (−1.308) = **+1.349m ABOVE road** (consistent across all 18 frames: +1.35 to +1.44m, ≈ cam_height).
   - This floating ego box causes background/terrain/vegetation points in `scene.ply` to intersect the box, triggering false NC/TTC collisions.

3. **`score_calculator.py` `_calculate_no_collision()` frozen z bug** (lines 373–405)
   - `ego_x, ego_y, z, ego_w, ego_l, ego_h, ego_yaw = ego_box` — `z` is captured once at current frame.
   - The loop `for idx in range(planned_traj.shape[0])` only updates `ego_x, ego_y, ego_yaw = planned_traj[idx]`.
   - `ego_trans_mat[:3, 3] = np.array([ego_x, ego_y, z])` — z stays frozen at current frame ego_box z.
   - On the 8% slope road: future waypoints at 30m ahead are ~2.4m higher in score_z than current.
   - Frozen z causes future boxes to sink into the road surface at far waypoints, causing additional false collision detections.
   - Same bug applies to `_calculate_time_to_collision()` which calls `_calculate_no_collision`.

4. **Data from `outputdrivor/scene_easy_00/data.pkl`**
   - ego_box z progression: 0.033 → 0.083 → ... → 2.444 over 18 frames (road rises steadily).
   - Planned traj extends ~4s into the future, covering ~20–30m ahead where z offset can be +1.88m.

5. **Collision debug visualization (`score_frame_collision_debug.ply`)**
   - Red scene collision points z range: 0.747–3.716, mean=2.039 (in score frame).
   - Green ground points z: 0.746–1.742, mean=0.928.
   - Ego box at t=0.25: box_z=0.033, z_top=1.533 → box bottom is at 0.033, overlapping scene pts at z≈0.75+.
   - t=1.25: 237 scene collision points, 921 ground points inside box — indicates ego box sitting partially on terrain.

### Observation

- Ego box floats ~1.35–1.44m above the actual road surface due to `ground_height()` missing `+ cam_height`.
- Future trajectory boxes inherit a frozen z value that becomes increasingly wrong on sloped road, adding more false collisions at far waypoints.
- These two bugs together cause NC=0 and TTC=0 on every frame after t=0.25.

### Interpretation

- The root cause is confirmed to be **C. Scoring-side issue** (ego box height wrong + future z frozen).
- The `merge_depth_ground.py` and 3DGS rendering quality are NOT the primary cause of low NC/TTC.
- Once Fix 1 is applied, ego_box_z will correctly sit at road surface (score_z ≈ −1.3 at t=0 instead of +0.033).
- Fix 2 (dynamic z for future waypoints) is a secondary improvement for sloped roads.

### Proposed Fixes

#### Fix 1 — `hug_sim.py` line 175
```python
# Before (wrong):
return uhv_world[1]

# After (correct, matches plan.py):
return uhv_world[1] + cam_height
```

**Why**: `cam_poses` are camera center poses; local Y=0 projects onto the camera plane, not the road surface. The road surface is `cam_height` further in world +Y (physical down) direction.

#### Fix 2 — `score_calculator.py` `_calculate_no_collision()` lines 373–405
```python
# Before:
ego_x, ego_y, z, ego_w, ego_l, ego_h, ego_yaw = ego_box
for idx in range(planned_traj.shape[0]):
    ego_x, ego_y, ego_yaw = planned_traj[idx]
    ego_trans_mat[:3, 3] = np.array([ego_x, ego_y, z])  # z frozen

# After (interpolate ground z at each future waypoint using ground_xyz_score):
ego_x, ego_y, z, ego_w, ego_l, ego_h, ego_yaw = ego_box
for idx in range(planned_traj.shape[0]):
    ego_x, ego_y, ego_yaw = planned_traj[idx]
    # Nearest-neighbor z from ground point cloud at (ego_x, ego_y)
    if ground_xyz_score is not None:
        dists = np.sum((ground_xyz_score[:, :2] - np.array([ego_x, ego_y]))**2, axis=1)
        z = ground_xyz_score[np.argmin(dists), 2]
    ego_trans_mat[:3, 3] = np.array([ego_x, ego_y, z])
```

**Why**: On sloped roads, future waypoints may be meters above/below current frame height. Reusing current z causes the future boxes to intersect road/terrain at the wrong elevation.

**Note**: Fix 2 requires passing `ground_xyz_score` (3D score-frame ground points) through the call chain: `parse_data` → `data[0]['ground_xyz_score']` → `calculate()` → `_calculate_no_collision(... ground_xyz_score=...)`.

### Decision / Next step

1. Apply Fix 1 to `hug_sim.py` and re-run DrivoR evaluation.
2. If NC/TTC still low after Fix 1, apply Fix 2 to `score_calculator.py`.
3. Regenerate `score_frame_collision_debug.ply` to visually verify ego boxes sit on road surface.
4. Compare scores before/after both fixes.

---

## 10. Current Summary

Current status:

```text
Goal:
Find and fix the reason for low HD score after HUGSIM rendering is connected to DrivOR.

Current focus:
Collision scoring stage, especially NC and TTC.

Current observation:
On sloped road sections, ground points are not properly detected and background/terrain/vegetation points may enter ego/future boxes.

Current concern:
If the ego or future box penetrates the ground, collision scoring may produce false positives.

Current likely cause:
Collision box z/height handling and scene point filtering, rather than a simple ground generation axis error.
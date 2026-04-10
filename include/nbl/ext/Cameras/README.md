# Shared Camera API

This directory contains the reusable Nabla camera stack.

The stack has two public faces:

- a runtime face used to move cameras during one frame
- a typed face used to capture, store, restore, compare, replay, and validate camera state

The runtime face is centered on [`ICamera.hpp`](ICamera.hpp).
The typed face is centered on [`CCameraGoal.hpp`](CCameraGoal.hpp) and [`CCameraGoalSolver.hpp`](CCameraGoalSolver.hpp).

## TL;DR

If you want to know which type to touch first, use this table.

| I want to... | Use |
|---|---|
| move a camera from live input this frame | `ICamera::manipulate(...)` |
| convert keyboard or mouse input into camera commands | `IGimbalInputProcessor` or `CGimbalInputBinder` |
| pair one camera with one or more projection entries | `CPlanarProjection` and `IPlanarProjection::CProjection` |
| apply one absolute rigid pose request at runtime | `camera->manipulate({}, &referenceFrame)` |
| set exact position or exact orientation on `Free` and `FPS` | `referenceFrame` built from `camera->getGimbal()` |
| set one absolute typed state that can be reused later | `CCameraGoal` + `CCameraGoalSolver` |
| capture current camera state | `CCameraGoalSolver::capture...` |
| restore a camera from typed state | `CCameraGoalSolver::apply...` |
| save a named camera state | `CCameraPreset` |
| store camera states over time | `CCameraKeyframeTrack` |
| keep playback cursor state | `CCameraPlaybackTimeline` |
| make a camera follow a moving target | `CCameraFollowUtilities` |
| author compact scripted camera sequences | `CCameraSequenceScript` |
| execute frame-by-frame scripted payloads | `CCameraScriptedRuntime` |
| use the path-rig camera | `CPathCamera` and `SCameraPathModel` |

## Quick start

This section shows the common entry points before any deeper explanation.

### 1. Live runtime camera control

Use this when keyboard, mouse, or ImGuizmo should move the camera right now.

```cpp
auto camera = core::make_smart_refctd_ptr<COrbitCamera>(eye, target);

ui::CGimbalInputBinder binder;
ui::CCameraInputBindingUtilities::applyDefaultCameraInputBindingPreset(binder, *camera);

auto collected = binder.collectVirtualEvents(timestamp, {
    .keyboardEvents = { keyEvents.data(), keyEvents.size() },
    .mouseEvents = { mouseEvents.data(), mouseEvents.size() },
    // .imguizmoEvents = { gizmoDeltaTransforms.data(), gizmoDeltaTransforms.size() },
});

camera->manipulate(collected.events);
```

The update payload currently accepts:

- `keyboardEvents`
- `mouseEvents`
- `imguizmoEvents`

What happens here:

1. device input is converted into semantic camera commands
2. the camera consumes those commands through `manipulate(...)`
3. the camera updates its gimbal pose

The controller-side stack is:

- `IGimbalBindingLayout` for the static mapping from device inputs to virtual events
- `IGimbalInputProcessor` for converting one frame of raw input into event magnitudes
- `CGimbalInputBinder` for the common runtime object that owns a layout and collects one frame of events
- `CCameraInputBindingUtilities` for shared preset layouts such as default `FPS`, `Orbit`, or `Path Rig` bindings

The two common ways to start are:

- apply one shared preset for a camera family
- write one binding layout explicitly

**Question: How do I bind `FPS` to `WASD`?**

Use the shared default binding preset for the active camera kind.

```cpp
auto camera = core::make_smart_refctd_ptr<CFPSCamera>(position, orientation);

ui::CGimbalInputBinder binder;
ui::CCameraInputBindingUtilities::applyDefaultCameraInputBindingPreset(binder, *camera);
```

For `FPS`, the default preset gives you:

- keyboard `W/S/A/D` -> forward, backward, left, right
- keyboard `I/K/J/L` -> tilt up, tilt down, pan left, pan right
- mouse relative movement -> look yaw and pitch

For `Free`, the default preset adds `Q/E` for roll.

For target-relative families and `Path Rig`, the default preset keeps the same physical inputs but maps them to the legal state space of that family.

**Question: How do I define custom bindings?**

Use one `IGimbalBindingLayout` implementation such as `CGimbalInputBinder` and write the mapping you want.

```cpp
ui::CGimbalInputBinder binder;
const double customMoveGain = /* choose a sensitivity for this binding */;

binder.updateKeyboardMapping([customMoveGain](auto& map)
{
    map.clear();
    map.emplace(ui::E_KEY_CODE::EKC_W, ui::IGimbalBindingLayout::CHashInfo(core::CVirtualGimbalEvent::MoveForward, customMoveGain));
    map.emplace(ui::E_KEY_CODE::EKC_S, ui::IGimbalBindingLayout::CHashInfo(core::CVirtualGimbalEvent::MoveBackward, customMoveGain));
    map.emplace(ui::E_KEY_CODE::EKC_A, ui::IGimbalBindingLayout::CHashInfo(core::CVirtualGimbalEvent::MoveLeft, customMoveGain));
    map.emplace(ui::E_KEY_CODE::EKC_D, ui::IGimbalBindingLayout::CHashInfo(core::CVirtualGimbalEvent::MoveRight, customMoveGain));
});
```

The same pattern works for:

- mouse bindings through `updateMouseMapping(...)`
- ImGuizmo bindings through `updateImguizmoMapping(...)`

**Question: How are `magnitude` values generated?**

`CVirtualGimbalEvent::magnitude` is one non-negative scalar attached to one semantic command.

It is not a raw device unit and it is not, by itself, the final world-space or angular motion applied by a camera.

What stays stable at the API level is the meaning by event family:

- translation events carry one controller-side translation amount
- rotation events carry one controller-side angular amount
- scale events carry one controller-side scale amount

The binding layer maps raw producer values onto those amounts. Different sources may start from:

- elapsed time for held input
- cursor deltas for relative mouse input
- scroll steps for wheel input
- world-space translation or angular deltas for gizmo-driven input

That means exact numeric gains are binding policy, not API contract. The binding layer owns sensitivity and repeat-rate tuning.

After the controller side emits virtual magnitudes, the camera runtime applies its own motion scales and legalizes the result to the concrete camera family.

The motion pipeline is therefore:

1. raw device input
2. binding-local gain
3. `CVirtualGimbalEvent { type, magnitude }`
4. camera-local motion scale
5. family-specific legalization and state update

### 2. Projection is separate from camera state

**Question: Where is `setProjectionMatrix(...)`?**

There is no `camera->setProjectionMatrix(...)`.

That is intentional.

The camera API keeps runtime camera state and projection state separate:

- `ICamera` owns pose and motion state
- `IProjection` and its derived types own projection state
- one projection wrapper references one camera when it needs `view`, `MV`, or `MVP`

This keeps the pairing flexible:

- one camera can be reused with different projection entries
- one viewport can switch projection preset without replacing the camera
- projection parameters such as FOV, orthographic width, near, and far do not have to live inside every camera kind

The split looks like this:

```cpp
auto camera = core::make_smart_refctd_ptr<COrbitCamera>(eye, target);

using projections_t = std::vector<core::IPlanarProjection::CProjection>;
auto planar = core::CPlanarProjection<projections_t>::create(core::smart_refctd_ptr(camera));

planar->getPlanarProjections().push_back(
    core::IPlanarProjection::CProjection::create<
        core::IPlanarProjection::CProjection::Perspective>(0.1f, 100.0f, 60.0f));

planar->getPlanarProjections().push_back(
    core::IPlanarProjection::CProjection::create<
        core::IPlanarProjection::CProjection::Orthographic>(0.1f, 100.0f, 10.0f));

auto& projection = planar->getPlanarProjections()[0];
projection.update(leftHanded, aspectRatio);

const auto& view = camera->getGimbal().getViewMatrix();
const auto& proj = projection.getProjectionMatrix();
```

So the camera does not own projection parameters.

Instead:

- the camera owns `view`
- the projection entry owns `projection`
- the wrapper combines both when code needs `MV`, `MVP`, or viewport-local binding state

When you want to change projection state, touch the projection layer:

- `IPlanarProjection::CProjection::setPerspective(...)`
- `IPlanarProjection::CProjection::setOrthographic(...)`
- `IPlanarProjection::CProjection::update(...)`

When you want to change pose or camera-family state, touch the camera layer:

- `ICamera::manipulate(...)`
- `referenceFrame`
- `CCameraGoal`
- family-specific typed hooks such as `trySetSphericalTarget(...)` or `trySetPathState(...)`

### 3. Apply one absolute rigid pose request

Use this when you already have one rigid transform and want the camera to consume it through the normal runtime entry point.

```cpp
const auto referenceFrame =
    hlsl::CCameraMathUtilities::composeTransformMatrix(desiredPosition, desiredOrientation);

if (camera->manipulate({}, &referenceFrame))
{
    // reference frame was accepted and applied
}
```

`manipulate(...)` can return `false`.

Common reasons are:

- there were no virtual events and no `referenceFrame`
- the supplied `referenceFrame` was not a valid rigid orthonormal transform
- the concrete camera kind could not legalize the request into its own runtime state

**Question: Why not just expose `setPosition(...)` and `setOrientation(...)` everywhere?**

Because not every camera kind stores arbitrary rigid pose as its native state.

`Free` can represent arbitrary position and orientation directly.

`FPS` cannot. Its legal runtime state is:

- world-space position
- yaw
- pitch
- upright orientation reconstructed from yaw and pitch

Consider this `FPS` example:

```cpp
const auto desiredPosition = hlsl::float64_t3(2.0, 1.0, -3.0);
const auto desiredOrientation =
    hlsl::CCameraMathUtilities::makeQuaternionFromEulerDegreesYXZ(
        hlsl::float64_t3(-15.0, 40.0, 25.0));
```

The requested rigid pose contains `roll = 25 deg`.

That roll is not legal for `FPS`.

If the API exposed unrestricted `setOrientation(...)` and accepted that quaternion as-is, the runtime camera would no longer match the rules of the `FPS` rig.

The current API does this instead:

1. accept one rigid pose request through `referenceFrame`
2. project that pose onto the legal state space of the concrete camera kind
3. rebuild the final runtime pose from that legal state

For `FPS` that means:

- keep the requested position
- read forward direction from the rigid reference
- rebuild legal `pitch/yaw`
- reject arbitrary roll
- write back one upright `FPS` pose

For `FPS`, that rejection applies to `referenceFrame`, not to stray roll virtual events. `CFPSCamera` advertises only translation plus pitch/yaw runtime control, so `RollLeft` and `RollRight` events are ignored by the `FPS` accumulator instead of causing failure.

The same pattern applies to every camera family:

- `Free` keeps the rigid pose directly
- `FPS` legalizes to upright `position + pitch/yaw`
- target-relative cameras legalize to `target + orbitUv + distance`
- `Path Rig` legalizes to `PathState`

Use this path for:

- one-shot runtime pose application
- ImGuizmo
- world-space or local-space pose anchoring

### 4. Set exact position or exact orientation on `Free` and `FPS`

Use this when the target camera is `Free` or `FPS` and you want to replace only one rigid-pose component.

```cpp
const auto& gimbal = camera->getGimbal();

const auto newPosition = desiredPosition;
const auto keepOrientation = gimbal.getOrientation();

const auto referenceFrame =
    hlsl::CCameraMathUtilities::composeTransformMatrix(newPosition, keepOrientation);

camera->manipulate({}, &referenceFrame);
```

```cpp
const auto& gimbal = camera->getGimbal();

const auto keepPosition = gimbal.getPosition();
const auto newOrientation = desiredOrientation;

const auto referenceFrame =
    hlsl::CCameraMathUtilities::composeTransformMatrix(keepPosition, newOrientation);

camera->manipulate({}, &referenceFrame);
```

`Free` applies these requests exactly.

`FPS` keeps the exact position but legalizes orientation to its upright `pitch/yaw` state.

For constrained target-relative and path cameras, prefer family-specific typed state or `CCameraGoal` instead of describing this as an exact component setter.

### 5. Set one absolute typed state

Use this when the state should survive beyond one frame or should be reused by presets, follow, playback, persistence, or scripts.

```cpp
core::CCameraGoal goal = {};
goal.position = desiredPosition;
goal.orientation = desiredOrientation;

core::CCameraGoalSolver solver;
auto apply = solver.applyDetailed(camera.get(), goal);

if (apply.succeeded() && apply.changed())
{
    // camera was updated during applyDetailed(...)
}
```

`applyDetailed(...)` does not build a deferred command object.
It immediately tries to apply `goal` to the runtime camera.

The returned `apply` value is a report describing what happened:

- whether the apply succeeded
- whether the camera actually changed
- whether the result was exact or approximate
- whether typed state was applied directly, virtual events were replayed, or both

So the control flow is:

1. build one `CCameraGoal`
2. call `applyDetailed(...)`
3. the solver immediately updates the camera if it can
4. inspect `apply` if you need status, exactness, or diagnostics

Use `applyDetailed(...)` when you want that report.
Use `apply(...)` when you only need a plain success/failure boolean.

**Question: How is this different from `composeTransformMatrix(...)` plus `camera->manipulate({}, &referenceFrame)`?**

`referenceFrame` carries one rigid pose request:

- position
- orientation

That is enough when the job is "try to place the runtime camera at this pose now".

`CCameraGoal` can carry more than one rigid pose:

- pose
- target-relative state
- path state
- dynamic perspective state
- source metadata used by tooling

So the two paths are different:

- `referenceFrame` asks the runtime camera to legalize one rigid pose request
- `CCameraGoal` asks the solver to apply one typed camera state, using direct typed hooks when available and virtual-event replay when needed

For `Free` and often `FPS`, both paths may end up close to the same result.

For constrained families they are not equivalent, because one rigid pose does not fully describe family-specific state such as:

- target position
- orbit angles plus distance
- `PathState`
- dynamic perspective parameters

Rule of thumb:

- use `referenceFrame` for one runtime rigid pose request now
- use `CCameraGoal` for one typed camera state that should be stored, compared, serialized, replayed, or applied later

### 6. Set one absolute camera-family state

Use this when you do not want a generic rigid pose and instead want to write the native state of one camera family.

Target-relative cameras:

```cpp
camera->trySetSphericalTarget(targetPosition);
camera->trySetSphericalDistance(distance);
```

Path camera:

```cpp
core::ICamera::PathState path = {
    .s = desiredS,
    .u = desiredU,
    .v = desiredV,
    .roll = desiredRoll
};

camera->trySetPathState(path);
```

Use this path when you already have:

- target-relative state
- path-rig state
- one other family-specific typed fragment exposed by `ICamera`

### 7. Capture a camera and restore it later

Use this when you want explicit camera state instead of one-frame runtime input.

```cpp
core::CCameraGoalSolver solver;

auto capture = solver.captureDetailed(camera.get());
if (capture.canUseGoal())
{
    auto apply = solver.applyDetailed(camera.get(), capture.goal);
}
```

What happens here:

1. the solver reads runtime camera state
2. the solver writes that state into one `CCameraGoal`
3. the solver later applies that goal back to a camera

### 8. Save a named camera state

Use this when one camera state needs a user-facing name or identifier.

```cpp
core::CCameraGoalSolver solver;

auto capture = solver.captureDetailed(camera.get());
if (capture.canUseGoal())
{
    core::CCameraPreset preset;
    preset.name = "Overview";
    preset.identifier = "overview";
    core::CCameraPresetUtilities::assignGoalToPreset(preset, capture.goal);
}
```

### 9. Make a camera follow a moving target

Use this when one tracked subject should drive camera behavior.

```cpp
core::CTrackedTarget trackedTarget(position, orientation);

core::SCameraFollowConfig follow = {};
follow.enabled = true;
follow.mode = core::ECameraFollowMode::LookAtTarget;

core::CCameraGoalSolver solver;
core::CCameraFollowUtilities::applyFollowToCamera(solver, camera.get(), trackedTarget, follow);
```

### 10. Build and evaluate scripted runtime payloads

Use this when camera playback is authored as compact camera-domain data and then evaluated through generic per-frame runtime payloads and checks.

```cpp
core::CCameraScriptedTimeline timeline;
core::CCameraScriptedRuntimeUtilities::finalizeScriptedTimeline(timeline);
```

## Core concepts

### `CVirtualGimbalEvent`

Defined in [`CVirtualGimbalEvent.hpp`](CVirtualGimbalEvent.hpp).

`CVirtualGimbalEvent` is one semantic camera command plus one scalar magnitude.

The scalar magnitude is a controller-side virtual amount emitted after binding
gains are applied. It is not a raw device delta and it is not, by itself, the
final world-space motion applied by a camera.

Examples:

- `MoveForward`
- `MoveLeft`
- `MoveUp`
- `TiltUp`
- `PanRight`
- `RollLeft`
- `ScaleZInc`

The event does not store device-specific origin.
The same event type can come from keyboard input, mouse input, ImGuizmo, scripted playback, or replay helpers.

### `IGimbal`

Defined in [`IGimbal.hpp`](IGimbal.hpp) and used by [`ICamera.hpp`](ICamera.hpp).

The gimbal stores runtime pose:

- position
- orientation
- scale
- orthonormal basis

It also accumulates one frame of semantic events into a `VirtualImpulse`.

`ICamera::CGimbal` extends the base gimbal with a cached world-to-view matrix.

Every runtime camera owns one `CGimbal`.

### `ICamera`

Defined in [`ICamera.hpp`](ICamera.hpp).

`ICamera` is the shared runtime interface implemented by every camera kind.

Its main job is:

- consume one frame of semantic virtual events
- optionally consume one rigid reference frame
- update internal camera state
- update runtime pose in the gimbal

Important members:

- `manipulate(...)`
- `getGimbal()`
- `getAllowedVirtualEvents()`
- `getKind()`
- `getCapabilities()`
- typed hooks such as `tryGetSphericalTargetState(...)` and `tryGetPathState(...)`

Each camera also stores one local motion-scale bundle in `SMotionConfig`.
Those scales are applied after the binding layer emits virtual magnitudes.

### `referenceFrame`

Defined by [`ICamera.hpp`](ICamera.hpp) and [`IGimbal.hpp`](IGimbal.hpp).

`referenceFrame` is the optional rigid transform passed to `ICamera::manipulate(...)`.

It is the runtime pose anchor for one manipulation step.

Typical producers:

- ImGuizmo
- restore helpers
- replay helpers
- code that wants world-space or local-space manipulation anchored to a specific rigid transform

See Quick start sections 1 to 4 for the concrete runtime usage patterns.

Shared runtime pattern:

```text
referenceFrame
  -> extract rigid reference transform
  -> resolve legal state for this camera kind
  -> accumulate virtual events
  -> apply deltas in that state space
  -> rebuild pose
```

### `SCameraRigPose`

Defined in [`SCameraRigPose.hpp`](SCameraRigPose.hpp).

`SCameraRigPose` stores only:

- world-space position
- world-space orientation

It is the smallest typed pose object reused across the stack.

### `CCameraGoal`

Defined in [`CCameraGoal.hpp`](CCameraGoal.hpp).

`CCameraGoal` is the canonical typed transport for camera state.

You can think of it as:

> one explicit camera-state snapshot used by higher-level tools

It may contain:

- pose
- target position
- target-relative distance
- orbit state
- path state
- dynamic perspective state
- source camera metadata

It is used by:

- capture
- restore
- preset flow
- playback
- follow
- scripted checks

It is not:

- a live input object
- a replacement for `manipulate(...)`
- a promise that every camera can represent every arbitrary pose exactly

For constrained cameras, the solver may project the goal onto legal camera-family state before or during apply.

### `CCameraGoalSolver`

Defined in [`CCameraGoalSolver.hpp`](CCameraGoalSolver.hpp).

`CCameraGoalSolver` converts between typed camera state and runtime cameras.

It captures runtime cameras into `CCameraGoal`, analyzes whether a target camera can represent that goal directly, and applies the result either through typed state or through runtime replay when needed.

If you want to restore one absolute camera state and you are not sure which family-specific hook to call, use `CCameraGoalSolver`.

### `CCameraPreset`

Defined in [`CCameraPreset.hpp`](CCameraPreset.hpp).

`CCameraPreset` is a named saved `CCameraGoal`.

It contains:

- `name`
- `identifier`
- `goal`

### `CCameraKeyframeTrack`

Defined in [`CCameraKeyframeTrack.hpp`](CCameraKeyframeTrack.hpp).

`CCameraKeyframeTrack` is a sequence of time-stamped presets.

Each keyframe contains:

- one preset
- one authored time

### `CCameraPlaybackTimeline`

Defined in [`CCameraPlaybackTimeline.hpp`](CCameraPlaybackTimeline.hpp).

`CCameraPlaybackTimeline` stores playback cursor state over time-based camera data.

It tracks things such as:

- current time
- direction
- looping
- paused or playing state

### `CTrackedTarget`

Defined in [`CCameraFollowUtilities.hpp`](CCameraFollowUtilities.hpp).

`CTrackedTarget` is the reusable tracked subject used by follow.

It owns its own gimbal.
It is not a mesh id and not a scene-node handle.

### `CCameraSequenceScript`

Defined in [`CCameraSequenceScript.hpp`](CCameraSequenceScript.hpp).

`CCameraSequenceScript` is the compact authored format for camera sequences.

It stores camera-domain data such as:

- targeted camera
- projection presentation requests
- camera keyframes
- tracked-target keyframes
- continuity settings
- capture fractions

It does not store frame-by-frame low-level input.

### `CCameraScriptedRuntime`

Defined in [`CCameraScriptedRuntime.hpp`](CCameraScriptedRuntime.hpp).

`CCameraScriptedRuntime` is the expanded executable form used during scripted playback and validation.

It stores runtime payloads such as:

- low-level input events
- goal and tracked-target events
- per-frame checks
- capture scheduling

Consumer-specific UI actions stay outside this shared runtime payload.

### `Path Rig`

Defined by [`CPathCamera.hpp`](CPathCamera.hpp), [`CCameraPathUtilities.hpp`](CCameraPathUtilities.hpp), and [`CCameraPathMetadata.hpp`](CCameraPathMetadata.hpp).

`Path Rig` is the camera family with typed state:

- `s`
- `u`
- `v`
- `roll`

Its runtime and typed tooling are driven by `SCameraPathModel`, which defines how path state is resolved, updated, and converted back into camera pose.

At the API boundary, you can think of `Path Rig` as:

$$
\text{choose any parametric camera function } f
\text{ that maps typed path state to pose.}
$$

In other words, the reusable seam is not "one built-in rail camera".
It is:

$$
f : (t, q, L) \mapsto (p, o)
$$

with:

- $t \in \mathbb{R}^3$:
  world-space target or anchor position used by the model
- $q \in \mathcal{Q}$:
  typed path state
- $L \in \mathcal{L}$:
  path-state limits
- $p \in \mathbb{R}^3$:
  evaluated world-space camera position
- $o \in \mathrm{SO}(3)$:
  evaluated camera orientation

In the shared built-in state representation,

$$
\mathcal{Q} = S^1 \times \mathbb{R} \times \mathbb{R} \times S^1
$$

with

$$
q = (s, u, v, \rho),
$$

where:

- $s \in S^1$ is one wrapped angular parameter
- $u \in \mathbb{R}$ is one lateral or radial parameter
- $v \in \mathbb{R}$ is one second shape or height parameter
- $\rho \in S^1$ is authored roll around the model forward axis

and the limit bundle is

$$
\mathcal{L} = \{(u_{\min}, d_{\min}, d_{\max})\},
$$

where:

- $u_{\min} \in \mathbb{R}_{\ge 0}$ is the minimal legal `u`
- $d_{\min} \in \mathbb{R}_{\ge 0}$ is the minimal legal radial distance
- $d_{\max} \in \mathbb{R}_{\ge 0} \cup \{\infty\}$ is the maximal legal radial distance

The important part is that one caller-provided model decides how typed state becomes final camera pose.

This is why the same API seam can model many constrained motions: circles, cylinders, orbits, guide curves, splines with offsets, crane-style rigs, banking on-rails cameras, and other custom parametric camera laws.

If you already have one path curve

$$
C(s) \in \mathbb{R}^3
$$

and one moving local frame

$$
R(s), U(s), F(s) \in \mathbb{R}^3,
$$

then one representative evaluator has the shape

$$
p(s,u,v) = C(s) + u\,R(s) + v\,U(s),
$$

with orientation built from the basis

$$
\bigl(R(s), U(s), F(s)\bigr)
$$

and then rotated by authored roll $\rho$ around the current forward axis.

The built-in model below is just one concrete default implementation of that seam.
It happens to have one simple closed-form `resolveState(...)` from world-space position, but custom models only need to provide a legal state-resolution callback. They do not need one strict analytical inverse of the evaluator.

**Default built-in model**

If you do not supply your own `SCameraPathModel`, `CPathCamera` uses the built-in cylindrical model.

That default model uses a cylindrical parameterization around the current target position
$t = (t_x, t_y, t_z)$ with typed state
$q = (s, u, v, \rho)$:

$$
\begin{aligned}
x &= t_x + u \cos s \\
y &= t_y + v \\
z &= t_z + u \sin s
\end{aligned}
$$

That means:

- `s` is the authored angle around the target in the world `XZ` plane
- `u` is the planar `XZ` radius
- `v` is the vertical offset on world `Y`
- `roll` is an extra rotation applied around the resulting forward axis

For the built-in model, `resolveState(...)` from one world-space position can be written as:

$$
\begin{aligned}
\Delta &= p - t \\
s &= \operatorname{wrap}\!\left(\operatorname{atan2}(\Delta_z, \Delta_x)\right) \\
u &= \max\!\left(u_{\min}, \sqrt{\Delta_x^2 + \Delta_z^2}\right) \\
v &= \Delta_y
\end{aligned}
$$

The default model also derives one radial camera distance from `(u, v)`:

$$
d = \lVert (u, v) \rVert_2 = \sqrt{u^2 + v^2}
$$

and sanitizes state as:

$$
\begin{aligned}
s &\leftarrow \operatorname{wrap}(s) \\
u &\leftarrow \max(u_{\min}, u) \\
\rho &\leftarrow \operatorname{wrap}(\rho)
\end{aligned}
$$

The base orientation is then built from the camera looking from the resolved position back at the target, and the authored roll is applied around that resulting forward axis.

The built-in control law maps runtime local motion into path-state delta as:

$$
\Delta q =
\begin{bmatrix}
\Delta s \\
\Delta u \\
\Delta v \\
\Delta \rho
\end{bmatrix}
=
\begin{bmatrix}
\Delta z_{\text{local}} \\
\Delta x_{\text{local}} \\
\Delta y_{\text{local}} \\
\Delta \mathrm{roll}
\end{bmatrix}
$$

and integrates it as:

$$
q_{n+1} = \operatorname{sanitize}(q_n + \Delta q)
$$

Equivalent pseudocode for the built-in model is:

```cpp
PathState state = sanitize(inputState, limits);

const double appliedU = max(limits.minU, state.u);
const dvec3 offset = {
    cos(state.s) * appliedU,
    state.v,
    sin(state.s) * appliedU
};

const dvec3 requestedPosition = target + offset;
const auto [orbitUv, distance] =
    buildOrbitFromPosition(target, requestedPosition, limits.minDistance, limits.maxDistance);

auto [position, orientation] =
    buildSphericalPoseFromOrbit(target, orbitUv, distance, limits.minDistance, limits.maxDistance);

if (state.roll != 0.0)
    orientation = applyRollAroundCurrentForward(orientation, state.roll);

PathDelta delta = {
    .s = localTranslation.z,
    .u = localTranslation.x,
    .v = localTranslation.y,
    .roll = localRotation.z
};

state = sanitize(state + delta, limits);
```

This is intentionally more general than one hardcoded "rail camera".

The built-in model shown above is only one concrete parameterization.
The reusable part is `SCameraPathModel`, which lets the runtime reinterpret the same typed `PathState` seam through custom:

- state resolution
- control law
- integration
- pose evaluation
- distance update

In practice that means the same `Path Rig` family can be used for many constrained camera designs, for example:

- cylindrical and orbital rigs around one subject
- dolly or crane-style motion with authored lateral and vertical offsets
- cameras constrained to one spline or guide path with side/up offsets
- banked path cameras where `roll` becomes authored banking around the current forward axis
- on-rails gameplay or cinematic cameras with one path parameter plus local offsets
- custom path-following rigs that keep the runtime API and typed tooling unchanged while replacing only the path model

So the important boundary is:

- the built-in model is one cylindrical target-relative parameterization
- the `Path Rig` API surface is the extensible typed seam for path-driven camera families

It can represent a large class of practical constrained camera motions, but it is still not "arbitrary free pose".
If a camera must store completely unconstrained 6DOF pose as its native state, use `Free`.

## Camera families

- `Free` cameras in [`CFPSCamera.hpp`](CFPSCamera.hpp) and [`CFreeLockCamera.hpp`](CFreeLockCamera.hpp) store world-space position plus free or FPS-constrained orientation.
- Target-relative cameras are built on [`CSphericalTargetCamera.hpp`](CSphericalTargetCamera.hpp) and include [`COrbitCamera.hpp`](COrbitCamera.hpp), [`CArcballCamera.hpp`](CArcballCamera.hpp), [`CTurntableCamera.hpp`](CTurntableCamera.hpp), [`CTopDownCamera.hpp`](CTopDownCamera.hpp), [`CIsometricCamera.hpp`](CIsometricCamera.hpp), [`CChaseCamera.hpp`](CChaseCamera.hpp), [`CDollyCamera.hpp`](CDollyCamera.hpp), and [`CDollyZoomCamera.hpp`](CDollyZoomCamera.hpp). They store target position, `orbitUv`, and distance instead of arbitrary free pose.
- [`CDollyZoomCamera.hpp`](CDollyZoomCamera.hpp) extends the target-relative family with dynamic perspective state `baseFov` and `referenceDistance`.
- [`CPathCamera.hpp`](CPathCamera.hpp) uses the parametric path-state seam described above together with limits `minU`, `minDistance`, and `maxDistance`.

## Typed tooling

Use the typed layer when camera state must outlive the current frame or be exchanged between tools:

1. `SCameraRigPose` is the smallest typed pose fragment.
2. `CCameraGoal` is the canonical typed transport for camera state.
3. `CCameraGoalSolver` captures runtime cameras into goals and applies goals back to runtime cameras.
4. `CCameraPreset` gives one goal a stable user-facing identity.
5. `CCameraKeyframeTrack` stores presets over authored time.
6. `CCameraPlaybackTimeline` stores playback cursor state while a track is being evaluated.

## Follow

Follow lives in [`CCameraFollowUtilities.hpp`](CCameraFollowUtilities.hpp) and [`CCameraFollowRegressionUtilities.hpp`](CCameraFollowRegressionUtilities.hpp). It combines one `CTrackedTarget`, one follow mode, and one follow configuration, then builds the resulting camera goal and applies it through the shared goal solver. Available modes are `OrbitTarget`, `LookAtTarget`, `KeepWorldOffset`, and `KeepLocalOffset`.

## Scripting

### Compact authored format

[`CCameraSequenceScript.hpp`](CCameraSequenceScript.hpp) and [`CCameraSequenceScriptPersistence.hpp`](CCameraSequenceScriptPersistence.hpp) store authored camera-domain data.

### Expanded runtime format

[`CCameraScriptedRuntime.hpp`](CCameraScriptedRuntime.hpp) and [`CCameraScriptedCheckRunner.hpp`](CCameraScriptedCheckRunner.hpp) store executable per-frame runtime payloads and validation checks.

Common flow:

```text
compact authored sequence
  -> compile or expand
  -> scripted runtime payload
  -> execute against runtime camera state
```

## Projection and presentation helpers

Projection types live in [`IProjection.hpp`](IProjection.hpp), [`ILinearProjection.hpp`](ILinearProjection.hpp), [`IPerspectiveProjection.hpp`](IPerspectiveProjection.hpp), [`IPlanarProjection.hpp`](IPlanarProjection.hpp), [`CLinearProjection.hpp`](CLinearProjection.hpp), [`CPlanarProjection.hpp`](CPlanarProjection.hpp), and [`CCubeProjection.hpp`](CCubeProjection.hpp).

Camera-facing presentation helpers live in [`CCameraPresentationUtilities.hpp`](CCameraPresentationUtilities.hpp), [`CCameraProjectionUtilities.hpp`](CCameraProjectionUtilities.hpp), [`CCameraTextUtilities.hpp`](CCameraTextUtilities.hpp), [`CCameraViewportOverlayUtilities.hpp`](CCameraViewportOverlayUtilities.hpp), [`CCameraControlPanelUiUtilities.hpp`](CCameraControlPanelUiUtilities.hpp), and [`CCameraScriptVisualDebugOverlayUtilities.hpp`](CCameraScriptVisualDebugOverlayUtilities.hpp).

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
| apply one absolute rigid pose request at runtime | `camera->manipulate({}, &referenceFrame)` |
| set exact position or exact orientation on `Free` and `FPS` | `referenceFrame` built from `camera->getGimbal()` |
| set one absolute typed state that can be reused later | `CCameraGoal` + `CCameraGoalSolver` |
| move a camera from live input this frame | `ICamera::manipulate(...)` |
| convert keyboard or mouse input into camera commands | `IGimbalInputProcessor` or `CGimbalInputBinder` |
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

### 1. Apply one absolute rigid pose request

Use this when you already have one rigid transform and want the camera to consume it through the normal runtime entry point.

```cpp
const auto referenceFrame =
    hlsl::CCameraMathUtilities::composeTransformMatrix(desiredPosition, desiredOrientation);

camera->manipulate({}, &referenceFrame);
```

#### Why not just expose `setPosition(...)` and `setOrientation(...)` everywhere?

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

The same pattern applies to every camera family:

- `Free` keeps the rigid pose directly
- `FPS` legalizes to upright `position + pitch/yaw`
- target-relative cameras legalize to `target + orbitUv + distance`
- `Path Rig` legalizes to `PathState`

That is why `camera->manipulate({}, &referenceFrame)` is the shared absolute runtime path.

It accepts one rigid pose request at the API boundary and lets each camera family legalize it according to its own runtime model.

Use this path for:

- one-shot runtime pose application
- ImGuizmo
- world-space or local-space pose anchoring

### 2. Set exact position or exact orientation on `Free` and `FPS`

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

Do not describe this path as exact position-only or exact orientation-only for constrained target-relative or path cameras. Those cameras legalize the rigid pose request into their own family state.

### 3. Set one absolute typed state

Use this when the state should survive beyond one frame or should be reused by presets, follow, playback, persistence, or scripts.

```cpp
core::CCameraGoal goal = {};
goal.position = desiredPosition;
goal.orientation = desiredOrientation;

core::CCameraGoalSolver solver;
auto apply = solver.applyDetailed(camera.get(), goal);
```

Rule of thumb:

- use `referenceFrame` for one runtime rigid pose request now
- use `CCameraGoal` for one typed camera state that should be stored, compared, serialized, replayed, or applied later

### 4. Set one absolute camera-family state

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

### 5. Live runtime camera control

Use this when keyboard, mouse, or ImGuizmo should move the camera right now.

```cpp
auto camera = core::make_smart_refctd_ptr<COrbitCamera>(eye, target);

ui::CGimbalInputBinder binder;
ui::CCameraInputBindingUtilities::applyDefaultCameraInputBindingPreset(binder, *camera);

auto collected = binder.collectVirtualEvents(timestamp, {
    .mouseEvents = { mouseEvents.data(), mouseEvents.size() },
    .keyboardEvents = { keyEvents.data(), keyEvents.size() }
});

camera->manipulate(collected.events);
```

What happens here:

1. device input is converted into semantic camera commands
2. the camera consumes those commands through `manipulate(...)`
3. the camera updates its gimbal pose

Main types involved:

- [`CVirtualGimbalEvent.hpp`](CVirtualGimbalEvent.hpp)
- [`IGimbalBindingLayout.hpp`](IGimbalBindingLayout.hpp)
- [`IGimbalInputProcessor.hpp`](IGimbalInputProcessor.hpp)
- [`CGimbalInputBinder.hpp`](CGimbalInputBinder.hpp)
- [`CCameraInputBindingUtilities.hpp`](CCameraInputBindingUtilities.hpp)
- [`ICamera.hpp`](ICamera.hpp)

The controller-side stack is:

- `IGimbalBindingLayout` for the static mapping from device inputs to virtual events
- `IGimbalInputProcessor` for converting one frame of raw input into event magnitudes
- `CGimbalInputBinder` for the common runtime object that owns a layout and collects one frame of events
- `CCameraInputBindingUtilities` for shared preset layouts such as default `FPS`, `Orbit`, or `Path Rig` bindings

#### How do I bind `FPS` to `WASD`?

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

#### How do I make my own bindings?

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

#### How are `magnitude` values generated?

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

### 6. Capture a camera and restore it later

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

Main types involved:

- [`CCameraGoal.hpp`](CCameraGoal.hpp)
- [`CCameraGoalSolver.hpp`](CCameraGoalSolver.hpp)

### 7. Save a named camera state

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

Main types involved:

- [`CCameraPreset.hpp`](CCameraPreset.hpp)
- [`CCameraPresetFlow.hpp`](CCameraPresetFlow.hpp)

### 8. Make a camera follow a moving target

Use this when one tracked subject should drive camera behavior.

```cpp
core::CTrackedTarget trackedTarget(position, orientation);

core::SCameraFollowConfig follow = {};
follow.enabled = true;
follow.mode = core::ECameraFollowMode::LookAtTarget;

core::CCameraGoalSolver solver;
core::CCameraFollowUtilities::applyFollowToCamera(solver, camera.get(), trackedTarget, follow);
```

Main types involved:

- [`CCameraFollowUtilities.hpp`](CCameraFollowUtilities.hpp)
- [`CCameraFollowRegressionUtilities.hpp`](CCameraFollowRegressionUtilities.hpp)

### 9. Build and evaluate scripted runtime payloads

Use this when camera playback is authored as compact camera-domain data and then evaluated through generic per-frame runtime payloads and checks.

```cpp
system::CCameraScriptedTimeline timeline;
system::CCameraScriptedRuntimeUtilities::finalizeScriptedTimeline(timeline);
```

Main types involved:

- [`CCameraSequenceScript.hpp`](CCameraSequenceScript.hpp)
- [`CCameraSequenceScriptPersistence.hpp`](CCameraSequenceScriptPersistence.hpp)
- [`CCameraScriptedRuntime.hpp`](CCameraScriptedRuntime.hpp)
- [`CCameraScriptedCheckRunner.hpp`](CCameraScriptedCheckRunner.hpp)

## Core concepts

### `CVirtualGimbalEvent`

File:

- [`CVirtualGimbalEvent.hpp`](CVirtualGimbalEvent.hpp)

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

Files:

- [`IGimbal.hpp`](IGimbal.hpp)
- [`ICamera.hpp`](ICamera.hpp)

The gimbal stores runtime pose:

- position
- orientation
- scale
- orthonormal basis

It also accumulates one frame of semantic events into a `VirtualImpulse`.

`ICamera::CGimbal` extends the base gimbal with a cached world-to-view matrix.

Every runtime camera owns one `CGimbal`.

### `ICamera`

File:

- [`ICamera.hpp`](ICamera.hpp)

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

Files:

- [`ICamera.hpp`](ICamera.hpp)
- [`IGimbal.hpp`](IGimbal.hpp)

`referenceFrame` is the optional rigid transform passed to `ICamera::manipulate(...)`.

It is the runtime pose anchor for one manipulation step.

Typical producers:

- ImGuizmo
- restore helpers
- replay helpers
- code that wants world-space or local-space manipulation anchored to a specific rigid transform

When you already have one absolute rigid pose, `referenceFrame` is the direct runtime entry point for requesting that pose through the runtime camera path.

See Quick start sections 1 and 2 for the concrete absolute-pose usage patterns.

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

File:

- [`SCameraRigPose.hpp`](SCameraRigPose.hpp)

`SCameraRigPose` stores only:

- world-space position
- world-space orientation

It is the smallest typed pose object reused across the stack.

### `CCameraGoal`

File:

- [`CCameraGoal.hpp`](CCameraGoal.hpp)

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

When you want to set a camera absolutely in a reusable, serializable, or comparable way, `CCameraGoal` is the main public state object for that job.

It is not:

- a live input object
- a replacement for `manipulate(...)`
- a promise that every camera can represent every arbitrary pose exactly

For constrained cameras, the solver may project the goal onto legal camera-family state before or during apply.

### `CCameraGoalSolver`

File:

- [`CCameraGoalSolver.hpp`](CCameraGoalSolver.hpp)

`CCameraGoalSolver` converts between typed camera state and runtime cameras.

It captures runtime cameras into `CCameraGoal`, analyzes whether a target camera can represent that goal directly, and applies the result either through typed state or through runtime replay when needed.

If you want to restore one absolute camera state and you are not sure which family-specific hook to call, use `CCameraGoalSolver`.

### `CCameraPreset`

File:

- [`CCameraPreset.hpp`](CCameraPreset.hpp)

`CCameraPreset` is a named saved `CCameraGoal`.

It contains:

- `name`
- `identifier`
- `goal`

### `CCameraKeyframeTrack`

File:

- [`CCameraKeyframeTrack.hpp`](CCameraKeyframeTrack.hpp)

`CCameraKeyframeTrack` is a sequence of time-stamped presets.

Each keyframe contains:

- one preset
- one authored time

### `CCameraPlaybackTimeline`

File:

- [`CCameraPlaybackTimeline.hpp`](CCameraPlaybackTimeline.hpp)

`CCameraPlaybackTimeline` stores playback cursor state over time-based camera data.

It tracks things such as:

- current time
- direction
- looping
- paused or playing state

### `CTrackedTarget`

File:

- [`CCameraFollowUtilities.hpp`](CCameraFollowUtilities.hpp)

`CTrackedTarget` is the reusable tracked subject used by follow.

It owns its own gimbal.
It is not a mesh id and not a scene-node handle.

### `CCameraSequenceScript`

File:

- [`CCameraSequenceScript.hpp`](CCameraSequenceScript.hpp)

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

File:

- [`CCameraScriptedRuntime.hpp`](CCameraScriptedRuntime.hpp)

`CCameraScriptedRuntime` is the expanded executable form used during scripted playback and validation.

It stores runtime payloads such as:

- low-level input events
- goal and tracked-target events
- per-frame checks
- capture scheduling

Consumer-specific UI actions stay outside this shared runtime payload.

### `Path Rig`

Files:

- [`CPathCamera.hpp`](CPathCamera.hpp)
- [`CCameraPathUtilities.hpp`](CCameraPathUtilities.hpp)
- [`CCameraPathMetadata.hpp`](CCameraPathMetadata.hpp)

`Path Rig` is the camera family with typed state:

- `s`
- `u`
- `v`
- `roll`

Its runtime and typed tooling are driven by `SCameraPathModel`, which defines how path state is resolved, updated, and converted back into camera pose.

## Camera families

### Free cameras

Files:

- [`CFPSCamera.hpp`](CFPSCamera.hpp)
- [`CFreeLockCamera.hpp`](CFreeLockCamera.hpp)

State:

- world-space position
- orientation or FPS-constrained yaw/pitch orientation

Typical use:

- free-fly navigation
- direct pose-driven manipulation

### Target-relative cameras

Base:

- [`CSphericalTargetCamera.hpp`](CSphericalTargetCamera.hpp)

Derived:

- [`COrbitCamera.hpp`](COrbitCamera.hpp)
- [`CArcballCamera.hpp`](CArcballCamera.hpp)
- [`CTurntableCamera.hpp`](CTurntableCamera.hpp)
- [`CTopDownCamera.hpp`](CTopDownCamera.hpp)
- [`CIsometricCamera.hpp`](CIsometricCamera.hpp)
- [`CChaseCamera.hpp`](CChaseCamera.hpp)
- [`CDollyCamera.hpp`](CDollyCamera.hpp)
- [`CDollyZoomCamera.hpp`](CDollyZoomCamera.hpp)

Shared state:

- target position
- `orbitUv`
- distance

These cameras resolve pose through target-relative state instead of arbitrary free pose.

### DollyZoom

File:

- [`CDollyZoomCamera.hpp`](CDollyZoomCamera.hpp)

This camera adds dynamic perspective state on top of target-relative state.

Typed dynamic perspective state:

- `baseFov`
- `referenceDistance`

### Path Rig

Files:

- [`CPathCamera.hpp`](CPathCamera.hpp)
- [`CCameraPathUtilities.hpp`](CCameraPathUtilities.hpp)
- [`CCameraPathMetadata.hpp`](CCameraPathMetadata.hpp)

Typed path state:

- `s`
- `u`
- `v`
- `roll`

Typed path limits:

- `minU`
- `minDistance`
- `maxDistance`

## Typed tooling

The key typed types are introduced in the `Core concepts` section above.

This section focuses on how they fit together in one workflow:

1. `SCameraRigPose` is the smallest typed pose fragment.
2. `CCameraGoal` is the canonical typed state transport built on top of pose and optional family-specific fragments.
3. `CCameraGoalSolver` captures runtime cameras into goals and applies goals back to runtime cameras.
4. `CCameraPreset` gives one goal a stable user-facing identity.
5. `CCameraKeyframeTrack` stores presets over authored time.
6. `CCameraPlaybackTimeline` stores playback cursor state while a track is being evaluated.

Use this layer when camera state must outlive the current frame or be exchanged between tools.

## Follow

Files:

- [`CCameraFollowUtilities.hpp`](CCameraFollowUtilities.hpp)
- [`CCameraFollowRegressionUtilities.hpp`](CCameraFollowRegressionUtilities.hpp)

Follow is built from:

- one tracked target
- one follow mode
- one follow configuration

Tracked target type:

- `CTrackedTarget`

Follow modes:

- `OrbitTarget`
- `LookAtTarget`
- `KeepWorldOffset`
- `KeepLocalOffset`

`CCameraFollowUtilities` reads tracked-target pose, builds resulting camera goal state, and applies it through the shared goal solver.

## Scripting

### Compact authored format

Files:

- [`CCameraSequenceScript.hpp`](CCameraSequenceScript.hpp)
- [`CCameraSequenceScriptPersistence.hpp`](CCameraSequenceScriptPersistence.hpp)

This layer stores authored camera-domain data.

### Expanded runtime format

Files:

- [`CCameraScriptedRuntime.hpp`](CCameraScriptedRuntime.hpp)
- [`CCameraScriptedCheckRunner.hpp`](CCameraScriptedCheckRunner.hpp)

This layer stores executable per-frame runtime payloads and validation checks.

Common flow:

```text
compact authored sequence
  -> compile or expand
  -> scripted runtime payload
  -> execute against runtime camera state
```

## Projection and presentation helpers

Projection layer:

- [`IProjection.hpp`](IProjection.hpp)
- [`ILinearProjection.hpp`](ILinearProjection.hpp)
- [`IPerspectiveProjection.hpp`](IPerspectiveProjection.hpp)
- [`IPlanarProjection.hpp`](IPlanarProjection.hpp)
- [`CLinearProjection.hpp`](CLinearProjection.hpp)
- [`CPlanarProjection.hpp`](CPlanarProjection.hpp)
- [`CCubeProjection.hpp`](CCubeProjection.hpp)

Camera-facing presentation helpers:

- [`CCameraPresentationUtilities.hpp`](CCameraPresentationUtilities.hpp)
- [`CCameraProjectionUtilities.hpp`](CCameraProjectionUtilities.hpp)
- [`CCameraTextUtilities.hpp`](CCameraTextUtilities.hpp)
- [`CCameraViewportOverlayUtilities.hpp`](CCameraViewportOverlayUtilities.hpp)
- [`CCameraControlPanelUiUtilities.hpp`](CCameraControlPanelUiUtilities.hpp)
- [`CCameraScriptVisualDebugOverlayUtilities.hpp`](CCameraScriptVisualDebugOverlayUtilities.hpp)

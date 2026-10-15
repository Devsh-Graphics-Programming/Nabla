# Shared Camera API

This directory contains the reusable Nabla camera stack.

It is the runtime face: moving cameras during a frame, and reading or writing the state a camera owns.
It is centered on [`ICamera.hpp`](ICamera.hpp).

## TL;DR

If you want to know which type to touch first, use this table.

| I want to... | Use |
|---|---|
| move a camera from live input this frame | `ICamera::manipulate(...)` |
| convert keyboard or mouse input into a control frame | `CCameraMouseKeyboardController` with `CCameraMouseKeyboardPresets` |
| pair one camera with one or more projections | `CCameraWithProjections` holding `CPlanarProjection` entries |
| apply one absolute rigid pose request at runtime | `camera->setPose(...)` |
| set exact position or exact orientation on `Free` and `FPS` | `SCameraRigPose` built from `camera->getGimbal()` |
| use a camera in a right-handed or Z-up world | the `SWorldFrame` constructor argument, see [Frames and handedness](#frames-and-handedness) |
| aim a camera at a point | `camera->setLookAt(position, target, up)` |
| capture, store, replay or script camera state | moved to the 61_UI example, see below |
| use the path-rig camera | `CPathCamera` and `SCameraPathModel` |

## Quick start

This section shows the common entry points before any deeper explanation.

### 1. Live runtime camera control

Use this when keyboard and mouse should move the camera right now.

```cpp
auto camera = core::make_smart_refctd_ptr<COrbitCamera>(eye, target);

CCameraMouseKeyboardController controller;
controller.binding = CCameraMouseKeyboardPresets::makeDefaultBinding(*camera);

const auto controls = controller.collect(nextPresentationTimestamp,
    { keyEvents.data(), keyEvents.size() },
    { mouseEvents.data(), mouseEvents.size() });

camera->manipulate(controls);
```

What happens here:

1. the controller turns one frame of `ui::SKeyboardEvent` and `ui::SMouseEvent` into one `SCameraControls`, in world units and radians
2. the camera applies the frame through `manipulate(...)`
3. the camera updates its gimbal pose

`manipulate(...)` takes only the frame. Nothing about it depends on where the frame came from, so the same call serves:

- the mouse/keyboard controller above
- a frame built by hand, in a script or a test:

```cpp
SCameraControls controls = {};
controls.rotate.y = hlsl::radians(15.0);   // yaw by 15 degrees
controls.distance = -0.5;                  // half a unit closer
camera->manipulate(controls);
```

- a frame built by any controller you write, for a gamepad, a network stream or a recording, as long as it fills `SCameraControls`

**Question: Which controls does a camera accept?**

`camera->getAcceptedControls()` returns an `ECameraControlAxis` mask. A frame that sets any axis outside it is refused whole: `manipulate` returns `false` and changes nothing. Each camera's header states what every accepted axis means for that rig: the frame a translation is applied in, the pivot of a rotation, the clamp. `SCameraControls::masked(mask)` zeroes the rest when one frame is sent to several cameras.

**Question: How do I bind `FPS` to `WASD`?**

Use the default binding for the camera kind.

```cpp
auto camera = core::make_smart_refctd_ptr<CFPSCamera>(position, orientation);

CCameraMouseKeyboardController controller;
controller.binding = CCameraMouseKeyboardPresets::makeDefaultBinding(*camera);
```

For `FPS`, the default binding gives you:

- keyboard `W/S` -> `translate.z`, `D/A` -> `translate.x`
- keyboard `K/I` -> `rotate.x`, `L/J` -> `rotate.y`
- mouse X -> `rotate.y`, mouse Y -> `rotate.x`

For `Free`, the default binding adds `E/Q` for `rotate.z`.

Target-relative families and `Path Rig` keep the same physical inputs on the axes they accept.

**Question: How do I define custom bindings?**

A binding has one slot per control axis. Fill the slots you want.

```cpp
CCameraMouseKeyboardController controller;
auto& binding = controller.binding;

binding[ECameraControlAxis::TranslateZ] = { .positiveKey = ui::EKC_W, .negativeKey = ui::EKC_S, .keyRate = 2.0 };   // 2 units per second held
binding[ECameraControlAxis::TranslateX] = { .positiveKey = ui::EKC_D, .negativeKey = ui::EKC_A, .keyRate = 2.0 };
binding[ECameraControlAxis::RotateY].mouseMovementGain = { 0.003, 0.0 };                                             // radians per mouse count along X
binding[ECameraControlAxis::RotateX].mouseMovementGain = { 0.0, 0.003 };
binding[ECameraControlAxis::RotateY].mouseMovementGate = ui::EMB_RIGHT_BUTTON;                                        // only while the right button is held
binding[ECameraControlAxis::RotateX].mouseMovementGate = ui::EMB_RIGHT_BUTTON;
binding[ECameraControlAxis::Distance].mouseScrollGain = { -0.1, 0.0 };                                                // scroll up moves closer
```

Sensitivity lives in the binding, not in the camera. A camera has no speed.

**Question: My world is right-handed, or Z-up. What changes?**

Every camera constructor takes the world's `SWorldFrame` as its last argument, default `yUpLeftHanded()`. The
frame is the up axis, the direction a zero yaw or azimuth refers to, and the handedness; `yUpRightHanded()` and
`zUpRightHanded()` are the other built-ins and `SWorldFrame::tryMake(up, forward, handedness)` takes any pair.
Pair the camera with the projection of the frame's handedness; the view matrix is the same call either way.

```cpp
const auto frame = SWorldFrame::zUpRightHanded();                            // +Z up, +Y forward, +X right
auto camera = core::make_smart_refctd_ptr<CFPSCamera>(position, quaternion::identity(), frame);
camera->setLookAt(position, target, frame.up);

const auto view = camera->getGimbal().getViewMatrix();                       // equals hlsl::math::linalg::rhLookAt(position, target, frame.up)
const auto projection = hlsl::math::thin_lens::rhPerspectiveFovMatrix<float>(fov, aspect, zNear, zFar);
```

Controls mean the same in every frame: `translate.x` is the camera's right, `translate.z` where it looks,
`rotate.y` turns toward its right about the frame's up, `rotate.x` tilts toward its down. See
[Frames and handedness](#frames-and-handedness) for what the frame does and does not touch.

TODO: revisit. How the controller turns held keys and event timestamps into an amount of motion
### 2. Projection is separate from camera state

**Question: Where is `setProjectionMatrix(...)`?**

There is no `camera->setProjectionMatrix(...)`.

That is intentional.

The camera API keeps runtime camera state and projection state separate:

- `ICamera` owns pose and motion state
- `CPlanarProjection` owns projection state
- `CCameraWithProjections` pairs one camera with several projections and gives the view, projection and
  view-projection matrices

This keeps the pairing flexible:

- one camera can be reused with different projection entries
- one viewport can switch projection preset without replacing the camera
- projection parameters such as FOV, orthographic width, near, and far do not have to live inside every camera kind

The split looks like this:

```cpp
auto camera = core::make_smart_refctd_ptr<COrbitCamera>(eye, target);

auto cameraWithProjections = CCameraWithProjections::create(core::smart_refctd_ptr(camera));

auto& projections = cameraWithProjections->getProjections();
projections.push_back(CPlanarProjection::createPerspective(0.1f, 100.0f, 60.0f));
projections.push_back(CPlanarProjection::createOrthographic(0.1f, 100.0f, 10.0f));

// the matrix is rebuilt from the parameters only here, so call it when the viewport or parameters change
projections[0].update(leftHanded, aspectRatio);

const auto view = cameraWithProjections->getViewMatrix();
const auto proj = cameraWithProjections->getProjectionMatrix(0u);
const auto viewProj = cameraWithProjections->getViewProjectionMatrix(0u);
```

So the camera does not own projection parameters.

Instead:

- the camera owns `view`
- the projection entry owns `projection`
- the wrapper combines both into `viewProjection`; multiplying in a model matrix is left to the caller

When you want to change projection state, touch the projection layer:

- `CPlanarProjection::setPerspective(...)`
- `CPlanarProjection::setOrthographic(...)`
- `CPlanarProjection::update(...)`

When you want to change pose or camera-family state, touch the camera layer:

- `ICamera::manipulate(...)`
- `ICamera::setPose(...)`
- family-specific typed hooks such as `trySetSphericalTarget(...)` or `trySetPathState(...)`

### 3. Apply one absolute rigid pose request

Use this when you already have one rigid transform and want the camera to consume it through the normal runtime entry point.

```cpp
const auto rigidFrame =
    CCameraMathUtilities::composeTransformMatrix(desiredPosition, desiredOrientation);

if (camera->setPose(rigidFrame))
{
    // the pose was accepted and applied
}
```

`setPose(...)` can return `false`.

Common reasons are:

- the supplied transform was not a valid rigid orthonormal transform
- the concrete camera kind could not project the pose onto its own runtime state

`manipulate(...)` returns `false` when the frame is all zero, is not finite or sets an axis the camera does not accept, or when the resulting gimbal pose is the one the call started with.

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
    hlsl::math::quaternion<hlsl::float64_t>::createFromYawPitchRoll(
        hlsl::radians(40.0), hlsl::radians(-15.0), hlsl::radians(25.0));
```

The requested rigid pose contains `roll = 25 deg`.

That roll is not legal for `FPS`.

If the API exposed unrestricted `setOrientation(...)` and accepted that quaternion as-is, the runtime camera would no longer match the rules of the `FPS` rig.

The current API does this instead:

1. accept one rigid pose request through `setPose(...)`
2. project that pose onto the legal state space of the concrete camera kind
3. rebuild the final runtime pose from that legal state

For `FPS` that means:

- keep the requested position
- read forward direction from the requested pose
- rebuild legal `pitch/yaw`
- drop arbitrary roll
- write back one upright `FPS` pose

`CFPSCamera` advertises only translation plus pitch/yaw runtime control, so `RollLeft` and `RollRight` events are ignored by the `FPS` accumulator.

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

const auto rigidFrame =
    CCameraMathUtilities::composeTransformMatrix(newPosition, keepOrientation);

camera->setPose(rigidFrame);
```

```cpp
const auto& gimbal = camera->getGimbal();

const auto keepPosition = gimbal.getPosition();
const auto newOrientation = desiredOrientation;

const auto rigidFrame =
    CCameraMathUtilities::composeTransformMatrix(keepPosition, newOrientation);

camera->setPose(rigidFrame);
```

`Free` applies these requests exactly.

`FPS` keeps the exact position but legalizes orientation to its upright `pitch/yaw` state.

For constrained target-relative and path cameras, prefer family-specific typed state instead of describing this as an exact component setter.

### 5. Set one absolute camera-family state

Use this when you do not want a generic rigid pose and instead want to write the native state of one camera family.

Target-relative cameras:

```cpp
camera->trySetSphericalTarget(targetPosition);
camera->trySetSphericalDistance(distance);
```

Path camera:

```cpp
ICamera::PathState path = {
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

## Frames and handedness

A frame is three perpendicular unit vectors with names. Three of them matter here.

- **World frame.** The user's X, Y, Z. Positions, targets and the scene live in it. Its handedness, whether
  `X x Y` is `+Z` or `-Z`, is what `ECameraHandedness` records. `SWorldFrame` adds the two conventions a camera
  needs on top of that: which axis is up (the horizon the FPS rig keeps level, the pole the orbit rigs turn
  about) and which direction a zero yaw or azimuth refers to (`forward`); `right` follows from the two by the
  handedness. Angles in the extension are measured in this frame.
- **Camera local frame, or view space.** The world axes rotated by the camera's quaternion: local X is `q * X`,
  local Y is `q * Y`, local Z is `q * Z`. The columns of the quaternion's rotation matrix are these three vectors.
  The view matrix maps world coordinates into this frame and the projection consumes them from it. A projection
  has a convention about where "in front" is: `lhPerspective` looks along +Z of this frame, `rhPerspective`
  against it.
- **Physical frame, `SCameraBasis`.** Right, up and forward as world-space directions, where forward is where
  the camera actually looks. Rigs and users think in this frame: `W` moves along forward, a look-at aims forward
  at the target.

The two camera frames share X and Y: right is local X and up is local Y in both worlds. They differ only in Z,
because a quaternion holds a rotation (`det = +1`) and `(right, up, forward)` is a rotation only in a left-handed
world. In a right-handed world the quaternion holds `(right, up, -forward)` instead. So

```text
forward = local Z          left-handed
forward = -local Z         right-handed
```

which is the convention of the LH/RH projection matrices and of `hlsl::math::linalg::lhLookAt` / `rhLookAt`.
With the identity quaternion the view matrix is the identity in both worlds; the camera looks along +Z in a
left-handed world and along -Z in a right-handed one, because that is where each projection puts "in front".

**The rule.** Code needs the handedness exactly when it relates a physical direction, forward or right, to the
quaternion or its frame. Code needs the world frame when it turns angles into directions or back: yaw and
pitch, azimuth and elevation, the horizon plane. Code that stays inside the quaternion's frame (pose, world
matrix, view matrix, decomposing a matrix) or inside the physical frame (moving along the basis, rotating one
basis vector toward another) is the same in every world and takes neither.

Functions that take an `ECameraHandedness` or an `SWorldFrame`, keep this list current when adding one:

| Function | Takes | What it decides there |
|---|---|---|
| `CCameraMathUtilities::orientationFromBasis` | handedness | whether `forward` or `-forward` goes on local Z |
| `CCameraMathUtilities::basisFromOrientation` | handedness | whether forward is read from local Z or `-local Z` |
| `CCameraMathUtilities::tryBuildBasis` | frame | which cross product is the physical right; the frame's up and forward stand in for a degenerate up hint |
| `CCameraMathUtilities::tryCreateQuaternionFromLookAt` | frame | builds a basis, so the above |
| `CCameraMathUtilities::tryBuildPoseFromOrbit`, `tryBuildOrbitFromPosition` | frame | azimuth and elevation are measured about the frame's up from its forward |
| `CCameraPathUtilities::tryBuildPathStateFromPosition`, `tryResolvePathState`, `tryBuildPathPoseFromState`, `tryBuildCanonicalPathState`, `makeDefaultPathModel` | frame | the path's radius lies in the frame's horizon plane, its height along the up |
| `CTopDownCamera::resolveTopDownYaw` | frame | the azimuth of an orientation about the frame's up |
| `CFPSCamera` | frame (via `getFrame()`) | yaw and pitch are measured in it, the up hint is its up |
| `CChaseCamera` | frame (via `getFrame()`) | the target moves in the frame's horizon plane |
| `ICamera` and every rig constructor | frame | fix the camera's frame for its lifetime; `getFrame()` and `getHandedness()` read it |
| `ICamera::getBasis`, `ICamera::setLookAt` | handedness, frame | the physical frame of the camera, and aiming it |
| `CPlanarProjection::update(leftHanded, ...)` | handedness | the projection side; it has to agree with the camera |

`CCameraMathUtilities::getPitchYawRollRadians` and the Euler helpers around it decompose the rotation about the world
Y, X and Z axes (the inverse of `createFromYawPitchRoll`); they take no handedness, but their angles describe local +Z,
so they read as the look direction's heading and pitch only for a left-handed camera.

Functions that look like they should take one and do not: `IGimbal` and `CCameraGimbal` (pose, world and view
matrices), `composeTransformMatrix`, `tryExtractPositionAndQuaternionFromTransform`, `setPose(float64_t4x4)`,
`directionFromYawPitch` and the angle helpers next to it (the caller passes the frame's axes), `SCameraControls`
and the controller. They never mention forward or right.

The rigs themselves contain no sign: they read their physical axes through `getBasis()`, move and turn in that
frame, and write the result back through `orientationFromBasis`. `CFreeCamera` is the simplest example.

## Core concepts

### `SCameraControls`

Defined in [`SCameraControls.hpp`](SCameraControls.hpp).

`SCameraControls` is one frame of physical deltas: `translate` in world units, `rotate` in radians (x pitch,
y yaw, z roll), `distance` in world units along the camera-target line, and `path.s`, `path.u`, `path.v`,
`path.roll` in the units the active path model defines. `ECameraControlAxis` names each field with one bit; the
group masks `Translate`, `Rotate`, `Path` and `AllControls` combine them.

The frame does not store where it came from. The same struct is filled by the mouse/keyboard controller, by
scripts, by solvers and by tests.

### `IGimbal`

Defined in [`IGimbal.hpp`](IGimbal.hpp) and used by [`ICamera.hpp`](ICamera.hpp).

The gimbal stores the runtime pose plus a manipulation counter:

- position
- orientation
- number of manipulations that changed the pose

The world matrix is derived from the pose on each call. Each setter returns whether the stored pose changed, and
a changed pose advances the counter by one. The gimbal takes no handedness; the camera's physical axes are read
through `ICamera::getBasis()`.

`CCameraGimbal` in [`CCameraGimbal.hpp`](CCameraGimbal.hpp) extends it with the world-to-view matrix.
That matrix is rebuilt on the first read after a manipulation and cached against the counter until the next
one, so a burst of writes costs one rebuild. Reading mutates the cache, so one instance must not be read from
several threads at once.

Every runtime camera owns one `CCameraGimbal`.

### `ICamera`

Defined in [`ICamera.hpp`](ICamera.hpp).

`ICamera` is the shared runtime interface implemented by every camera kind.

Its main job is:

- apply one frame of `SCameraControls` along the axes it accepts
- optionally take one rigid pose
- update internal camera state
- update runtime pose in the gimbal

Important members:

- `manipulate(...)`
- `getAcceptedControls()`
- `getGimbal()`
- `getHandedness()`, `getBasis()` and `setLookAt(...)`, see [Frames and handedness](#frames-and-handedness)
- `getKind()`
- `getCapabilities()`
- typed hooks such as `tryGetSphericalTargetState(...)` and `tryGetPathState(...)`

A camera stores no speed. Sensitivity belongs to whatever fills the frame.

### `setPose`

Defined by [`ICamera.hpp`](ICamera.hpp).

`setPose(...)` takes one authored world-space pose, either as `SCameraRigPose` or as a rigid `float64_t4x4`.

Typical producers:

- ImGuizmo
- restore helpers
- replay helpers
- code that wants to place a camera at a specific rigid transform

See Quick start sections 1 to 4 for the concrete runtime usage patterns.

Shared runtime pattern:

```text
setPose
  -> decompose the rigid transform
  -> project onto the legal state of this camera kind
  -> rebuild the gimbal pose

manipulate
  -> refuse the frame if it sets an axis the rig does not accept
  -> apply the accepted axes in that state space
  -> rebuild the gimbal pose
```

### `SCameraRigPose`

Defined in [`SCameraTypes.hpp`](SCameraTypes.hpp), next to `STargetOrbit` and `SCameraBasis`.

`SCameraRigPose` stores only:

- world-space position
- world-space orientation: local X is the camera's right, local Y its up, local Z its look direction in a
  left-handed world and the opposite of it in a right-handed one

It is the smallest typed pose object reused across the stack.

### `Path Rig`

Defined by [`CPathCamera.hpp`](CPathCamera.hpp), [`CCameraPathUtilities.hpp`](CCameraPathUtilities.hpp), and [`CCameraPathMetadata.hpp`](CCameraPathMetadata.hpp).

`Path Rig` is the camera family with typed state:

- `s`
- `u`
- `v`
- `roll`

Its runtime and typed tooling are driven by `SCameraPathModel`, which defines how path state is resolved, updated, and converted back into camera pose.

At the API boundary, you can think of `Path Rig` as one parametric camera map that turns typed path state into pose.

In other words, the reusable seam is not "one built-in rail camera". It is:

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
(R(s), U(s), F(s))
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
s &= \mathrm{wrap}(\mathrm{atan2}(\Delta_z, \Delta_x)) \\
u &= \max(u_{\min}, \sqrt{\Delta_x^2 + \Delta_z^2}) \\
v &= \Delta_y
\end{aligned}
$$

The default model also derives one radial camera distance from `(u, v)`:

$$
d = \sqrt{u^2 + v^2}
$$

and sanitizes state as:

$$
\begin{aligned}
s &\leftarrow \mathrm{wrap}(s) \\
u &\leftarrow \max(u_{\min}, u) \\
\rho &\leftarrow \mathrm{wrap}(\rho)
\end{aligned}
$$

The base orientation is then built from the camera looking from the resolved position back at the target, and the authored roll is applied around that resulting forward axis.

The built-in control law maps runtime local motion into path-state delta as:

$$
\Delta q = (\Delta s, \Delta u, \Delta v, \Delta \rho)^{\mathsf{T}}
= (\Delta z_{\mathrm{local}}, \Delta x_{\mathrm{local}}, \Delta y_{\mathrm{local}}, \Delta \mathrm{roll})^{\mathsf{T}}
$$

and integrates it as:

$$
q_{n+1} = \mathrm{sanitize}(q_n + \Delta q)
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

- [`CFreeCamera.hpp`](CFreeCamera.hpp) and [`CFPSCamera.hpp`](CFPSCamera.hpp) store world-space position plus free or FPS-constrained orientation.
- Target-relative cameras are built on [`CSphericalTargetCamera.hpp`](CSphericalTargetCamera.hpp) and include [`COrbitCamera.hpp`](COrbitCamera.hpp), [`CArcballCamera.hpp`](CArcballCamera.hpp), [`CTurntableCamera.hpp`](CTurntableCamera.hpp), [`CTopDownCamera.hpp`](CTopDownCamera.hpp), [`CIsometricCamera.hpp`](CIsometricCamera.hpp), [`CChaseCamera.hpp`](CChaseCamera.hpp), [`CDollyCamera.hpp`](CDollyCamera.hpp), and [`CDollyZoomCamera.hpp`](CDollyZoomCamera.hpp). They store target position, `orbitUv`, and distance instead of arbitrary free pose.
- [`CDollyZoomCamera.hpp`](CDollyZoomCamera.hpp) extends the target-relative family with dynamic perspective state `baseFov` and `referenceDistance`.
- [`CPathCamera.hpp`](CPathCamera.hpp) uses the parametric path-state seam described above together with limits `minU`, `minDistance`, and `maxDistance`.

## Projections

- [`IProjection.hpp`](IProjection.hpp) is the abstract projection: `project`, `unproject` and the family it
  belongs to.
- [`CPlanarProjection.hpp`](CPlanarProjection.hpp) is the one implementation, held by value. It is
  perspective (FOV in degrees), orthographic (width), or `Custom` (a caller-provided matrix), each with near and
  far planes. Create it with `createPerspective`, `createOrthographic`, `create(SParameters)` or
  `create(float64_t4x4)`. `update(leftHanded, aspectRatio)` rebuilds the matrix from the parameters; setting
  parameters alone does not.
- [`ICameraWithProjections.hpp`](ICameraWithProjections.hpp) and
  [`CCameraWithProjections.hpp`](CCameraWithProjections.hpp) pair one camera with a list of `CPlanarProjection`
  entries, for example one per viewport or preset, and return the view, projection and view-projection matrices.
  The view matrix is for the projection of the camera's handedness, so `update(leftHanded, ...)` has to be given
  the same handedness the camera was built with.

`CDollyZoomCamera` derives its FOV from its distance, so a projection paired with it has to be told the new FOV. The extension exposes the value through `ICamera::tryGetDynamicPerspectiveFov(...)` and leaves the push into `CPlanarProjection::setPerspective(...)` to the application; 61_UI does it in `CCameraProjectionUtilities`.

## Camera tooling lives in the 61_UI example

The layer that captured one camera's state into a `CCameraGoal` and applied it to another camera, plus
everything built on it (presets, keyframe tracks, playback, persistence, follow, sequence scripts, the
scripted runtime and its checks), now lives in
[`examples_tests/61_UI/include/camera/`](../../../../examples_tests/61_UI/include/camera/).

It moved because that example was its only user and because the design is under review: a goal is the union
of every rig's internal state, so each new camera kind has to answer for fragments it does not own. The
README in that folder explains what has to be true before any of it comes back.

A few small helpers followed later for the same reason, 61_UI being their only user: syncing a dynamic
perspective FOV into a projection entry (`CCameraProjectionUtilities`), whole-file read/write (`CFileUtilities`)
and the stable string names of key codes and mouse buttons (`CInputCodeNames`).

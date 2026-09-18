// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _I_CAMERA_HPP_
#define _I_CAMERA_HPP_

#include <limits>
#include <optional>
#include <utility>

#include "nbl/core/IReferenceCounted.h"
#include "nbl/core/util/bitflag.h"
#include "SCameraToolingThresholds.hpp"
#include "CCameraGimbal.hpp"
#include "SCameraControls.hpp"

namespace nbl::ext::cameras
{

/// @brief Shared runtime camera interface.
///
/// A camera owns one pose in a `CCameraGimbal`. It changes through two entry points: `manipulate(...)`
/// applies one frame of physical deltas along the axes the rig accepts, and `setPose(...)` projects an
/// authored world-space pose onto the state the rig stores. `manipulate` does not know what filled the frame;
/// the mouse/keyboard controller, a script, a solver and a test all build the same `SCameraControls`.
///
/// The optional typed hooks expose camera-family state for code that needs
/// capture, restore, compatibility analysis, persistence, or validation.
class ICamera : virtual public core::IReferenceCounted
{
public:
    /// @brief Smallest target distance accepted by target-relative cameras; see `STargetOrbit::DefaultMinDistance`.
    static inline constexpr hlsl::float64_t DefaultMinTargetDistance = STargetOrbit::DefaultMinDistance;
    /// @brief Interim unbounded default for the largest target distance; see `STargetOrbit::DefaultMaxDistance`.
    static inline constexpr hlsl::float64_t DefaultMaxTargetDistance = STargetOrbit::DefaultMaxDistance;

    /// @brief Stable camera-family identifier used by metadata, presets, follow, and scripted helpers.
    enum class CameraKind : uint8_t
    {
        Unknown,
        FPS,
        Free,
        Orbit,
        Arcball,
        Turntable,
        TopDown,
        Isometric,
        Chase,
        Dolly,
        DollyZoom,
        Path
    };

    /// @brief Optional typed capabilities exposed by a concrete runtime camera implementation.
    enum CameraCapability : uint32_t
    {
        None = 0u,
        SphericalTarget = core::createBitmask({ 0 }),
        DynamicPerspectiveFov = core::createBitmask({ 1 })
    };

    /// @brief Typed state fragments that helper layers may capture from or apply to a camera.
    enum GoalStateMask : uint32_t
    {
        GoalStateNone = 0u,
        GoalStateSphericalTarget = core::createBitmask({ 0 }),
        GoalStateDynamicPerspective = core::createBitmask({ 1 }),
        GoalStatePath = core::createBitmask({ 2 })
    };

    using capability_flags_t = core::bitflag<CameraCapability>;
    using goal_state_flags_t = core::bitflag<GoalStateMask>;

    /// @brief Canonical target-relative state reported by spherical camera families.
    ///
    /// The state stores the tracked target position, orbit angles in `orbitUv`,
    /// and distance limits needed by tooling that wants to capture or reapply a
    /// target-relative camera pose without going through free-form setters.
    /// `maxDistance` is an optional upper bound and may be infinite when the
    /// active camera family does not impose a finite cap.
    struct SphericalTargetState
    {
        /// @brief Tracked target position in world space.
        hlsl::float64_t3 target = hlsl::float64_t3(0.0);
        /// @brief Orbit yaw and pitch around the target, expressed in radians.
        hlsl::float64_t2 orbitUv = hlsl::float64_t2(0.0);
        /// @brief Current camera-to-target distance.
        float distance = 0.f;
        /// @brief Lowest distance that remains valid for the current camera.
        float minDistance = 0.f;
        /// @brief Highest distance that remains valid for the current camera, or infinity when unbounded.
        float maxDistance = ICamera::DefaultMaxTargetDistance;
    };

    /// @brief Typed perspective state reported by cameras with derived FOV behavior.
    struct DynamicPerspectiveState
    {
        /// @brief Authored reference FOV in degrees.
        float baseFov = 0.f;
        /// @brief Distance at which `baseFov` should be preserved.
        float referenceDistance = 0.f;
    };

    /// @brief Limits constraining reusable `PathState` coordinates for `Path Rig` cameras.
    ///
    /// These limits are part of the typed path-model surface. They are not
    /// global engine rules. A concrete `Path Rig` instance may expose an
    /// unbounded `maxDistance` by returning infinity.
    struct PathStateLimits
    {
        /// @brief Minimal valid `u` coordinate after path-state sanitization.
        hlsl::float64_t minU = ICamera::DefaultMinTargetDistance;
        /// @brief Minimal valid radial distance derived from the `(u, v)` pair.
        hlsl::float64_t minDistance = ICamera::DefaultMinTargetDistance;
        /// @brief Maximal valid radial distance derived from the `(u, v)` pair, or infinity when unbounded.
        hlsl::float64_t maxDistance = ICamera::DefaultMaxTargetDistance;
    };

    /// @brief Parametric path-rig state used by the `Path Rig` camera kind.
    ///
    /// The built-in path model interprets `(s, u, v, roll)` as path progress,
    /// lateral shape coordinates, and roll around the local forward axis.
    /// Other path models may map the same coordinates onto different geometry.
    struct PathState
    {
        /// @brief Primary path-progress coordinate interpreted by the active path model.
        hlsl::float64_t s = 0.0;
        /// @brief First lateral/shape coordinate interpreted by the active path model.
        hlsl::float64_t u = 0.0;
        /// @brief Second lateral/shape coordinate interpreted by the active path model.
        hlsl::float64_t v = 0.0;
        /// @brief Roll around the path-model forward axis, expressed in radians.
        hlsl::float64_t roll = 0.0;

        /// @brief Pack the state into one four-component vector.
        inline hlsl::float64_t4 asVector() const
        {
            return hlsl::float64_t4(s, u, v, roll);
        }

        /// @brief Rebuild one path state from the packed vector representation.
        static inline PathState fromVector(const hlsl::float64_t4& value)
        {
            return {
                .s = value.x,
                .u = value.y,
                .v = value.z,
                .roll = value.w
            };
        }
    };

    ICamera() {}
	virtual ~ICamera() = default;

    /// @brief Return the gimbal holding the runtime camera pose.
	virtual const CCameraGimbal& getGimbal() = 0u;

    /// @brief Apply one frame of physical deltas on top of the pose currently held by the gimbal.
    ///
    /// Refused whole, with nothing changed, when a value is not finite, when every axis is zero, or when an
    /// axis outside `getAcceptedControls()` is non-zero.
    ///
    /// @return whether the resulting gimbal pose differs from the one the call started with.
    inline bool manipulate(const SCameraControls& controls)
    {
        if (!controls.isFinite())
            return false;

        const auto set = controls.nonZeroAxes();
        if (set == 0u || (set & ~getAcceptedControls()) != 0u)
            return false;

        return applyControls(controls);
    }

    /// @brief Project one authored world-space pose onto the state this rig stores, then commit it to the gimbal.
    ///
    /// Each rig keeps the part of `pose` its own state model expresses: a target-relative rig takes the
    /// position and derives the orientation from the resulting orbit angles, an FPS rig takes the position
    /// plus pitch and yaw, a fixed-angle rig takes the position alone.
    ///
    /// @return whether `pose` was accepted.
    virtual bool setPose(const SCameraRigPose& pose) = 0;

    /// @brief Decompose one rigid world-space transform (basis in the columns, translation in the last
    /// column) and apply it as a pose. Rejects non-rigid and degenerate input.
    inline bool setPose(const hlsl::float64_t4x4& rigidFrame)
    {
        SCameraRigPose pose = {};
        if (!CCameraMathUtilities::tryExtractRigidPoseFromTransform(rigidFrame, pose.position, pose.orientation))
            return false;

        return setPose(pose);
    }

    /// @brief Return the `ECameraControlAxis` mask this rig applies. Every other axis must be zero in a frame passed to `manipulate(...)`.
    virtual uint32_t getAcceptedControls() const = 0u;

    /// @brief Return the stable camera-family identifier for this concrete runtime camera.
    virtual CameraKind getKind() const = 0;
    /// @brief Return the optional typed capabilities exposed by this camera implementation.
    virtual uint32_t getCapabilities() const { return None; }
    /// @brief Return the typed goal-state fragments that helper layers may safely use with this camera.
    virtual uint32_t getGoalStateMask() const
    {
        goal_state_flags_t mask = GoalStateNone;
        if (hasCapability(SphericalTarget))
            mask |= GoalStateSphericalTarget;
        if (hasCapability(DynamicPerspectiveFov))
            mask |= GoalStateDynamicPerspective;
        return static_cast<uint32_t>(mask.value);
    }

    /// @brief Return the stable human-readable identifier for this concrete camera instance.
    virtual std::string_view getIdentifier() const = 0u;

    /// @brief Check whether the camera exposes the requested optional capability.
    inline bool hasCapability(CameraCapability capability) const
    {
        return capability_flags_t(getCapabilities()).hasFlags(capability);
    }

    /// @brief Check whether the camera can exchange the requested typed goal-state fragment.
    inline bool supportsGoalState(GoalStateMask goalState) const
    {
        return goal_state_flags_t(getGoalStateMask()).hasFlags(goalState);
    }

    // TODO: the typed goal-state hooks below have no consumer left inside this extension. They were read and
    // written only by the goal solver and the follow regression checks, which now live in the 61_UI example under
    // `examples_tests/61_UI/include/camera/`. They exist so one rig's state can be pulled out and pushed onto a
    // different rig, which is the part of that design under review. Either they follow the tooling out, or they
    // are redesigned with it around each rig's own state instead of a shared union.

    /// @brief Query the current spherical-target state when the camera exposes it.
    virtual bool tryGetSphericalTargetState(SphericalTargetState& out) const
    {
        return false;
    }

    /// @brief Replace only the tracked target position for spherical-target cameras.
    virtual bool trySetSphericalTarget(const hlsl::float64_t3& target)
    {
        return false;
    }

    /// @brief Replace only the tracked target distance for spherical-target cameras.
    virtual bool trySetSphericalDistance(float distance)
    {
        return false;
    }

    /// @brief Query the current derived dynamic perspective FOV when the camera exposes it.
    virtual bool tryGetDynamicPerspectiveFov(float& outFov) const
    {
        return false;
    }

    /// @brief Query the current authored dynamic perspective state when the camera exposes it.
    virtual bool tryGetDynamicPerspectiveState(DynamicPerspectiveState& out) const
    {
        return false;
    }

    /// @brief Replace the authored dynamic perspective state when the camera exposes it.
    virtual bool trySetDynamicPerspectiveState(const DynamicPerspectiveState& state)
    {
        return false;
    }

    /// @brief Query the current typed path state when the camera exposes it.
    virtual bool tryGetPathState(PathState& out) const
    {
        return false;
    }

    /// @brief Query the active typed limits constraining the current path state.
    virtual bool tryGetPathStateLimits(PathStateLimits& out) const
    {
        return false;
    }

    /// @brief Replace the current typed path state when the camera exposes it.
    virtual bool trySetPathState(const PathState& state)
    {
        return false;
    }

protected:
    /// @brief Apply a frame already checked against `getAcceptedControls()`.
    virtual bool applyControls(const SCameraControls& controls) = 0;
};

}

#endif // _I_CAMERA_HPP_

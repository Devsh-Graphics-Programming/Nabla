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
#include "CVirtualGimbalEvent.hpp"

namespace nbl::ext::cameras
{

/// @brief Shared runtime camera interface.
///
/// `ICamera` consumes batches of `CVirtualGimbalEvent` values and updates one
/// camera pose stored in `CCameraGimbal`. A `CVirtualGimbalEvent` identifies one
/// semantic command such as `MoveForward`, `PanLeft`, or `RollRight` and carries
/// one source-normalized scalar magnitude for that command.
///
/// Keyboard input, mouse input, ImGuizmo interaction, scripted playback,
/// preset replay, follow helpers, and goal solving all drive cameras through
/// the same `manipulate(...)` entry point.
///
/// The optional typed hooks expose camera-family state for code that needs
/// capture, restore, compatibility analysis, persistence, or validation.
class ICamera : virtual public core::IReferenceCounted
{ 
private:
    static inline constexpr hlsl::float64_t DefaultMoveSpeedScaleValue = 0.01;
    static inline constexpr hlsl::float64_t DefaultRotationSpeedScaleValue = 0.003;
    static inline constexpr hlsl::float64_t VirtualTranslationUnit = 0.01;

public:
    /// @brief Smallest target distance accepted by target-relative cameras; see `STargetOrbit::DefaultMinDistance`.
    static inline constexpr hlsl::float64_t DefaultMinTargetDistance = STargetOrbit::DefaultMinDistance;
    /// @brief Interim unbounded default for the largest target distance; see `STargetOrbit::DefaultMaxDistance`.
    static inline constexpr hlsl::float64_t DefaultMaxTargetDistance = STargetOrbit::DefaultMaxDistance;

    /// @brief Camera-local multipliers applied when semantic virtual events are converted into motion.
    ///
    /// Input binders emit virtual magnitudes. Concrete cameras multiply those
    /// magnitudes by this per-camera configuration before applying them to
    /// their own state model.
    struct SMotionConfig
    {
        /// @brief Camera-local scale applied to virtual translation magnitudes.
        double moveSpeedScale = DefaultMoveSpeedScaleValue;
        /// @brief Camera-local scale applied to virtual rotation magnitudes.
        double rotationSpeedScale = DefaultRotationSpeedScaleValue;
    };

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

        /// @brief Project the state onto the translation-style representation used by replay helpers.
        inline hlsl::float64_t3 asTranslationVector() const
        {
            return hlsl::float64_t3(u, v, s);
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

        /// @brief Rebuild one path state from the translation-style helper representation.
        static inline PathState fromTranslationVector(const hlsl::float64_t3& value, const hlsl::float64_t pathRoll = 0.0)
        {
            return {
                .s = value.z,
                .u = value.x,
                .v = value.y,
                .roll = pathRoll
            };
        }
    };

    ICamera() {}
	virtual ~ICamera() = default;

    /// @brief Return the gimbal holding the runtime camera pose.
	virtual const CCameraGimbal& getGimbal() = 0u;

    /// @brief Apply one frame of semantic virtual events on top of the pose currently held by the gimbal.
    ///
    /// `virtualEvents` stores one frame of semantic movement, rotation, and
    /// scale commands. Translation commands use `Move*`, rotation commands use
    /// `Tilt*`, `Pan*`, and `Roll*`, and scale commands use `Scale*`. Cameras
    /// interpret only the subset advertised by `getAllowedVirtualEvents()`.
    ///
    /// @return whether the resulting gimbal pose differs from the one the call started with.
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents) = 0;

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

    /// @brief Return the semantic virtual-event mask accepted by this camera kind.
    ///
    /// Input binders, scripted replay, and restore helpers use this mask to
    /// decide which `CVirtualGimbalEvent` categories may be passed to
    /// `manipulate(...)`.
	virtual uint32_t getAllowedVirtualEvents() const = 0u;

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

    /// @brief Update only the translation motion scale used by the camera runtime.
    inline void setMoveSpeedScale(double scalar)
    {
        m_motionConfig.moveSpeedScale = scalar;
    }

    /// @brief Update only the rotation motion scale used by the camera runtime.
    inline void setRotationSpeedScale(double scalar)
    {
        m_motionConfig.rotationSpeedScale = scalar;
    }

    /// @brief Update both translation and rotation motion scales at once.
    inline void setMotionScales(const double moveScale, const double rotationScale)
    {
        setMoveSpeedScale(moveScale);
        setRotationSpeedScale(rotationScale);
    }

    /// @brief Return the current translation motion scale.
    inline double getMoveSpeedScale() const { return m_motionConfig.moveSpeedScale; }
    /// @brief Return the current rotation motion scale.
    inline double getRotationSpeedScale() const { return m_motionConfig.rotationSpeedScale; }
    /// @brief Return the full motion-scale bundle.
    inline const SMotionConfig& getMotionConfig() const { return m_motionConfig; }
    /// @brief Return the effective world-space translation represented by a unit virtual move event.
    inline double getScaledVirtualTranslationMagnitude() const
    {
        return getUnscaledVirtualTranslationMagnitude() * getMoveSpeedScale();
    }
    /// @brief Return the raw translation magnitude before applying the camera-local move scale.
    ///
    /// TODO: target-relative rigs drive their distance through this, so zooming ignores `moveSpeedScale`
    /// while panning honours it. Either route both through `scaleVirtualTranslation` or give the distance
    /// its own documented scale.
    inline double getUnscaledVirtualTranslationMagnitude() const
    {
        return VirtualTranslationUnit;
    }
    /// @brief Scale one scalar translation magnitude through the active move scale.
    inline double scaleVirtualTranslation(const double magnitude) const
    {
        return magnitude * getScaledVirtualTranslationMagnitude();
    }
    /// @brief Scale one translation vector through the active move scale.
    template<typename T, uint32_t N>
    inline hlsl::vector<T, N> scaleVirtualTranslation(const hlsl::vector<T, N>& magnitude) const
    {
        return magnitude * static_cast<T>(getScaledVirtualTranslationMagnitude());
    }
    /// @brief Scale one scalar translation magnitude without applying the camera-local move scale.
    inline double scaleUnscaledVirtualTranslation(const double magnitude) const
    {
        return magnitude * getUnscaledVirtualTranslationMagnitude();
    }
    /// @brief Scale one translation vector without applying the camera-local move scale.
    template<typename T, uint32_t N>
    inline hlsl::vector<T, N> scaleUnscaledVirtualTranslation(const hlsl::vector<T, N>& magnitude) const
    {
        return magnitude * static_cast<T>(getUnscaledVirtualTranslationMagnitude());
    }
    /// @brief Scale one scalar rotation magnitude through the active rotation scale.
    inline double scaleVirtualRotation(const double magnitude) const
    {
        return magnitude * getRotationSpeedScale();
    }
    /// @brief Scale one rotation vector through the active rotation scale.
    template<typename T, uint32_t N>
    inline hlsl::vector<T, N> scaleVirtualRotation(const hlsl::vector<T, N>& magnitude) const
    {
        return magnitude * static_cast<T>(getRotationSpeedScale());
    }
protected:
    SMotionConfig m_motionConfig;
};

}

#endif // _I_CAMERA_HPP_

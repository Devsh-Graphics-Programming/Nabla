// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _I_CAMERA_HPP_
#define _I_CAMERA_HPP_

#include <optional>
#include <utility>

#include "nbl/core/IReferenceCounted.h"
#include "CCameraTraits.hpp"
#include "IGimbal.hpp"

namespace nbl::core
{

/// @brief Shared runtime camera interface.
///
/// `ICamera` consumes batches of `CVirtualGimbalEvent` values and updates one
/// camera pose stored in `CGimbal`. A `CVirtualGimbalEvent` identifies one
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
    static inline constexpr double DefaultMoveSpeedScaleValue = 0.01;
    static inline constexpr double DefaultRotationSpeedScaleValue = 0.003;
    static inline constexpr double VirtualTranslationUnit = 0.01;

public:
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
        float maxDistance = SCameraTargetRelativeTraits::DefaultMaxDistance;
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
        double minU = static_cast<double>(SCameraTargetRelativeTraits::MinDistance);
        /// @brief Minimal valid radial distance derived from the `(u, v)` pair.
        hlsl::float64_t minDistance = static_cast<hlsl::float64_t>(SCameraTargetRelativeTraits::MinDistance);
        /// @brief Maximal valid radial distance derived from the `(u, v)` pair, or infinity when unbounded.
        hlsl::float64_t maxDistance = static_cast<hlsl::float64_t>(SCameraTargetRelativeTraits::DefaultMaxDistance);
    };

    /// @brief Parametric path-rig state used by the `Path Rig` camera kind.
    ///
    /// The built-in path model interprets `(s, u, v, roll)` as path progress,
    /// lateral shape coordinates, and roll around the local forward axis.
    /// Other path models may map the same coordinates onto different geometry.
    struct PathState
    {
        /// @brief Primary path-progress coordinate interpreted by the active path model.
        double s = 0.0;
        /// @brief First lateral/shape coordinate interpreted by the active path model.
        double u = 0.0;
        /// @brief Second lateral/shape coordinate interpreted by the active path model.
        double v = 0.0;
        /// @brief Roll around the path-model forward axis, expressed in radians.
        double roll = 0.0;

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
        static inline PathState fromTranslationVector(const hlsl::float64_t3& value, const double pathRoll = 0.0)
        {
            return {
                .s = value.z,
                .u = value.x,
                .v = value.y,
                .roll = pathRoll
            };
        }
    };

    /// @brief Gimbal that stores the runtime camera pose and cached world-to-view transform.
    ///
    /// Camera implementations own one `CGimbal` instance and update it after
    /// applying their internal state model. The gimbal stores world-space
    /// position, orientation, and the cached view matrix derived from them.
    class CGimbal : public IGimbal<hlsl::float64_t>
    {
    public:
        using base_t = IGimbal<hlsl::float64_t>;
        using model_matrix_t = typename base_t::model_matrix_t;

        CGimbal(typename base_t::SCreationParameters parameters) : base_t(std::move(parameters)) { updateView(); }
        ~CGimbal() = default;

        inline void begin() { base_t::begin(); }
        inline void setPosition(const hlsl::float64_t3& position) { base_t::setPosition(position); }
        inline void setScale(const hlsl::float64_t3& scale) { base_t::setScale(scale); }
        inline void setOrientation(const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation) { base_t::setOrientation(orientation); }
        inline void transform(const CReferenceTransform& reference, const typename base_t::VirtualImpulse& impulse) { base_t::transform(reference, impulse); }
        inline void rotate(const hlsl::float64_t3& axis, float dRadians) { base_t::rotate(axis, dRadians); }
        inline void move(hlsl::float64_t3 delta) { base_t::move(delta); }
        inline void strafe(hlsl::float64_t distance) { base_t::strafe(distance); }
        inline void climb(hlsl::float64_t distance) { base_t::climb(distance); }
        inline void advance(hlsl::float64_t distance) { base_t::advance(distance); }
        inline void end() { base_t::end(); }

        inline const hlsl::float64_t3& getPosition() const { return base_t::getPosition(); }
        inline const hlsl::camera_quaternion_t<hlsl::float64_t>& getOrientation() const { return base_t::getOrientation(); }
        inline const hlsl::float64_t3& getScale() const { return base_t::getScale(); }
        inline const hlsl::matrix<hlsl::float64_t, 3, 3>& getOrthonornalMatrix() const { return base_t::getOrthonornalMatrix(); }
        inline const hlsl::float64_t3& getXAxis() const { return base_t::getXAxis(); }
        inline const hlsl::float64_t3& getYAxis() const { return base_t::getYAxis(); }
        inline const hlsl::float64_t3& getZAxis() const { return base_t::getZAxis(); }
        inline hlsl::float64_t3 getLocalTarget() const { return base_t::getLocalTarget(); }
        inline hlsl::float64_t3 getWorldTarget() const { return base_t::getWorldTarget(); }
        inline const size_t& getManipulationCounter() const { return base_t::getManipulationCounter(); }
        inline bool isManipulating() const { return base_t::isManipulating(); }
        inline bool extractReferenceTransform(CReferenceTransform* out, const hlsl::float64_t4x4* referenceFrame = nullptr) const
        {
            return base_t::extractReferenceTransform(out, referenceFrame);
        }

        template <uint32_t AllowedEvents>
        inline typename base_t::VirtualImpulse accumulate(std::span<const CVirtualGimbalEvent> virtualEvents)
        {
            return base_t::template accumulate<AllowedEvents>(virtualEvents);
        }

        /// @brief Rebuild the cached world-to-view matrix from the current gimbal pose.
        inline void updateView()
        {            
            const auto& gRight = this->getXAxis();
            const auto& gUp = this->getYAxis();
            const auto& gForward = this->getZAxis();

            assert(hlsl::isOrthoBase(gRight, gUp, gForward));

            const auto& position = this->getPosition();

            m_viewMatrix[0u] = hlsl::float64_t4(gRight, -hlsl::dot(gRight, position));
            m_viewMatrix[1u] = hlsl::float64_t4(gUp, -hlsl::dot(gUp, position));
            m_viewMatrix[2u] = hlsl::float64_t4(gForward, -hlsl::dot(gForward, position));
        }

        /// @brief Return the cached world-to-view matrix derived from the current pose.
        inline const hlsl::float64_t3x4& getViewMatrix() const { return m_viewMatrix; }

    private:
        hlsl::float64_t3x4 m_viewMatrix;
    };

    class SScopedMotionScaleOverride
    {
    public:
        /// @brief Temporarily override both motion scales and restore the previous values on destruction.
        SScopedMotionScaleOverride(ICamera* camera, const double moveScale, const double rotationScale)
            : m_camera(camera)
        {
            if (!m_camera)
                return;

            m_prevMoveScale = m_camera->getMoveSpeedScale();
            m_prevRotationScale = m_camera->getRotationSpeedScale();
            m_camera->setMotionScales(moveScale, rotationScale);
        }

        SScopedMotionScaleOverride(const SScopedMotionScaleOverride&) = delete;
        SScopedMotionScaleOverride& operator=(const SScopedMotionScaleOverride&) = delete;

        SScopedMotionScaleOverride(SScopedMotionScaleOverride&& other) noexcept
            : m_camera(std::exchange(other.m_camera, nullptr)),
            m_prevMoveScale(other.m_prevMoveScale),
            m_prevRotationScale(other.m_prevRotationScale)
        {
        }

        SScopedMotionScaleOverride& operator=(SScopedMotionScaleOverride&& other) = delete;

        ~SScopedMotionScaleOverride()
        {
            if (m_camera)
                m_camera->setMotionScales(m_prevMoveScale, m_prevRotationScale);
        }

    private:
        ICamera* m_camera = nullptr;
        double m_prevMoveScale = 0.0;
        double m_prevRotationScale = 0.0;
    };

    ICamera() {}
	virtual ~ICamera() = default;

    /// @brief Return the mutable gimbal backing the runtime camera pose.
	virtual const CGimbal& getGimbal() = 0u;

    /// @brief Apply one frame of semantic virtual events and an optional rigid reference-frame anchor.
    ///
    /// `virtualEvents` stores one frame of semantic movement, rotation, and
    /// scale commands. Translation commands use `Move*`, rotation commands use
    /// `Tilt*`, `Pan*`, and `Roll*`, and scale commands use `Scale*`. Cameras
    /// interpret only the subset advertised by `getAllowedVirtualEvents()`.
    ///
    /// `referenceFrame` is an optional rigid world-space transform used as the
    /// anchor for this manipulation step. Free-like cameras may apply it
    /// directly as pose input. Constrained cameras may first resolve it into
    /// their own typed legal state and then apply event deltas in that state
    /// space.
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) = 0;
    /// @brief Apply one frame of virtual events while temporarily overriding the camera-local motion scales.
    inline bool manipulateWithMotionScales(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame, const double moveScale, const double rotationScale)
    {
        auto scopedOverride = overrideMotionScales(moveScale, rotationScale);
        return manipulate(virtualEvents, referenceFrame);
    }
    /// @brief Apply one frame of virtual events with unit translation and rotation scales.
    inline bool manipulateWithUnitMotionScales(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr)
    {
        return manipulateWithMotionScales(virtualEvents, referenceFrame, 1.0, 1.0);
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
        uint32_t mask = GoalStateNone;
        if (hasCapability(SphericalTarget))
            mask |= GoalStateSphericalTarget;
        if (hasCapability(DynamicPerspectiveFov))
            mask |= GoalStateDynamicPerspective;
        return mask;
    }

    /// @brief Return the stable human-readable identifier for this concrete camera instance.
    virtual std::string_view getIdentifier() const = 0u;

    /// @brief Check whether the camera exposes the requested optional capability.
    inline bool hasCapability(CameraCapability capability) const
    {
        return (getCapabilities() & capability) == capability;
    }

    /// @brief Check whether the camera can exchange the requested typed goal-state fragment.
    inline bool supportsGoalState(GoalStateMask goalState) const
    {
        return (getGoalStateMask() & goalState) == goalState;
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
    inline hlsl::camera_vector_t<T, N> scaleVirtualTranslation(const hlsl::camera_vector_t<T, N>& magnitude) const
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
    inline hlsl::camera_vector_t<T, N> scaleUnscaledVirtualTranslation(const hlsl::camera_vector_t<T, N>& magnitude) const
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
    inline hlsl::camera_vector_t<T, N> scaleVirtualRotation(const hlsl::camera_vector_t<T, N>& magnitude) const
    {
        return magnitude * static_cast<T>(getRotationSpeedScale());
    }
    /// @brief Create a scoped helper that restores the previous motion scales on destruction.
    inline SScopedMotionScaleOverride overrideMotionScales(const double moveScale, const double rotationScale)
    {
        return SScopedMotionScaleOverride(this, moveScale, rotationScale);
    }

protected:
    SMotionConfig m_motionConfig;
};

}

#endif // _I_CAMERA_HPP_

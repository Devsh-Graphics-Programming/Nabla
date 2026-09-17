#ifndef _NBL_IGIMBAL_HPP_
#define _NBL_IGIMBAL_HPP_

#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

#include "nbl/type_traits.h"
#include "CCameraMathUtilities.hpp"
#include "CVirtualGimbalEvent.hpp"

namespace nbl::ext::cameras
{
    /// @brief Optional rigid reference frame used to reinterpret a frame of semantic camera input.
    ///
    /// Some camera consumers replay authored input relative to an external frame
    /// instead of the current camera pose. This bundle stores the rigid transform
    /// and its orientation in a form ready for `IGimbal::transform(...)`.
    struct CReferenceTransform
    {
        /// @brief Rigid frame in the engine convention: basis vectors in the columns, translation in the last column.
        hlsl::float64_t4x4 frame;
        hlsl::math::quaternion<hlsl::float64_t> orientation = hlsl::math::quaternion<hlsl::float64_t>::identity();

        /// @brief World-space position, which lives in the last column of `frame`.
        inline hlsl::float64_t3 getPosition() const
        {
            return hlsl::float64_t3(frame[0].w, frame[1].w, frame[2].w);
        }

        /// @brief The reference basis, taken from `orientation` so it cannot be read with the wrong layout.
        inline SCameraBasis<hlsl::float64_t> getBasis() const
        {
            return CCameraMathUtilities::getOrientationBasis(orientation);
        }
    };

    /// @brief Generic world-space gimbal used by runtime cameras and tracked targets.
    ///
    /// The gimbal stores position, orientation, scale, and an orthonormal local
    /// basis. It also exposes `accumulate(...)`, which converts one batch of
    /// semantic `CVirtualGimbalEvent` values into translation, rotation, and
    /// scale impulses for a single manipulation step.
    template<typename T>
    requires is_any_of_v<T, hlsl::float32_t, hlsl::float64_t>
    class IGimbal
    {
    public:
        using precision_t = T;
        using quaternion_t = hlsl::math::quaternion<precision_t>;
        template<uint32_t N>
        using vector_t = hlsl::vector<precision_t, N>;
        /// @brief underlying type for world matrix (TRS)
        using model_matrix_t = hlsl::matrix<precision_t, 3, 4>;

        /// @brief One frame of accumulated virtual translation, rotation, and scaling intent.
        struct VirtualImpulse
        {
            vector_t<3u> dVirtualTranslate { 0.0f }, dVirtualRotation { 0.0f }, dVirtualScale { 1.0f };
        };

        /// @brief Accumulates one frame of virtual events into a translation/rotation/scale impulse.
        template <uint32_t AllowedEvents>
        VirtualImpulse accumulate(std::span<const CVirtualGimbalEvent> virtualEvents, const vector_t<3u>& gRightOverride, const vector_t<3u>& gUpOverride, const vector_t<3u>& gForwardOverride)
        {
            VirtualImpulse impulse;

            for (const auto& event : virtualEvents)
            {
                assert(event.magnitude >= 0);

                // translation events
                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveRight)
                    if (event.type == CVirtualGimbalEvent::MoveRight)
                        impulse.dVirtualTranslate.x += static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveLeft)
                    if (event.type == CVirtualGimbalEvent::MoveLeft)
                        impulse.dVirtualTranslate.x -= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveUp)
                    if (event.type == CVirtualGimbalEvent::MoveUp)
                        impulse.dVirtualTranslate.y += static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveDown)
                    if (event.type == CVirtualGimbalEvent::MoveDown)
                        impulse.dVirtualTranslate.y -= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveForward)
                    if (event.type == CVirtualGimbalEvent::MoveForward)
                        impulse.dVirtualTranslate.z += static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveBackward)
                    if (event.type == CVirtualGimbalEvent::MoveBackward)
                        impulse.dVirtualTranslate.z -= static_cast<precision_t>(event.magnitude);

                // rotation events
                if constexpr (AllowedEvents & CVirtualGimbalEvent::TiltUp)
                    if (event.type == CVirtualGimbalEvent::TiltUp)
                        impulse.dVirtualRotation.x += static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::TiltDown)
                    if (event.type == CVirtualGimbalEvent::TiltDown)
                        impulse.dVirtualRotation.x -= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::PanRight)
                    if (event.type == CVirtualGimbalEvent::PanRight)
                        impulse.dVirtualRotation.y += static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::PanLeft)
                    if (event.type == CVirtualGimbalEvent::PanLeft)
                        impulse.dVirtualRotation.y -= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::RollRight)
                    if (event.type == CVirtualGimbalEvent::RollRight)
                        impulse.dVirtualRotation.z += static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::RollLeft)
                    if (event.type == CVirtualGimbalEvent::RollLeft)
                        impulse.dVirtualRotation.z -= static_cast<precision_t>(event.magnitude);

                // scaling events
                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleXInc)
                    if (event.type == CVirtualGimbalEvent::ScaleXInc)
                        impulse.dVirtualScale.x *= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleXDec)
                    if (event.type == CVirtualGimbalEvent::ScaleXDec)
                        impulse.dVirtualScale.x *= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleYInc)
                    if (event.type == CVirtualGimbalEvent::ScaleYInc)
                        impulse.dVirtualScale.y *= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleYDec)
                    if (event.type == CVirtualGimbalEvent::ScaleYDec)
                        impulse.dVirtualScale.y *= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleZInc)
                    if (event.type == CVirtualGimbalEvent::ScaleZInc)
                        impulse.dVirtualScale.z *= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleZDec)
                    if (event.type == CVirtualGimbalEvent::ScaleZDec)
                        impulse.dVirtualScale.z *= static_cast<precision_t>(event.magnitude);
            }

            return impulse;
        }

        /// @brief Accumulate one frame of virtual events using the current gimbal basis as the reference frame.
        template <uint32_t AllowedEvents>
        VirtualImpulse accumulate(std::span<const CVirtualGimbalEvent> virtualEvents)
        {
            return accumulate<AllowedEvents>(virtualEvents, getXAxis(), getYAxis(), getZAxis());
        }

        /// @brief Construction-time pose for one gimbal instance.
        struct SCreationParameters
        {
            vector_t<3u> position;
            quaternion_t orientation = hlsl::math::quaternion<precision_t>::identity();
        };

        IGimbal(const IGimbal&) = default;
        IGimbal(IGimbal&&) noexcept = default;
        IGimbal& operator=(const IGimbal&) = default;
        IGimbal& operator=(IGimbal&&) noexcept = default;

        IGimbal(SCreationParameters&& parameters)
            : m_position(parameters.position), m_orientation(parameters.orientation)
        {
            updateOrthonormalOrientationBase();
        }

        /// @brief Enter manipulation mode and reset the per-frame manipulation counter.
        void begin()
        {
            m_isManipulating = true;
            m_counter = 0u;
        }

        /// @brief Replace the world-space position while the gimbal is in manipulation mode.
        inline void setPosition(const vector_t<3u>& position)
        {
            assert(m_isManipulating);

            if (m_position != position)
                m_counter++;

            m_position = position;
        }

        /// @brief Replace the scale component stored by the gimbal.
        inline void setScale(const vector_t<3u>& scale)
        {
            m_scale = scale;
        }

        /// @brief Replace the orientation while keeping the orthonormal basis normalized.
        inline void setOrientation(const quaternion_t& orientation)
        {
            assert(m_isManipulating);

            if (m_orientation.data != orientation.data)
                m_counter++;

            m_orientation = hlsl::normalize(orientation);
            updateOrthonormalOrientationBase();
        }

        /// @brief Apply a prebuilt rigid reference transform and an accumulated impulse in one step.
        inline void transform(const CReferenceTransform& reference, const VirtualImpulse& impulse)
        {
            setOrientation(reference.orientation * CCameraMathUtilities::makeQuaternionFromEulerRadiansYXZ(impulse.dVirtualRotation));
            setPosition(
                reference.getPosition() +
                hlsl::normalize(reference.orientation).transformVector(hlsl::float64_t3(impulse.dVirtualTranslate), true)
            );
        }

        /// @brief Rotate the gimbal around a world-space axis by the requested angle in radians.
        inline void rotate(const vector_t<3u>& axis, float dRadians)
        {
            assert(m_isManipulating);

            if(dRadians)
                m_counter++;

            const auto dRotation = hlsl::math::quaternion<precision_t>::createFromAxisAngle(axis, static_cast<precision_t>(dRadians));
            m_orientation = hlsl::normalize(dRotation * m_orientation);
            updateOrthonormalOrientationBase();
        }

        /// @brief Translate the gimbal directly in world space.
        inline void move(vector_t<3u> delta)
        {
            assert(m_isManipulating);

            auto newPosition = m_position + delta;

            if (newPosition != m_position)
                m_counter++;

            m_position = newPosition;
        }

        /// @brief Translate the gimbal along its local right axis.
        inline void strafe(precision_t distance)
        {
            move(getXAxis() * distance);
        }

        /// @brief Translate the gimbal along its local up axis.
        inline void climb(precision_t distance)
        {
            move(getYAxis() * distance);
        }

        /// @brief Translate the gimbal along its local forward axis.
        inline void advance(precision_t distance)
        {
            move(getZAxis() * distance);
        }

        /// @brief Leave manipulation mode after all pose updates for the current frame are finished.
        inline void end()
        {
            m_isManipulating = false;
        }

        /// @brief Position of gimbal in world space
        inline const vector_t<3u>& getPosition() const { return m_position; }

        /// @brief Orientation of gimbal
        inline const quaternion_t& getOrientation() const { return m_orientation; }

        /// @brief Scale transform component
        inline const vector_t<3u>& getScale() const { return m_scale; }

        /// @brief World matrix (TRS)
        template<typename TRS = model_matrix_t>
        requires is_any_of_v<TRS, model_matrix_t, hlsl::matrix<T, 4u, 4u>>
        const TRS operator()() const
        { 
            const auto& position = getPosition();
            const auto& basis = getBasis();
            const auto& scale = getScale();

            if constexpr (std::is_same_v<TRS, model_matrix_t>)
            {
                return
                {
                    hlsl::vector<precision_t, 4>(basis.right * scale.x, position.x),
                    hlsl::vector<precision_t, 4>(basis.up * scale.y, position.y),
                    hlsl::vector<precision_t, 4>(basis.forward * scale.z, position.z)
                };
            }
            else
            {
                return
                {
                    hlsl::vector<precision_t, 4>(basis.right * scale.x, T(0)),
                    hlsl::vector<precision_t, 4>(basis.up * scale.y, T(0)),
                    hlsl::vector<precision_t, 4>(basis.forward * scale.z, T(0)),
                    hlsl::vector<precision_t, 4>(position, T(1))
                };
            }
        }

        /// @brief Orthonormal local basis, as three named vectors
        inline const SCameraBasis<precision_t>& getBasis() const { return m_basis; }

        /// @brief Base "right" vector in orthonormal orientation basis (X-axis)
        inline const vector_t<3u>& getXAxis() const { return m_basis.right; }

        /// @brief Base "up" vector in orthonormal orientation basis (Y-axis)
        inline const vector_t<3u>& getYAxis() const { return m_basis.up; }

        /// @brief Base "forward" vector in orthonormal orientation basis (Z-axis)
        inline const vector_t<3u>& getZAxis() const { return m_basis.forward; }

        /// @brief Target vector in local space, alias for getZAxis()
        inline vector_t<3u> getLocalTarget() const { return getZAxis(); }

        /// @brief Target vector in world space
        inline vector_t<3u> getWorldTarget() const { return getPosition() + getLocalTarget(); }

        /// @brief Counts how many times a valid manipulation has been performed, the counter resets when begin() is called
        inline const size_t& getManipulationCounter() const { return m_counter; }

        /// @brief Returns true if gimbal records a manipulation 
        inline bool isManipulating() const { return m_isManipulating; }

        /// @brief Build a rigid reference transform either from an external frame or from the current gimbal pose.
        bool extractReferenceTransform(CReferenceTransform* out, const hlsl::float64_t4x4* referenceFrame = nullptr) const
        {
            if (not out)
                return false;

            if (referenceFrame)
            {
                if (!CCameraMathUtilities::tryBuildRigidFrameFromTransform(*referenceFrame, out->frame, out->orientation))
                    return false;
            }
            else
            {
                out->orientation = getOrientation();
                out->frame = CCameraMathUtilities::composeTransformMatrix(getPosition(), out->orientation);
            }

            return true;
        }

    private:
        inline void updateOrthonormalOrientationBase()
        {
            m_basis = CCameraMathUtilities::getOrientationBasis(m_orientation);
        }

        /// @brief Position of a gimbal in world space
        vector_t<3u> m_position;

        /// @brief Normalized orientation of gimbal
        quaternion_t m_orientation;

        /// @brief Scale transform component
        vector_t<3u> m_scale = { 1.f, 1.f , 1.f };

        /// @brief Orthonormal basis reconstructed from the current orientation.
        SCameraBasis<precision_t> m_basis;

        /// @brief Counter that increments for each performed manipulation, resets with each begin() call
        size_t m_counter = {};

        /// @brief Tracks whether gimbal is currently in manipulation mode
        bool m_isManipulating = false;

    };
} // namespace nbl::ext::cameras

#endif // _NBL_IGIMBAL_HPP_

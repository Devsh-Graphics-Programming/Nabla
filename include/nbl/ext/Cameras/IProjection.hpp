#ifndef _NBL_I_PROJECTION_HPP_
#define _NBL_I_PROJECTION_HPP_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl::core
{

/// @brief Base interface for any reusable projection model in the camera stack.
///
/// A projection transforms vectors between some input space and the projection
/// space understood by a concrete viewport or projection consumer. Specialized
/// interfaces such as `ILinearProjection`, `IPlanarProjection`, and
/// `IPerspectiveProjection` refine this abstraction with additional structure.
class IProjection
{
public:
    /// @brief Common vector type used by projection and unprojection operations.
    using projection_vector_t = hlsl::float64_t4;

    /// @brief Stable runtime classification of supported projection families.
    enum class ProjectionType
    {
        /// @brief Any raw linear transformation, for example it may represent Perspective, Orthographic, Oblique, Axonometric, Shear projections
        Linear,

        /// @brief Specialized linear projection for planar projections with parameters
        Planar,

        /// @brief Extension of planar projection represented by pre-transform & planar transform combined projecting onto R3 cave quad
        CaveQuad,

        /// @brief Specialized CaveQuad projection, represents planar projections onto cube with 6 quad cube faces
        Cube,

        Spherical,
        ThinLens,
        
        Count
    };

    IProjection() = default;
    virtual ~IProjection() = default;

    /// @brief Transform a vector from its input space into projection space.
    ///
    /// @param vecToProjectionSpace Vector to transform into projection space.
    /// @param output Result vector in projection space.
    virtual void project(const projection_vector_t& vecToProjectionSpace, projection_vector_t& output) const = 0;

    /// @brief Transform a vector from projection space back to the original space.
    ///
    /// The inverse transform may fail because the original projection may be singular.
    ///
    /// @param vecFromProjectionSpace Vector in projection space.
    /// @param output Result vector in the original space.
    /// @return `true` when the inverse transform succeeded, otherwise `false`.
    virtual bool unproject(const projection_vector_t& vecFromProjectionSpace, projection_vector_t& output) const = 0;

    /// @brief Return the specific projection family implemented by the concrete instance.
    ///
    /// Examples include linear, spherical, and thin-lens projections as defined
    /// by `ProjectionType`.
    ///
    /// @return The type of this projection.
    virtual ProjectionType getProjectionType() const = 0;
};

} // namespace nbl::core

#endif // _NBL_IPROJECTION_HPP_

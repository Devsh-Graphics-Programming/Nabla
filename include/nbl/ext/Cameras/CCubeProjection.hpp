#ifndef _NBL_CCUBE_PROJECTION_HPP_
#define _NBL_CCUBE_PROJECTION_HPP_

#include "IRange.hpp"
#include "IPerspectiveProjection.hpp"

namespace nbl::core
{

/// @brief Projection where each cube face is a perspective quad.
///
/// This represents a cube projection for a direction vector where each face of
/// the cube is treated as a quad. Projection onto the cube is done through
/// those quads, each with its own pre-transform and viewport linear matrix.
class CCubeProjection final : public IPerspectiveProjection, public IProjection
{
public:
    /// @brief Represents six face identifiers of a cube.
    enum CubeFaces : uint8_t
    {
        /// @brief Cube face in the +X base direction
        PositiveX = 0,

        /// @brief Cube face in the -X base direction
        NegativeX,

        /// @brief Cube face in the +Y base direction
        PositiveY,

        /// @brief Cube face in the -Y base direction
        NegativeY,

        /// @brief Cube face in the +Z base direction
        PositiveZ,

        /// @brief Cube face in the -Z base direction
        NegativeZ,

        CubeFacesCount
    };

    inline static core::smart_refctd_ptr<CCubeProjection> create(core::smart_refctd_ptr<ICamera>&& camera)
    {
        if (!camera)
            return nullptr;

        return core::smart_refctd_ptr<CCubeProjection>(new CCubeProjection(core::smart_refctd_ptr(camera)), core::dont_grab);
    }

    virtual uint32_t getLinearProjectionCount() const override
    {
        return static_cast<uint32_t>(m_quads.size());
    }

    virtual const ILinearProjection::CProjection& getLinearProjection(uint32_t index) const override
    {
        assert(index < m_quads.size());
        return m_quads[index];
    }

    void transformCube()
    {
        // Cube-face quad generation is not implemented yet.
    }

    virtual ProjectionType getProjectionType() const override { return ProjectionType::Cube; }

    virtual void project(const projection_vector_t& vecToProjectionSpace, projection_vector_t& output) const override
    {
        auto direction = hlsl::normalize(vecToProjectionSpace);

        // Cube-face projection is not implemented yet.
    }

    virtual bool unproject(const projection_vector_t& vecFromProjectionSpace, projection_vector_t& output) const override
    {
        // Reverse projection is not implemented yet.
    }

    template<CubeFaces FaceIx>
    requires (FaceIx != CubeFacesCount)
    inline const CProjection& getProjectionQuad()
    {
        return m_quads[FaceIx];
    }

private:
    CCubeProjection(core::smart_refctd_ptr<ICamera>&& camera)
        : IPerspectiveProjection(core::smart_refctd_ptr(camera)) {}
    virtual ~CCubeProjection() = default;

    std::array<CProjection, CubeFacesCount> m_quads;
};

} // namespace nbl::core

#endif // _NBL_CCUBE_PROJECTION_HPP_

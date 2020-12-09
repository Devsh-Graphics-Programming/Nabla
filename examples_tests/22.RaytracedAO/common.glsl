#ifndef _COMMON_INCLUDED_
#define _COMMON_INCLUDED_

#define MAX_ACCUMULATED_SAMPLES (1024*1024)

#define WORKGROUP_SIZE 256

#ifdef __cplusplus
	#define uint uint32_t
	struct vec3
	{
		float x,y,z;
	};
	#define mat4 irr::core::matrix4SIMD
	#define mat4x3 irr::core::matrix3x4SIMD
#endif


struct SLight
{
	#ifdef __cplusplus
	SLight() : obb() {}
	SLight(const SLight& other) : obb(other.obb) {}
	SLight(const irr::core::aabbox3df& bbox, const irr::core::matrix3x4SIMD& tform) : SLight()
	{
		auto extent = bbox.getExtent();
		obb.setScale(irr::core::vectorSIMDf(extent.X,extent.Y,extent.Z));
		obb.setTranslation(irr::core::vectorSIMDf(bbox.MinEdge.X,bbox.MinEdge.Y,bbox.MinEdge.Z));

		obb = irr::core::concatenateBFollowedByA(tform,obb);
	}
	
	inline SLight& operator=(SLight&& other) noexcept
	{
		std::swap(obb, other.obb);

		return *this;
	}

	// also known as an upper bound on lumens put into the scene
	inline float computeFluxBound(const irr::core::vectorSIMDf& radiance) const
	{
		const irr::core::vectorSIMDf rec709LumaCoeffs(0.2126f, 0.7152f, 0.0722f, 0.f);
		const auto unitHemisphereArea = 2.f*irr::core::PI<float>();
				
		const auto unitBoxScale = obb.getScale();
		const float obbArea = 2.f*(unitBoxScale.x*unitBoxScale.y+unitBoxScale.x*unitBoxScale.z+unitBoxScale.y*unitBoxScale.z);
				
		return irr::core::dot(radiance,rec709LumaCoeffs).x*unitHemisphereArea*obbArea;
	}
	#endif

	mat4x3 obb; // needs row_major qualifier
};


#endif

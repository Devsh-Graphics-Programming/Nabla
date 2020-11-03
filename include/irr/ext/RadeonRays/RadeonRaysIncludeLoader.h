#ifndef __IRR_EXT_RADEON_RAYS_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_EXT_RADEON_RAYS_INCLUDE_LOADER_H_INCLUDED__

#include "irr/asset/IBuiltinIncludeLoader.h"

class RadeonRaysIncludeLoader : public irr::asset::IBuiltinIncludeLoader
{
	public:
		const char* getVirtualDirectoryName() const override { return "glsl/ext/RadeonRays/"; }

	private:
		static std::string getRay(const std::string&)
		{
			return
R"(#ifndef __IRR_EXT_RADEON_RAYS_RAY_INCLUDED__
#define __IRR_EXT_RADEON_RAYS_RAY_INCLUDED__

struct RadeonRays_ray
{
	vec3 origin;
	float maxT; // FLT_MAX
	vec3 direction;
	float time;
	int mask; // want to have it to -1
	int _active; // want to have it to 1
	int backfaceCulling; // want to have it to 0
	int useless_padding; // can be used to forward data
}; 

RadeonRays_ray ext_RadeonRays_constructDefaultRay(in vec3 origin, in vec3 direction, in float maxLen, in int userData)
{
	RadeonRays_ray retval;
	retval.origin = origin;
	retval.maxT = maxLen;
	retval.direction = direction;
	retval.time = 0.0;
	retval.mask = -1;
	retval._active = 1;
	retval.backfaceCulling = 0;
	retval.useless_padding = userData;
	return retval;
}

#endif
)";
		}
		static std::string getIntersection(const std::string&)
		{
			return
R"(#ifndef __IRR_EXT_RADEON_RAYS_INTERSECTION_INCLUDED__
#define __IRR_EXT_RADEON_RAYS_INTERSECTION_INCLUDED__

struct RadeonRays_Intersection
{
	// Shape ID
	int shapeid;
	// Primitve ID
	int primid;

	int padding0;
	int padding1;
        
	// UV parametrization
	vec4 uvwt;
};

#endif
)";
		}

	protected:
		irr::core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
		{
			return {
				{ std::regex{"ray\\.glsl"}, &getRay },
				{ std::regex{"intersection\\.glsl"}, &getIntersection }
			};
		}
};

#endif //__IRR_EXT_RADEON_RAYS_INCLUDE_LOADER_H_INCLUDED__
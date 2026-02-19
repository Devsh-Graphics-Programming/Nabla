#ifndef _NBL_BUILTIN_HLSL_SAMPLING_WARP_SPHERICAL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_WARP_SPHERICAL_INCLUDED_

#include <nbl/builtin/hlsl/numbers.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/sampling/warp.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{
namespace warp
{

template <typename T = float32_t>
struct Spherical 
{
	using density_type = T;
	using domain_type = vector<density_type, 2>;
	using codomain_type = vector<density_type, 3>;

	template <typename DomainT NBL_FUNC_REQUIRES(is_same_v<DomainT, domain_type>)
	static WarpResult<codomain_type> warp(const DomainT uv)
	{
		codomain_type dir;
		dir.x = cos(uv.x * density_type(2) * numbers::pi<density_type>);
		dir.z = sqrt(density_type(1) - (dir.x * dir.x));
    if (uv.x > density_type(0.5))
			dir.z = -dir.z;
		const density_type theta = uv.y * numbers::pi<density_type>;
		const density_type cosTheta = cos(theta);
		const density_type sinTheta = sqrt(density_type(1) - (cosTheta * cosTheta));
		dir.xz *= sinTheta;
		dir.y = cosTheta;

		WarpResult<codomain_type> warpResult;
		warpResult.dst = dir;
		warpResult.density = density_type(1) / (density_type(2) * sinTheta * numbers::pi<density_type> * numbers::pi<density_type>);

		return warpResult;
	}

	template <typename CodomainT NBL_FUNC_REQUIRES(is_same_v<CodomainT, codomain_type>) 
	static domain_type inverseWarp(const CodomainT v)
	{
		const density_type phi = atan2(v.z, v.x);
		const density_type theta = acos(v.y);
		density_type uv_x = phi * density_type(0.5) * numbers::inv_pi<density_type>;
		if (uv_x < density_type(0))
			uv_x += density_type(1);
		density_type uv_y = theta * numbers::inv_pi<density_type>;
    return domain_type(uv_x, uv_y);
	}


	template <typename DomainT NBL_FUNC_REQUIRES(is_same_v<DomainT, domain_type>)
	static density_type forwardDensity(const DomainT uv)
	{
		const density_type theta = uv.y * numbers::pi<density_type>;
		return density_type(1) / (sin(theta) * density_type(2) * numbers::pi<density_type> * numbers::pi<density_type>);

	}

	template <typename CodomainT NBL_FUNC_REQUIRES(is_same_v<CodomainT, codomain_type>)
	static density_type backwardDensity(const CodomainT dst)
	{
		const density_type cosTheta = dst.y;
		const density_type sinTheta = sqrt(density_type(1) - (cosTheta * cosTheta));
		return density_type(1) / (sinTheta * density_type(2) * numbers::pi<density_type> * numbers::pi<density_type>);
	}
};

}
}
}
}

#endif
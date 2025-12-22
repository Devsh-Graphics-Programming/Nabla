#ifndef _NBL_BUILTIN_HLSL_WARP_SPHERICAL_INCLUDED_
#define _NBL_BUILTIN_HLSL_WARP_SPHERICAL_INCLUDED_

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
  struct Spherical 
  {
	  using domain_type = float32_t2;
      using codomain_type = float32_t3;

      template <typename D NBL_FUNC_REQUIRES(is_same_v<D, domain_type>)
      static WarpResult<codomain_type> warp(const D uv)
      {
        const float32_t phi = 2 * uv.x * numbers::pi<float32_t>;
        const float32_t theta = uv.y * numbers::pi<float32_t>;
        float32_t3 dir;
        dir.x = cos(uv.x * 2.f * numbers::pi<float32_t>);
        dir.y = sqrt(1.f - dir.x * dir.x);
        if (uv.x > 0.5f) dir.y = -dir.y;
        const float32_t cosTheta = cos(theta);
        float32_t sinTheta = (1.0 - cosTheta * cosTheta);
        dir.xy *= sinTheta;
        dir.z = cosTheta;
        WarpResult warpResult;
        warpResult.dst = dir;
        warpResult.density = 1 / (sinTheta * numbers::pi<float32_t> * numbers::pi<float32_t>);
        return warpResult;
      }

      template <typename D NBL_FUNC_REQUIRES(is_same_v<D, domain_type>)
      static float32_t forwardDensity(const D uv)
      {
        const float32_t theta = uv.y * numbers::pi<float32_t>;
        return 1.0f / (sin(theta) * 2 * numbers::pi<float32_t> * numbers::pi<float32_t>);

      }

      template <typename C NBL_FUNC_REQUIRES(is_same_v<C, codomain_type>)
      static float32_t backwardDensity(const C dst)
      {
        return 1.0f / (sqrt(1.0f - dst.z * dst.z) * 2 * numbers::pi<float32_t> * numbers::pi<float32_t>);
      }
  };

}
}
}
}

#endif
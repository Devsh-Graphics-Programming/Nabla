#ifndef _NBL_BUILTIN_HLSL_WARP_SPHERICAL_INCLUDED_
#define _NBL_BUILTIN_HLSL_WARP_SPHERICAL_INCLUDED_

#include <nbl/builtin/hlsl/numbers.hlsl>

namespace nbl
{
namespace hlsl
{
namespace warp
{

  class Spherical 
  {
    public:
      using codomain_type = float32_t3;

      template <typename UV NBL_FUNC_REQUIRES(is_same_v<UV, float32_t2>)
      static codomain_type warp(const UV uv)
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
        return dir;
      }

      template <typename UV NBL_FUNC_REQUIRES(is_same_v<UV, float32_t2>)
      static float32_t forwardDensity(const UV uv)
      {
        const float32_t theta = uv.y * numbers::pi<float32_t>;
        return 1.0f / (sin(theta) * 2 * PI * PI);

      }

      template <typename C NBL_FUNC_REQUIRES(is_same_v<C, codomain_type>)
      static float32_t backwardDensity(const C out)
      {
        //TODO(kevinyu): Derive this density
      }
  };

}
}
}

#endif
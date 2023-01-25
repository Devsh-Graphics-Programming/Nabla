#include <nbl/builtin/hlsl/shapes/line.hlsl>
#include <nbl/builtin/hlsl/shapes/circle.hlsl>

namespace nbl
{
namespace hlsl
{
namespace shapes
{
	struct RoundedLine_t
	{
		float2 start, end;

		float getSignedDistance(float2 p, float thickness)
		{
			Circle_t startCircle = {start, thickness};
			Circle_t endCircle = {end, thickness};
			Line_t mainLine = {start, end};
			const float startCircleSD = startCircle.getSignedDistance(p);
			const float endCircleSD = endCircle.getSignedDistance(p);
			const float lineSD = mainLine.getSignedDistance(p, thickness);
			return min(lineSD, min(startCircleSD, endCircleSD));
		}
	};
}
}
}
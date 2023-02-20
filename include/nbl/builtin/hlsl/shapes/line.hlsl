namespace nbl
{
namespace hlsl
{
namespace shapes
{
	struct Line_t
	{
		float2 start, end;

		// https://www.shadertoy.com/view/stcfzn with modifications
		float getSignedDistance(float2 p, float thickness)
		{
			const float l = length(end - start);
			const float2  d = (end - start) / l;
			float2  q = p - (start + end) * 0.5;
			q = mul(float2x2(d.x, d.y, -d.y, d.x), q);
			q = abs(q) - float2(l * 0.5, thickness);
			return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
		}
	};
}
}
}
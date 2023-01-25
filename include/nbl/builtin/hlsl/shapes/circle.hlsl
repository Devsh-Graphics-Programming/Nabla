namespace nbl
{
namespace hlsl
{
namespace shapes
{
	struct Circle_t
	{
		float2 center;
		float radius;

		Circle_t(float2 center, float radius) :
			center(center),
			radius(radius)
		{}

		float getSignedDistance(float2 p)
		{
			return distance(p, center) - radius;
		}
	};
}
}
}
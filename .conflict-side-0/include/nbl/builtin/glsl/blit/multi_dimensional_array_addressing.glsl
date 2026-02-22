#ifndef _NBL_GLSL_BLIT_MULTI_DIMENSIONAL_ARRAY_ADDRESSING_INCLUDED_
#define _NBL_GLSL_BLIT_MULTI_DIMENSIONAL_ARRAY_ADDRESSING_INCLUDED_

uint nbl_glsl_multi_dimensional_array_addressing_snakeCurve(in uvec3 coordinate, in uvec3 extents)
{
	return (coordinate.z * extents.y + coordinate.y) * extents.x + coordinate.x;
}

uvec3 nbl_glsl_multi_dimensional_array_addressing_snakeCurveInverse(in uint linearIndex, in uvec2 gridDimPrefixProduct)
{
	uvec3 index3D;

	index3D.z = linearIndex / gridDimPrefixProduct.y;

	const uint tmp = linearIndex - (index3D.z * gridDimPrefixProduct.y);
	index3D.y = tmp / gridDimPrefixProduct.x;
	index3D.x = tmp - (index3D.y * gridDimPrefixProduct.x);

	return index3D;
}

uvec3 nbl_glsl_multi_dimensional_array_addressing_snakeCurveInverse(in uint linearIndex, in uvec3 gridDim)
{
	return nbl_glsl_multi_dimensional_array_addressing_snakeCurveInverse(linearIndex, uvec2(gridDim.x, gridDim.x*gridDim.y));
}

#endif
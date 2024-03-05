#ifndef _NBL_BUILTIN_HLSL_CENTRAL_LIMIT_BLUR_BOX_BLUR_INCLUDED_
#define _NBL_BUILTIN_HLSL_CENTRAL_LIMIT_BLUR_BOX_BLUR_INCLUDED_

#include <nbl/builtin/hlsl/central_limit_blur/common.hlsl>

#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/workgroup/arithmetic.hlsl>
#include <nbl/builtin/hlsl/workgroup/scratch_size.hlsl>


// https://github.com/microsoft/DirectXShaderCompiler/issues/6144
static nbl::hlsl::uint32_t3 /*nbl::hlsl::glsl::*/gl_WorkGroupSize() { return uint32_t3( WORKGROUP_SIZE, 1, 1 ); }

namespace nbl
{
namespace hlsl
{
namespace central_limit_blur
{

static const uint32_t arithmeticSz = nbl::hlsl::workgroup::scratch_size_arithmetic<WORKGROUP_SIZE>::value;
static const uint32_t broadcastSz = nbl::hlsl::workgroup::scratch_size_broadcast;
static const uint32_t scratchSz = arithmeticSz + broadcastSz;

groupshared uint32_t scratch[ scratchSz ];

template<typename T, uint16_t offset>
struct ScratchProxy
{
	T get( const uint32_t ix )
	{
		return nbl::hlsl::bit_cast< T, uint32_t >( scratch[ ix + offset ] );
	}
	void set( const uint32_t ix, NBL_CONST_REF_ARG( T ) value )
	{
		scratch[ ix + offset ] = nbl::hlsl::bit_cast< uint32_t, T >( value );
	}

	void workgroupExecutionAndMemoryBarrier()
	{
		nbl::hlsl::glsl::barrier();
	}
};

ScratchProxy<float32_t, 0> prefixSumsAccessor;
ScratchProxy<float32_t, arithmeticSz> broadcastAccessor;

struct prefix_sum_t
{
	using bin_op_t = nbl::hlsl::plus<float32_t>;
	using type_t = bin_op_t::type_t;

	type_t operator()( NBL_CONST_REF_ARG( type_t ) value )
	{
		type_t retval = nbl::hlsl::workgroup::inclusive_scan<bin_op_t, arithmeticSz>::template __call<ScratchProxy<type_t, 0> >( value, prefixSumsAccessor );
		// we barrier before because we alias the accessors for Binop
		prefixSumsAccessor.workgroupExecutionAndMemoryBarrier();
		return retval;
	}
};

prefix_sum_t workgroupPrefixSum;

float32_t workgroupBroadcast( NBL_CONST_REF_ARG( float32_t ) value, uint32_t index )
{
	using accessor_t = ScratchProxy<float32_t, arithmeticSz>;
	return nbl::hlsl::workgroup::Broadcast<float32_t, accessor_t>( value, broadcastAccessor, index );
}

// Todo: This spillage calculation is hacky! The lower bound of `_NBL_GLSL_EXT_BLUR_SPILLAGE_LOWER_BOUND_` is just an adhoc
// thing which happens to work for all images until now but it could very well break for the next image.
// This:
// `#define LOCAL_SPILLAGE ITEMS_PER_THREAD`
// seems to be a safe lower bound, however, it must be inefficient.
static const uint32_t SPILLAGE_LOWER_BOUND = ( WORKGROUP_SIZE / 4 );

static const uint32_t IMPL_LOCAL_SPILLAGE = ( ( arithmeticSz - 1 ) / WORKGROUP_SIZE + 1 );

static const uint32_t LOCAL_SPILLAGE =
( IMPL_LOCAL_SPILLAGE > WORKGROUP_SIZE ) ? WORKGROUP_SIZE :
	( ( IMPL_LOCAL_SPILLAGE < WORKGROUP_SIZE ) ? SPILLAGE_LOWER_BOUND : IMPL_LOCAL_SPILLAGE );


uint32_t DataIndex( uint32_t i ) // ?
{
	return ( i * gl_WorkGroupSize().x ) + nbl::hlsl::glsl::gl_LocalInvocationIndex();
}

//TODO: glsl's mod(x, y) = x - y * floor(x/y). Would prolly need to implement a glsl mod
void BoxBlur(
	const uint32_t ch,
	const float32_t inRadius,
	const uint32_t wrapMode,
	const float32_t4 borderColor,
	BufferAccessor textureAccessor
) {
	float32_t blurred = textureAccessor.get( DataIndex( i ), ch );

	for( uint32_t pass = 0u; pass < PASSES_PER_AXIS; ++pass )
	{
		float32_t previousBlockSum = 0.f;

		float32_t spill[ LOCAL_SPILLAGE ];
		for( uint32_t i = 0u; i < LOCAL_SPILLAGE; ++i )
		{
			float32_t scanResult = workgroupPrefixSum( blurred ) + previousBlockSum;
			spill[ i ] = scanResult;
			previousBlockSum = workgroupBroadcast( spill[ i ], gl_WorkGroupSize().x - 1u );
		}

		float32_t scanResult = workgroupPrefixSum( blurred ) + previousBlockSum;
		previousBlockSum = workgroupBroadcast( scanResult, gl_WorkGroupSize().x - 1u );

		uint32_t idx = DataIndex( i ); // ?
		prefixSumsAccessor.set( idx, scanResult );

		for( uint i = 0u; i < LOCAL_SPILLAGE; ++i )
		{
			uint32_t idx = DataIndex( i ); // ?
			prefixSumsAccessor.set( idx, spill[ i ] );
		}
		prefixSumsAccessor.workgroupExecutionAndMemoryBarrier();

		uint32_t idx = DataIndex( i );

		if( idx < AXIS_DIM )
		{
			const float32_t radius = inRadius * AXIS_DIM;
			float32_t left = float32_t( idx ) - radius - 1.f;
			float32_t right = float32_t( idx ) + radius;
			const uint32_t last = AXIS_DIM - 1;

			float32_t result = 0.f;

			if( right <= last )
			{
				float32_t floorRight = prefixSumsAccessor.get( floor( right ) );
				float32_t ceilRight = prefixSumsAccessor.get( ceil( right ) );
				result = lerp( floorRight, ceilRight, frac( right ) );
			}
			else
			{
				switch( wrapMode )
				{
				case WRAP_MODE_CLAMP_TO_EDGE:
				{
					float32_t sumLast = prefixSumsAccessor.get( last );
					float32_t sumLastMinusOne = prefixSumsAccessor.get( last - 1u );
					result = ( right - float32_t( last ) ) * ( sumLast - sumLastMinusOne ) + sumLast;
				} break;

				case WRAP_MODE_CLAMP_TO_BORDER:
				{
					float32_t sumLast = prefixSumsAccessor.get( last );
					result = sumLast + ( right - last ) * borderColor[ ch ];
				} break;

				case WRAP_MODE_REPEAT:
				{
					float32_t sumLast = prefixSumsAccessor.get( last );
					float32_t sumModFloorRight = prefixSumsAccessor.get( fmod( floor( right ) - AXIS_DIM, AXIS_DIM ) );
					float32_t sumModCeilRight = prefixSumsAccessor.get( fmod( ceil( right ) - AXIS_DIM, AXIS_DIM ) );
					const float32_t v_floored = ceil( ( floor( right ) - last ) / AXIS_DIM ) * sumLast + sumModFloorRight;
					const float32_t v_ceiled = ceil( ( ceil( right ) - last ) / AXIS_DIM ) * sumLast + sumModCeilRight;
					result = lerp( v_floored, v_ceiled, frac( right ) );
				} break;

				case WRAP_MODE_MIRROR:
				{
					float32_t v_floored;
					{
						const uint32_t floored = uint32_t( floor( right ) );
						const uint32_t d = floored - last; // distance from the right-most boundary, >=0

						if( fmod( d, 2 * AXIS_DIM ) == AXIS_DIM )
						{
							v_floored = ( ( d + AXIS_DIM ) / AXIS_DIM ) * prefixSumsAccessor.get( last );
						}
						else
						{
							const uint32_t period = uint32_t( ceil( float32_t( d ) / AXIS_DIM ) );

							if( ( period & 0x1u ) == 1 )
								v_floored = period * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( last - uint32_t( fmod( d, AXIS_DIM ) ) );
							else
								v_floored = period * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( fmod( d - 1, AXIS_DIM ) );
						}
					}

					float32_t v_ceiled;
					{
						const uint32_t ceiled = uint32_t( ceil( right ) );
						const uint32_t d = ceiled - last; // distance from the right-most boundary, >=0

						if( fmod( d, 2 * AXIS_DIM ) == AXIS_DIM )
						{
							v_ceiled = ( ( d + AXIS_DIM ) / AXIS_DIM ) * prefixSumsAccessor.get( last );
						}
						else
						{
							const uint32_t period = uint32_t( ceil( float32_t( d ) / AXIS_DIM ) );

							if( ( period & 0x1u ) == 1 )
								v_ceiled = period * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( last - uint32_t( fmod( d, AXIS_DIM ) ) );
							else
								v_ceiled = period * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( fmod( d - 1, AXIS_DIM ) );
						}
					}

					result = lerp( v_floored, v_ceiled, frac( right ) );
				} break;
				}
			}

			if( left >= 0 )
			{
				result -= lerp( prefixSumsAccessor.get( floor( left ) ), prefixSumsAccessor.get( ceil( left ) ), frac( left ) );
			}
			else
			{
				switch( wrapMode )
				{
				case WRAP_MODE_CLAMP_TO_EDGE:
				{
					result -= ( 1.f - abs( left ) ) * prefixSumsAccessor.get( 0 );
				} break;

				case WRAP_MODE_CLAMP_TO_BORDER:
				{
					result -= ( left + 1 ) * ( borderColor[ ch ] );
				} break;

				case WRAP_MODE_REPEAT:
				{
					const float32_t v_floored = floor( floor( left ) / AXIS_DIM ) * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( fmod( floor( left ), AXIS_DIM ) );
					const float32_t v_ceiled = floor( ceil( left ) / AXIS_DIM ) * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( fmod( ceil( left ), AXIS_DIM ) );
					result -= lerp( v_floored, v_ceiled, frac( left ) );
				} break;

				case WRAP_MODE_MIRROR:
				{
					float32_t v_floored;
					{
						const uint32_t floored = uint32_t( floor( left ) );
						if( fmod( abs( floored + 1 ), 2 * AXIS_DIM ) == 0 )
						{
							v_floored = -( abs( floored + 1 ) / AXIS_DIM ) * prefixSumsAccessor.get( last );
						}
						else
						{
							const uint32_t period = uint32_t( ceil( float32_t( abs( floored + 1 ) ) / AXIS_DIM ) );

							if( ( period & 0x1u ) == 1 )
								v_floored = -1 * ( period - 1 ) * prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( fmod( abs( floored + 1 ) - 1, AXIS_DIM ) );
							else
								v_floored = -1 * ( period - 1 ) * prefixSumsAccessor.get( last ) - ( prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( fmod( floored + 1, AXIS_DIM ) - 1 ) );
						}
					}

					float32_t v_ceiled;
					{
						const uint32_t ceiled = uint32_t( ceil( left ) );
						if( ceiled == 0 ) // Special case, wouldn't be possible for `floored` above
						{
							v_ceiled = 0;
						}
						else if( fmod( abs( ceiled + 1 ), 2 * AXIS_DIM ) == 0 )
						{
							v_ceiled = -( abs( ceiled + 1 ) / AXIS_DIM ) * prefixSumsAccessor.get( last );
						}
						else
						{
							const uint32_t period = uint32_t( ceil( float32_t( abs( ceiled + 1 ) ) / AXIS_DIM ) );

							if( ( period & 0x1u ) == 1 )
								v_ceiled = -1 * ( period - 1 ) * prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( fmod( abs( ceiled + 1 ) - 1, AXIS_DIM ) );
							else
								v_ceiled = -1 * ( period - 1 ) * prefixSumsAccessor.get( last ) - ( prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( fmod( ceiled + 1, AXIS_DIM ) - 1 ) );
						}
					}

					result -= lerp( v_floored, v_ceiled, frac( left ) );
				} break;
				}
			}

			blurred = result / ( 2.f * radius + 1.f );
		}
	}

	textureAccessor.set( DataIndex( i ), ch, blurred );
}

}
}
}
#endif
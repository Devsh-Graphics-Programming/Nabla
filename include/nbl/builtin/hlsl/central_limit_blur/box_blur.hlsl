#ifndef _NBL_BUILTIN_HLSL_CENTRAL_LIMIT_BLUR_BOX_BLUR_INCLUDED_
#define _NBL_BUILTIN_HLSL_CENTRAL_LIMIT_BLUR_BOX_BLUR_INCLUDED_

#include <nbl/builtin/hlsl/central_limit_blur/common.hlsl>

#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/workgroup/arithmetic.hlsl>
#include <nbl/builtin/hlsl/workgroup/scratch_size.hlsl>

namespace nbl
{
namespace hlsl
{
namespace central_limit_blur
{

static const uint32_t scratchSz = nbl::hlsl::workgroup::scratch_size_arithmetic<WORKGROUP_SIZE>::value;

groupshared uint32_t scratch[ scratchSz ];

// TODO: move to userspace
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

template<uint16_t ItemCount>
struct prefix_sum_t
{
	using bin_op_t = nbl::hlsl::plus<float32_t>;
	using type_t = bin_op_t::type_t;

	template<class Accessor>
	static type_t __call( NBL_CONST_REF_ARG(type_t) value, NBL_REF_ARG(Accessor) accessor )
	{
		type_t retval = nbl::hlsl::workgroup::inclusive_scan<bin_op_t, ItemCount>::template __call<Accessor>( value, accessor );
		accessor.workgroupExecutionAndMemoryBarrier();
		return retval;
	}
};


static const uint32_t SPILLAGE_LOWER_BOUND = ( WORKGROUP_SIZE / 4 );

static const uint32_t IMPL_LOCAL_SPILLAGE = ( ( scratchSz - 1 ) / WORKGROUP_SIZE + 1 );

static const uint32_t LOCAL_SPILLAGE =
( IMPL_LOCAL_SPILLAGE > WORKGROUP_SIZE ) ? WORKGROUP_SIZE : 
	( ( IMPL_LOCAL_SPILLAGE < WORKGROUP_SIZE ) ? SPILLAGE_LOWER_BOUND : IMPL_LOCAL_SPILLAGE );


//uint32_t globalInvocationIndex()
//{
//	return ( ITEMS_PER_THREAD * nbl::hlsl::glsl::gl_WorkGroupID().x ) + nbl::hlsl::workgroup::SubgroupContiguousIndex();
//}

//TODO: glsl's mod(x, y) = x - y * floor(x/y). Would prolly need to implement a glsl mod
template<class TextrueAccessor, uint16_t ITEMS_PER_THREAD>
void BoxBlur(
	const uint32_t channelCount,
	const float32_t inRadius,
	const uint32_t wrapMode,
	const float32_t4 borderColor,
	NBL_REF_ARG(TextrueAccessor) texAccessor
) {
	/*
	for( uint32_t pass = 0u; pass < PASSES_PER_AXIS; ++pass )
	{
		float32_t previousBlockSum = 0.f;

		uint32_t idx = workgroup::SubgroupContiguousIndex();
		//if( ix == gl_WorkGroupSize().x - 1u )
		//{
		//	broadcastAccessor.set( ix, )
		//}

		float32_t spill[ LOCAL_SPILLAGE ];
		for( uint32_t i = 0u; i < LOCAL_SPILLAGE; ++i )
		{
			float32_t scanResult = prefix_sum_t::__call( blurred[ i ] ) + previousBlockSum;
			spill[ i ] = scanResult;
			previousBlockSum = workgroupBroadcast( spill[ i ], gl_WorkGroupSize().x - 1u );
		}

		for( uint32_t i = LOCAL_SPILLAGE; i < ITEMS_PER_THREAD; ++i )
		{
			float32_t scanResult = prefix_sum_t::__call( blurred[ i ] ) + previousBlockSum;
			previousBlockSum = workgroupBroadcast( scanResult, gl_WorkGroupSize().x - 1u );
			prefixSumsAccessor.set( idx + i, scanResult );
		}

		for( uint i = 0u; i < LOCAL_SPILLAGE; ++i )
		{
			prefixSumsAccessor.set( idx + i, spill[ i ] );
		}
		prefixSumsAccessor.workgroupExecutionAndMemoryBarrier();

		for( uint32_t i = 0; i < ITEMS_PER_THREAD; ++i )
		{
			uint32_t _idx = idx + i;

			if( _idx < AXIS_DIM )
			{
				const float32_t radius = inRadius * AXIS_DIM;
				float32_t left = float32_t( _idx ) - radius - 1.f;
				float32_t right = float32_t( _idx ) + radius;
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

				blurred[ i ] = result / ( 2.f * radius + 1.f );
			}
		}
	}
	*/
}

}
}
}
#endif
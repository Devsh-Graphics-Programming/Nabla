#pragma once

#include <nbl/builtin/hlsl/blur/common.hlsl>

#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/workgroup/arithmetic.hlsl>
#include <nbl/builtin/hlsl/workgroup/scratch_size.hlsl>


// https://github.com/microsoft/DirectXShaderCompiler/issues/6144
uint32_t3 /*nbl::hlsl::glsl::*/gl_WorkGroupSize() { return uint32_t3( WORKGROUP_SIZE, 1, 1 ); }

uint32_t IndexInSharedMemory( uint32_t i )
{
	return ( i * gl_WorkGroupSize().x ) + nbl::hlsl::gl_LocalInvocationIndex();
}

static const uint32_t ITEMS_PER_THREAD = ( AXIS_DIM + gl_WorkGroupSize().x - 1 ) / gl_WorkGroupSize().x
static const uint32_t ITEMS_PER_WG = ITEMS_PER_THREAD * gl_WorkGroupSize().x;

static const uint32_t arithmeticSz = nbl::hlsl::workgroup::scratch_size_arithmetic<ITEMS_PER_WG>::value; 
static const uint32_t broadcastSz = 1u; // According to Broadcast impl note 
static const uint32_t scratchSz = arithmeticSz + broadcastSz;

groupshared uint32_t scratch[ scratchSz ];

template<typename T, uint16_t offset>
struct ScratchProxy
{
	T get( const uint32_t ix )
	{
		return bit_cast< T, uint32_t >( scratch[ ix + offset ] )
	}
	void set( const uint32_t ix, const T value )
	{
		scratch[ ix + offset ] = bit_cast< uint32_t, T >( value );
	}

	void workgroupExecutionAndMemoryBarrier()
	{
		nbl::hlsl::glsl::barrier();
	}
};

static ScratchProxy<float32_t, 0> prefixSumsAccessor; 
static ScratchProxy<float32_t, scanSz> broadcastAccessor;

struct prefix_sum_t
{
	using type_t = nbl::hlsl::plus<float32_t>::type_t;

	type_t operator()( type_t value )
	{
		type_t retval = nbl::hlsl::workgroup::inclusive_scan<nbl::hlsl::plus<float32_t>, ITEMS_PER_WG>::template __call<ScratchProxy<float32_t, 0> >( value, prefixSumsAccessor );
		// we barrier before because we alias the accessors for Binop
		arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
		return retval;
	}
};

static prefix_sum_t workgroupPrefixSum;


// Todo: This spillage calculation is hacky! The lower bound of `_NBL_GLSL_EXT_BLUR_SPILLAGE_LOWER_BOUND_` is just an adhoc
// thing which happens to work for all images until now but it could very well break for the next image.
// This:
// `#define LOCAL_SPILLAGE ITEMS_PER_THREAD`
// seems to be a safe lower bound, however, it must be inefficient.
static const uint32_t SPILLAGE_LOWER_BOUND = ( ITEMS_PER_THREAD / 4 )

static const uint32_t IMPL_LOCAL_SPILLAGE = ( ( arithmeticSz - 1 ) / gl_WorkGroupSize().x + 1 )

static const uint32_t chooseSpillageSize()
{
	if( IMPL_LOCAL_SPILLAGE > ITEMS_PER_THREAD )
	{
		return ITEMS_PER_THREAD;
	}
	else if( IMPL_LOCAL_SPILLAGE < ITEMS_PER_THREAD )
	{
		return SPILLAGE_LOWER_BOUND;
	}

	return IMPL_LOCAL_SPILLAGE;
}

static const uint32_t LOCAL_SPILLAGE = chooseSpillageSize();


nbl::hlsl::uint32_t3 getCoordinates( uint32_t idx, uint32_t direction )
{
	nbl::hlsl::uint32_t3 result = nbl::hlsl::gl_WorkGroupID();
	result[ direction ] = IndexInSharedMemory( idx );
	return result;
}


void BoxBlur( 
	const uint32_t ch, 
	const uint32_t direction, 
	const float32_t inRadius, 
	const uint32_t wrapMode, 
	const float32_t4 borderColor,
	BufferAccessor textureAccessor
) {
	using namespace nbl::hlsl;

	float32_t blurred[ ITEMS_PER_THREAD ];
	for( uint32_t i = 0u; i < ITEMS_PER_THREAD; ++i )
	{
		float32_t val = textureAccessor.getPaddedData( getCoordinates( i, direction ), ch );
		blurred[ i ] = val;
	}
		
	for( uint32_t pass = 0u; pass < PASSES_PER_AXIS; ++pass )
	{
		float32_t previousBlockSum = 0.f;

		float32_t spill[ LOCAL_SPILLAGE ];
		for( uint32_t i = 0u; i < LOCAL_SPILLAGE; ++i )
		{
			float32_t scanResult = workgroupPrefixSum( blurred[ i ] ) + previousBlockSum;
			spill[ i ] =  scanResult;
			previousBlockSum = nbl::hlsl::Broadcast( spill[ i ], broadcastAccessor, gl_WorkGroupSize().x - 1u );
		}

		for( uint32_t i = LOCAL_SPILLAGE; i < ITEMS_PER_THREAD; ++i )
		{
			float32_t scanResult = workgroupPrefixSum( blurred[ i ] ) + previousBlockSum;
			previousBlockSum = Broadcast( scanResult, broadcastAccessor, gl_WorkGroupSize().x - 1u );

			uint32_t idx = IndexInSharedMemory( i );
			prefixSumsAccessor.set( idx, nbl::hlsl::bit_cast<uint32_t, float32_t>( scanResult ) );
		}

		for( uint i = 0u; i < LOCAL_SPILLAGE; ++i )
		{
			uint32_t idx = IndexInSharedMemory( i );
			prefixSumsAccessor.set( idx, spill[ i ] );
		}
		prefixSumsAccessor.workgroupExecutionAndMemoryBarrier();

		for( uint32_t i = 0; i < ITEMS_PER_THREAD; ++i )
		{
			uint32_t idx = IndexInSharedMemory( i );

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
					result = lerp( floorRight, ceilRight, fract( right ) );
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
						float32_t sumModFloorRight = prefixSumsAccessor.get( mod( floor( right ) - AXIS_DIM, AXIS_DIM ) );
						float32_t sumModCeilRight = prefixSumsAccessor.get( mod( ceil( right ) - AXIS_DIM, AXIS_DIM ) );
						const float32_t v_floored = ceil( ( floor( right ) - last ) / AXIS_DIM ) * sumLast + sumModFloorRight;
						const float32_t v_ceiled = ceil( ( ceil( right ) - last ) / AXIS_DIM ) * sumLast + sumModCeilRight;
						result = lerp( v_floored, v_ceiled, fract( right ) );
					} break;

					case WRAP_MODE_MIRROR:
					{
						float32_t v_floored;
						{
							const uint32_t floored = uint32_t( floor( right ) );
							const uint32_t d = floored - last; // distance from the right-most boundary, >=0

							if( mod( d, 2 * AXIS_DIM ) == AXIS_DIM )
							{
								v_floored = ( ( d + AXIS_DIM ) / AXIS_DIM ) * prefixSumsAccessor.get( last );
							}
							else
							{
								const uint32_t period = uint32_t( ceil( float32_t( d ) / AXIS_DIM ) );

								if( ( period & 0x1u ) == 1 )
									v_floored = period * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( last - uint32_t( mod( d, AXIS_DIM ) ) );
								else
									v_floored = period * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( mod( d - 1, AXIS_DIM ) );
							}
						}

						float32_t v_ceiled;
						{
							const uint32_t ceiled = uint32_t( ceil( right ) );
							const uint32_t d = ceiled - last; // distance from the right-most boundary, >=0

							if( mod( d, 2 * AXIS_DIM ) == AXIS_DIM )
							{
								v_ceiled = ( ( d + AXIS_DIM ) / AXIS_DIM ) * prefixSumsAccessor.get( last );
							}
							else
							{
								const uint32_t period = uint32_t( ceil( float32_t( d ) / AXIS_DIM ) );

								if( ( period & 0x1u ) == 1 )
									v_ceiled = period * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( last - uint32_t( mod( d, AXIS_DIM ) ) );
								else
									v_ceiled = period * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( mod( d - 1, AXIS_DIM ) );
							}
						}

						result = lerp( v_floored, v_ceiled, fract( right ) );
					} break;
					}
				}

				if( left >= 0 )
				{
					result -= lerp( prefixSumsAccessor.get( floor( left ) ), prefixSumsAccessor.get( ceil( left ) ), fract( left ) );
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
						const float32_t v_floored = floor( floor( left ) / AXIS_DIM ) * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( mod( floor( left ), AXIS_DIM ) );
						const float32_t v_ceiled = floor( ceil( left ) / AXIS_DIM ) * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( mod( ceil( left ), AXIS_DIM ) );
						result -= lerp( v_floored, v_ceiled, fract( left ) );
					} break;

					case WRAP_MODE_MIRROR:
					{
						float32_t v_floored;
						{
							const uint32_t floored = uint32_t( floor( left ) );
							if( mod( abs( floored + 1 ), 2 * AXIS_DIM ) == 0 )
							{
								v_floored = -( abs( floored + 1 ) / AXIS_DIM ) * prefixSumsAccessor.get( last );
							}
							else
							{
								const uint32_t period = uint32_t( ceil( float32_t( abs( floored + 1 ) ) / AXIS_DIM ) );

								if( ( period & 0x1u ) == 1 )
									v_floored = -1 * ( period - 1 ) * prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( mod( abs( floored + 1 ) - 1, AXIS_DIM ) );
								else
									v_floored = -1 * ( period - 1 ) * prefixSumsAccessor.get( last ) - ( prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( mod( floored + 1, AXIS_DIM ) - 1 ) );
							}
						}

						float32_t v_ceiled;
						{
							const uint32_t ceiled = uint32_t( ceil( left ) );
							if( ceiled == 0 ) // Special case, wouldn't be possible for `floored` above
							{
								v_ceiled = 0;
							}
							else if( mod( abs( ceiled + 1 ), 2 * AXIS_DIM ) == 0 )
							{
								v_ceiled = -( abs( ceiled + 1 ) / AXIS_DIM ) * prefixSumsAccessor.get( last );
							}
							else
							{
								const uint32_t period = uint32_t( ceil( float32_t( abs( ceiled + 1 ) ) / AXIS_DIM ) );

								if( ( period & 0x1u ) == 1 )
									v_ceiled = -1 * ( period - 1 ) * prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( mod( abs( ceiled + 1 ) - 1, AXIS_DIM ) );
								else
									v_ceiled = -1 * ( period - 1 ) * prefixSumsAccessor.get( last ) - ( prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( mod( ceiled + 1, AXIS_DIM ) - 1 ) );
							}
						}

						result -= lerp( v_floored, v_ceiled, fract( left ) );
					} break;
					}
				}

				blurred[ i ] = result / ( 2.f * radius + 1.f );
			}
		}

		for( uint32_t i = 0; i < ITEMS_PER_THREAD; ++i )
		{
			textureAcessor.setData( getCoordinates( i, direction ), ch, blurred[ i ] );
		}
	}
}
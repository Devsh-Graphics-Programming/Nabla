// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/asset/utils/CGeometryCreator.h"
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/math/linalg/transform.hlsl"
#include "nbl/builtin/hlsl/math/quaternions.hlsl"

#include <cmath>
#include <cstdint>

namespace nbl::asset
{

namespace
{
using snorm_normal_t = hlsl::vector<int8_t, 4>;
constexpr int8_t snorm_one = std::numeric_limits<int8_t>::max();
constexpr int8_t snorm_neg_one = std::numeric_limits<int8_t>::min();
constexpr auto snorm_positive_x = hlsl::vector<int8_t, 4>(snorm_one, 0, 0, 0);
constexpr auto snorm_negative_x = hlsl::vector<int8_t, 4>(snorm_neg_one, 0, 0, 0);
constexpr auto snorm_positive_y = hlsl::vector<int8_t, 4>(0, snorm_one, 0, 0);
constexpr auto snorm_negative_y = hlsl::vector<int8_t, 4>(0, snorm_neg_one, 0, 0);
constexpr auto snorm_positive_z = hlsl::vector<int8_t, 4>(0, 0, snorm_one, 0);
constexpr auto snorm_negative_z = hlsl::vector<int8_t, 4>(0, 0, snorm_neg_one, 0);

constexpr auto snorm_all_ones = hlsl::vector<int8_t, 4>(snorm_one, snorm_one, snorm_one, snorm_one);

template <typename ElementT>
  requires(std::is_same_v<ElementT, uint8_t> || std::is_same_v<ElementT, uint16_t>)
constexpr E_FORMAT get_uv_format()
{
  if constexpr(std::is_same_v<ElementT, uint8_t>)
  {
    return EF_R8G8_UNORM;
  } else
  {
    return EF_R16G16_UNORM;
  }
}
}

template <typename ElementT>
	requires(std::is_same_v<ElementT, uint8_t> || std::is_same_v<ElementT, uint16_t>)
static ICPUPolygonGeometry::SDataView createUvView(size_t vertexCount)
{
	const auto elementCount = 2;
	const auto attrSize = sizeof(ElementT) * elementCount;
	auto buff = ICPUBuffer::create({{attrSize * vertexCount,IBuffer::EUF_NONE}});
	hlsl::shapes::AABB<4, ElementT> aabb;
	aabb.minVx = hlsl::vector<ElementT, 4>(0,0,0,0);
	aabb.maxVx = hlsl::vector<ElementT, 4>(std::numeric_limits<ElementT>::max(), std::numeric_limits<ElementT>::max(), 0, 0);

	auto retval = ICPUPolygonGeometry::SDataView{
		.composed = {
			.stride = attrSize,
		},
		.src = {
			.offset = 0,
			.size = buff->getSize(),
			.buffer = std::move(buff),
		}
	};

	if constexpr(std::is_same_v<ElementT, uint8_t>)
	{
		retval.composed.encodedDataRange.u8 = aabb;
		retval.composed.format = get_uv_format<ElementT>();
		retval.composed.rangeFormat = IGeometryBase::EAABBFormat::U8_NORM;
	}
	else if constexpr(std::is_same_v<ElementT, uint16_t>)
	{
		retval.composed.encodedDataRange.u16 = aabb;
		retval.composed.format = get_uv_format<ElementT>();
		retval.composed.rangeFormat = IGeometryBase::EAABBFormat::U16_NORM;
	}

	return retval;
}

template <typename IndexT>
	requires(std::is_same_v<IndexT, uint16_t> || std::is_same_v<IndexT, uint32_t>)
static ICPUPolygonGeometry::SDataView createIndexView(size_t indexCount, size_t maxIndex)
{
	const auto bytesize = sizeof(IndexT) * indexCount;
	auto indices = ICPUBuffer::create({bytesize,IBuffer::EUF_INDEX_BUFFER_BIT});

	hlsl::shapes::AABB<4,IndexT> aabb;
	aabb.minVx[0] = 0;
	aabb.maxVx[0] = maxIndex;

	auto retval = ICPUPolygonGeometry::SDataView{
		.composed = {
			.stride = sizeof(IndexT),
		},
		.src = {.offset = 0,.size = bytesize,.buffer = std::move(indices)},
	};

	if constexpr(std::is_same_v<IndexT, uint16_t>)
	{
		retval.composed.encodedDataRange.u16 = aabb;
		retval.composed.format = EF_R16_UINT;
		retval.composed.rangeFormat = IGeometryBase::EAABBFormat::U16;
	}
	else if constexpr(std::is_same_v<IndexT, uint32_t>)
	{
		retval.composed.encodedDataRange.u32 = aabb;
		retval.composed.format = EF_R32_UINT;
		retval.composed.rangeFormat = IGeometryBase::EAABBFormat::U32;
	}

	return retval;
}

template <size_t ElementCountV = 3>
	requires(ElementCountV > 0 && ElementCountV <= 4)
static ICPUPolygonGeometry::SDataView createPositionView(size_t positionCount, const hlsl::shapes::AABB<4, hlsl::float32_t>& aabb)
{
	using position_t = hlsl::vector<hlsl::float32_t, ElementCountV>;
	constexpr auto AttrSize = sizeof(position_t);
	auto buff = ICPUBuffer::create({AttrSize * positionCount,IBuffer::EUF_NONE});

	constexpr auto format = []()
	{
		if constexpr (ElementCountV == 1) return EF_R32_SFLOAT;
		if constexpr (ElementCountV == 2) return EF_R32G32_SFLOAT;
		if constexpr (ElementCountV == 3) return EF_R32G32B32_SFLOAT;
		if constexpr (ElementCountV == 4) return EF_R32G32B32A32_SFLOAT;
	}();

	return {
		.composed = {
			.encodedDataRange = {.f32 = aabb},
			.stride = AttrSize,
			.format = format,
			.rangeFormat = IGeometryBase::EAABBFormat::F32
		},
		.src = {.offset = 0,.size = buff->getSize(),.buffer = std::move(buff)}
	};
}

static ICPUPolygonGeometry::SDataView createSnormNormalView(size_t normalCount, const hlsl::shapes::AABB<4, int8_t>& aabb)
{
	constexpr auto AttrSize = sizeof(snorm_normal_t);
	auto buff = ICPUBuffer::create({AttrSize * normalCount,IBuffer::EUF_NONE});
	return {
		.composed = {
			.encodedDataRange = {.s8=aabb},
			.stride = AttrSize,
			.format = EF_R8G8B8A8_SNORM,
			.rangeFormat = IGeometryBase::EAABBFormat::S8_NORM
		},
		.src = {.offset=0,.size=buff->getSize(),.buffer=std::move(buff)}
	};
}

static void encodeUv(hlsl::vector<uint16_t, 2>* uvDst, hlsl::float32_t2 uvSrc)
{
	uint32_t u32_uv = hlsl::packUnorm2x16(uvSrc);
	memcpy(uvDst, &u32_uv, sizeof(uint16_t) * 2);
}

core::smart_refctd_ptr<ICPUPolygonGeometry> CGeometryCreator::createCube(const hlsl::float32_t3 size) const
{
	using namespace hlsl;

	auto retval = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
	retval->setIndexing(IPolygonGeometryBase::TriangleList());

	constexpr auto CubeUniqueVertices = 24;

	// Create indices
	using index_t = uint16_t;
	{
		constexpr auto IndexCount = 36;
		constexpr auto MaxIndex = CubeUniqueVertices - 1;
		auto indexView = createIndexView<index_t>(IndexCount, MaxIndex);
		auto u = reinterpret_cast<index_t*>(indexView.src.buffer->getPointer());
		for (uint32_t i=0u; i<6u; ++i)
		{
			u[i*6+0] = 4*i+0;
			u[i*6+1] = 4*i+1;
			u[i*6+2] = 4*i+3;
			u[i*6+3] = 4*i+1;
			u[i*6+4] = 4*i+2;
			u[i*6+5] = 4*i+3;
		}
		retval->setIndexView(std::move(indexView));
	}


	// Create vertex attributes with NONE usage because we have no clue how they'll be used
	hlsl::float32_t3* positions;

	// for now because no reliable RGB10A2 encode and scant support for 24-bit UTB formats
	snorm_normal_t* normals;

	using uv_element_t = uint8_t;
	constexpr auto UnityUV = std::numeric_limits<uv_element_t>::max();
	hlsl::vector<uv_element_t,2>* uvs;
	{
		{
			shapes::AABB<4,float32_t> aabb;
			aabb.maxVx = float32_t4(size*0.5f,0.f);
			aabb.minVx = - aabb.maxVx;

			auto positionView = createPositionView(CubeUniqueVertices, aabb);
			positions = reinterpret_cast<decltype(positions)>(positionView.src.buffer->getPointer());
			retval->setPositionView(std::move(positionView));
		}
		{
			shapes::AABB<4,int8_t> aabb;
			aabb.maxVx = snorm_all_ones;
			aabb.minVx = -aabb.maxVx;
			auto normalView = createSnormNormalView(CubeUniqueVertices, aabb);
			normals = reinterpret_cast<decltype(normals)>(normalView.src.buffer->getPointer());
			retval->setNormalView(std::move(normalView));
		}

		{
			auto uvView = createUvView<uv_element_t>(CubeUniqueVertices);
			uvs = reinterpret_cast<decltype(uvs)>(uvView.src.buffer->getPointer());
			retval->getAuxAttributeViews()->push_back(std::move(uvView));
		}
	}

	//
	{
		const hlsl::float32_t3 pos[8] =
		{
			hlsl::float32_t3(-0.5f,-0.5f, 0.5f) * size,
			hlsl::float32_t3(0.5f,-0.5f, 0.5f) * size,
			hlsl::float32_t3(0.5f, 0.5f, 0.5f) * size,
			hlsl::float32_t3(-0.5f, 0.5f, 0.5f) * size,
			hlsl::float32_t3(0.5f,-0.5f,-0.5f) * size,
			hlsl::float32_t3(-0.5f, 0.5f,-0.5f) * size,
			hlsl::float32_t3(-0.5f,-0.5f,-0.5f) * size,
			hlsl::float32_t3(0.5f, 0.5f,-0.5f) * size
		};
		positions[0] = hlsl::float32_t3(pos[0][0], pos[0][1], pos[0][2]);
		positions[1] = hlsl::float32_t3(pos[1][0], pos[1][1], pos[1][2]);
		positions[2] = hlsl::float32_t3(pos[2][0], pos[2][1], pos[2][2]);
		positions[3] = hlsl::float32_t3(pos[3][0], pos[3][1], pos[3][2]);
		positions[4] = hlsl::float32_t3(pos[1][0], pos[1][1], pos[1][2]);
		positions[5] = hlsl::float32_t3(pos[4][0], pos[4][1], pos[4][2]);
		positions[6] = hlsl::float32_t3(pos[7][0], pos[7][1], pos[7][2]);
		positions[7] = hlsl::float32_t3(pos[2][0], pos[2][1], pos[2][2]);
		positions[8] = hlsl::float32_t3(pos[4][0], pos[4][1], pos[4][2]);
		positions[9] = hlsl::float32_t3(pos[6][0], pos[6][1], pos[6][2]);
		positions[10] = hlsl::float32_t3(pos[5][0], pos[5][1], pos[5][2]);
		positions[11] = hlsl::float32_t3(pos[7][0], pos[7][1], pos[7][2]);
		positions[12] = hlsl::float32_t3(pos[6][0], pos[6][1], pos[6][2]);
		positions[13] = hlsl::float32_t3(pos[0][0], pos[0][1], pos[0][2]);
		positions[14] = hlsl::float32_t3(pos[3][0], pos[3][1], pos[3][2]);
		positions[15] = hlsl::float32_t3(pos[5][0], pos[5][1], pos[5][2]);
		positions[16] = hlsl::float32_t3(pos[3][0], pos[3][1], pos[3][2]);
		positions[17] = hlsl::float32_t3(pos[2][0], pos[2][1], pos[2][2]);
		positions[18] = hlsl::float32_t3(pos[7][0], pos[7][1], pos[7][2]);
		positions[19] = hlsl::float32_t3(pos[5][0], pos[5][1], pos[5][2]);
		positions[20] = hlsl::float32_t3(pos[0][0], pos[0][1], pos[0][2]);
		positions[21] = hlsl::float32_t3(pos[6][0], pos[6][1], pos[6][2]);
		positions[22] = hlsl::float32_t3(pos[4][0], pos[4][1], pos[4][2]);
		positions[23] = hlsl::float32_t3(pos[1][0], pos[1][1], pos[1][2]);
	}

	//
	{
		const snorm_normal_t norm[6] =
		{
			snorm_positive_z,
			snorm_positive_x,
			snorm_negative_z,
			snorm_negative_x,
			snorm_positive_y,
			snorm_negative_y
		};
		const hlsl::vector<uv_element_t, 2> uv[4] =
		{
			hlsl::vector<uv_element_t,2>(  0, UnityUV),
			hlsl::vector<uv_element_t,2>(UnityUV, UnityUV),
			hlsl::vector<uv_element_t,2>(UnityUV,  0),
			hlsl::vector<uv_element_t,2>(  0,  0)
		};

		for (size_t f = 0ull; f < 6ull; ++f)
		{
			const size_t v = f * 4ull;

			for (size_t i = 0ull; i < 4ull; ++i)
			{
				normals[v + i] = snorm_normal_t(norm[f]);
				uvs[v + i] = uv[i];
			}
		}
	}

	CPolygonGeometryManipulator::recomputeContentHashes(retval.get());
	return retval;
}

core::smart_refctd_ptr<ICPUPolygonGeometry> CGeometryCreator::createSphere(float radius,
				uint32_t polyCountX, uint32_t polyCountY, CQuantNormalCache* const quantNormalCacheOverride) const
{
	using namespace hlsl;

	CQuantNormalCache* const quantNormalCache = quantNormalCacheOverride == nullptr ? m_params.normalCache.get() : quantNormalCacheOverride;

	if (polyCountX < 2)
		polyCountX = 2;
	if (polyCountY < 2)
		polyCountY = 2;

	const uint32_t polyCountXPitch = polyCountX + 1; // get to same vertex on next level
	const size_t vertexCount = (polyCountXPitch * polyCountY) + 2;

	auto retval = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
	retval->setIndexing(IPolygonGeometryBase::TriangleList());

	// Create indices
	{
		using index_t = uint32_t;

		const auto indexCount = (polyCountX * polyCountY) * 6;
		auto indexView = createIndexView<index_t>(indexCount, vertexCount - 1);
		auto indexPtr = reinterpret_cast<index_t*>(indexView.src.buffer->getPointer());

		uint32_t level = 0;
		size_t indexAddIx = 0;
		for (uint32_t p1 = 0; p1 < polyCountY - 1; ++p1)
		{
			//main quads, top to bottom
			for (uint32_t p2 = 0; p2 < polyCountX - 1; ++p2)
			{
				const uint32_t curr = level + p2;
				indexPtr[indexAddIx++] = curr + polyCountXPitch;
				indexPtr[indexAddIx++] = curr;
				indexPtr[indexAddIx++] = curr + 1;
				indexPtr[indexAddIx++] = curr + polyCountXPitch;
				indexPtr[indexAddIx++] = curr + 1;
				indexPtr[indexAddIx++] = curr + 1 + polyCountXPitch;
			}

			// the connectors from front to end
			indexPtr[indexAddIx++] = level + polyCountX - 1 + polyCountXPitch;
			indexPtr[indexAddIx++] = level + polyCountX - 1;
			indexPtr[indexAddIx++] = level + polyCountX;

			indexPtr[indexAddIx++] = level + polyCountX - 1 + polyCountXPitch;
			indexPtr[indexAddIx++] = level + polyCountX;
			indexPtr[indexAddIx++] = level + polyCountX + polyCountXPitch;
			level += polyCountXPitch;
		}

		const uint32_t polyCountSq = polyCountXPitch * polyCountY; // top point
		const uint32_t polyCountSq1 = polyCountSq + 1; // bottom point
		const uint32_t polyCountSqM1 = (polyCountY - 1) * polyCountXPitch; // last row's first vertex

		for (uint32_t p2 = 0; p2 < polyCountX - 1; ++p2)
		{
			// create triangles which are at the top of the sphere

			indexPtr[indexAddIx++] = polyCountSq;
			indexPtr[indexAddIx++] = p2 + 1;
			indexPtr[indexAddIx++] = p2;

			// create triangles which are at the bottom of the sphere

			indexPtr[indexAddIx++] = polyCountSqM1 + p2;
			indexPtr[indexAddIx++] = polyCountSqM1 + p2 + 1;
			indexPtr[indexAddIx++] = polyCountSq1;
		}

		// create final triangle which is at the top of the sphere

		indexPtr[indexAddIx++] = polyCountSq;
		indexPtr[indexAddIx++] = polyCountX;
		indexPtr[indexAddIx++] = polyCountX - 1;

		// create final triangle which is at the bottom of the sphere

		indexPtr[indexAddIx++] = polyCountSqM1 + polyCountX - 1;
		indexPtr[indexAddIx++] = polyCountSqM1;
		indexPtr[indexAddIx++] = polyCountSq1;

		retval->setIndexView(std::move(indexView));

	}

	constexpr auto NormalCacheFormat = EF_R8G8B8_SNORM;

	// Create vertex attributes with NONE usage because we have no clue how they'll be used
	hlsl::float32_t3* positions;

	snorm_normal_t* normals;

	using uv_element_t = uint16_t;
	constexpr auto UnityUV = std::numeric_limits<uv_element_t>::max();

	hlsl::vector<uv_element_t, 2>* uvs;
	{
		{
			shapes::AABB<4, float32_t> aabb;
			aabb.maxVx = float32_t4(radius, radius, radius, 0.0f);
			aabb.minVx = float32_t4(-radius, -radius, -radius, 0.0f);
			auto positionView = createPositionView(vertexCount, aabb);
			positions = reinterpret_cast<decltype(positions)>(positionView.src.buffer->getPointer());
			retval->setPositionView(std::move(positionView));
		}
		{
			shapes::AABB<4, int8_t> aabb;
			aabb.maxVx = snorm_all_ones;
			aabb.minVx = -aabb.maxVx;
			auto normalView = createSnormNormalView(vertexCount, aabb);
			normals = reinterpret_cast<decltype(normals)>(normalView.src.buffer->getPointer());
			retval->setNormalView(std::move(normalView));
		}
		{
			auto uvView = createUvView<uv_element_t>(vertexCount);
			uvs = reinterpret_cast<decltype(uvs)>(uvView.src.buffer->getPointer());
			retval->getAuxAttributeViews()->push_back(std::move(uvView));
		}
	}

	// fill vertices
	{
		// calculate the angle which separates all points in a circle
		const float AngleX = 2 * core::PI<float>() / polyCountX;
		const float AngleY = core::PI<float>() / polyCountY;

		double axz;

		// we don't start at 0.

		double ay = 0;//AngleY / 2;
		auto vertex_i = 0;
		for (uint32_t y = 0; y < polyCountY; ++y)
		{
			ay += AngleY;
			const double sinay = sin(ay);
			axz = 0;

			// calculate the necessary vertices without the doubled one
			const auto old_vertex_i = vertex_i;
			for (uint32_t xz = 0; xz < polyCountX; ++xz)
			{
				// calculate points position

				float32_t3 pos(static_cast<float>(cos(axz) * sinay),
					static_cast<float>(cos(ay)),
					static_cast<float>(sin(axz) * sinay));
				// for spheres the normal is the position
				const auto normal = pos;
				const auto quantizedNormal = quantNormalCache->quantize<NormalCacheFormat>(normal);
				pos *= radius;

				// calculate texture coordinates via sphere mapping
				// tu is the same on each level, so only calculate once
				float tu = 0.5f;
				//if (y==0)
				//{
				if (normal.y != -1.0f && normal.y != 1.0f)
					tu = static_cast<float>(acos(core::clamp(normal.x / sinay, -1.0, 1.0)) * 0.5 * numbers::inv_pi<float32_t>);
				if (normal.z < 0.0f)
					tu = 1 - tu;
				//}
				//else
					//tu = ((float*)(tmpMem+(i-polyCountXPitch)*vertexSize))[4];

				positions[vertex_i] = pos;
				encodeUv(uvs + vertex_i, float32_t2(tu, static_cast<float>(ay* numbers::inv_pi<float32_t>)));
				memcpy(normals + vertex_i, &quantizedNormal, sizeof(quantizedNormal));

				vertex_i++;
				axz += AngleX;
			}
			// This is the doubled vertex on the initial position

			positions[vertex_i] = positions[old_vertex_i];
			uvs[vertex_i] = { UnityUV, uvs[old_vertex_i].y };
			normals[vertex_i] = normals[old_vertex_i];

			vertex_i++;
		}

		// the vertex at the top of the sphere
		positions[vertex_i] = { 0.f, radius, 0.f };
		uvs[vertex_i] = { 0, UnityUV / 2};
		const auto quantizedTopNormal = quantNormalCache->quantize<NormalCacheFormat>(hlsl::float32_t3(0.f, 1.f, 0.f));
		memcpy(normals + vertex_i, &quantizedTopNormal, sizeof(quantizedTopNormal));

		// the vertex at the bottom of the sphere
		vertex_i++;
		positions[vertex_i] = { 0.f, -radius, 0.f };
		uvs[vertex_i] = { UnityUV / 2, UnityUV};
		const auto quantizedBottomNormal = quantNormalCache->quantize<NormalCacheFormat>(hlsl::float32_t3(0.f, -1.f, 0.f));
		memcpy(normals + vertex_i, &quantizedBottomNormal, sizeof(quantizedBottomNormal));
	}

	CPolygonGeometryManipulator::recomputeContentHashes(retval.get());
	return retval;
}

core::smart_refctd_ptr<ICPUPolygonGeometry> CGeometryCreator::createCylinder(
	float radius, float length,
	uint16_t tesselation, CQuantNormalCache* const quantNormalCacheOverride) const
{
	using namespace hlsl;

	CQuantNormalCache* const quantNormalCache = quantNormalCacheOverride == nullptr ? m_params.normalCache.get() : quantNormalCacheOverride;

	const auto halfIx = tesselation;
	const uint32_t u32_vertexCount = 2 * tesselation;
	if (u32_vertexCount > std::numeric_limits<uint16_t>::max())
		return nullptr;
	const auto vertexCount = static_cast<uint16_t>(u32_vertexCount);

	auto retval = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
	retval->setIndexing(IPolygonGeometryBase::TriangleList());

	// Create indices
	using index_t = uint16_t;
	{
		constexpr uint32_t RowCount = 2u;
		const auto IndexCount = RowCount * 3 * tesselation;
		auto indexView = createIndexView<index_t>(IndexCount, vertexCount - 1);
		auto u = reinterpret_cast<index_t*>(indexView.src.buffer->getPointer());

		for (uint16_t i = 0u, j = 0u; i < halfIx; ++i)
		{
			u[j++] = i;
			u[j++] = (i + 1u) != halfIx ? (i + 1u):0u;
			u[j++] = i + halfIx;
			u[j++] = i + halfIx;
			u[j++] = (i + 1u)!= halfIx ? (i + 1u):0u;
			u[j++] = (i + 1u)!= halfIx ? (i + 1u + halfIx) : halfIx;
		}

		retval->setIndexView(std::move(indexView));
	}

	constexpr auto NormalCacheFormat = EF_R8G8B8_SNORM;

	// Create vertex attributes with NONE usage because we have no clue how they'll be used
	hlsl::float32_t3* positions;

	snorm_normal_t* normals;

	using uv_element_t = uint16_t;
	constexpr auto UnityUV = std::numeric_limits<uv_element_t>::max();
	hlsl::vector<uv_element_t, 2>* uvs;
	{
		{
			shapes::AABB<4, float32_t> aabb;
			aabb.maxVx = float32_t4(radius, radius, length, 0.0f);
			aabb.minVx = float32_t4(-radius, -radius, 0.0f, 0.0f);
			auto positionView = createPositionView(vertexCount, aabb);
			positions = reinterpret_cast<decltype(positions)>(positionView.src.buffer->getPointer());
			retval->setPositionView(std::move(positionView));
		}
		{
			shapes::AABB<4, int8_t> aabb;
			aabb.maxVx = hlsl::vector<int8_t,4>(127,127,127,0);
			aabb.minVx = -aabb.maxVx;
			auto normalView = createSnormNormalView(vertexCount, aabb);
			normals = reinterpret_cast<decltype(normals)>(normalView.src.buffer->getPointer());
			retval->setNormalView(std::move(normalView));
		}
		{
			auto uvView = createUvView<uv_element_t>(vertexCount);
			uvs = reinterpret_cast<decltype(uvs)>(uvView.src.buffer->getPointer());
			retval->getAuxAttributeViews()->push_back(std::move(uvView));
		}
	}

	const float tesselationRec = 1.f / static_cast<float>(tesselation);
	const float step = 2.f * numbers::pi<float32_t> * tesselationRec;
	for (uint32_t i = 0u; i < tesselation; ++i)
	{
		const auto f_i = static_cast<float>(i);
		hlsl::float32_t3 p(std::cos(f_i * step), std::sin(f_i * step), 0.f);
		const auto n = quantNormalCache->quantize<NormalCacheFormat>(p);
		p *= radius;

		positions[i] = { p.x, p.y, p.z };
		memcpy(normals + i, &n, sizeof(n));
		encodeUv(uvs + i, float32_t2(f_i * tesselationRec, 0.f));

		positions[i + halfIx] = { p.x, p.y, length };
		normals[i + halfIx] = normals[i];
		uvs[i + halfIx] = { 1.f * tesselationRec, UnityUV };
	}

	CPolygonGeometryManipulator::recomputeContentHashes(retval.get());
	return retval;
}

core::smart_refctd_ptr<ICPUPolygonGeometry> CGeometryCreator::createCone(
	float radius, float length, uint16_t tesselation,
	float oblique, CQuantNormalCache* const quantNormalCacheOverride) const
{

	using namespace hlsl;

	const uint32_t u32_vertexCount = tesselation + 1;
	if (u32_vertexCount > std::numeric_limits<uint16_t>::max())
		return nullptr;
	const auto vertexCount = static_cast<uint16_t>(u32_vertexCount);

	auto retval = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
	retval->setIndexing(IPolygonGeometryBase::TriangleList());

	// Create indices
	using index_t = uint16_t;
	{
		const auto IndexCount = 3 * tesselation;

		auto indexView = createIndexView<index_t>(IndexCount, vertexCount - 1);
		auto u = reinterpret_cast<index_t*>(indexView.src.buffer->getPointer());

		const uint32_t apexVertexIndex = tesselation;

		for (uint32_t i = 0; i < tesselation; i++)
		{
			u[i * 3] = apexVertexIndex;
			u[(i * 3) + 1] = i;
			u[(i * 3) + 2] = i == (tesselation - 1) ? 0 : i + 1;
		}

		retval->setIndexView(std::move(indexView));
	}

	// Create vertex attributes with NONE usage because we have no clue how they'll be used
	hlsl::float32_t3* positions;
	{
		{
			shapes::AABB<4, float32_t> aabb;
			aabb.maxVx = float32_t4(radius, radius, length, 0.0f);
			aabb.minVx = float32_t4(-radius, -radius, 0.0f, 0.0f);
			auto positionView = createPositionView(vertexCount, aabb);
			positions = reinterpret_cast<decltype(positions)>(positionView.src.buffer->getPointer());
			retval->setPositionView(std::move(positionView));
		}
	}

	const float step = (2.f*core::PI<float>()) / tesselation;

	const hlsl::float32_t3 apexVertexCoords(oblique, length, 0.0f);

	const auto apexVertexBase_i = tesselation;

	for (uint32_t i = 0u; i < tesselation; i++)
	{
		hlsl::float32_t3 v(std::cos(i * step), 0.0f, std::sin(i * step));
		v *= radius;
		positions[i] = v;
	}
	positions[apexVertexBase_i] = apexVertexCoords;

	CPolygonGeometryManipulator::recomputeContentHashes(retval.get());
	return retval;
}

core::smart_refctd_ptr<ICPUPolygonGeometry> CGeometryCreator::createPrism(
	float radius, float length, uint16_t sideCount) const
{
	using namespace hlsl;

	if (sideCount < 3) return nullptr;

	const auto halfIx = sideCount;
	const uint32_t u32_vertexCount = 2 * sideCount;
	if (u32_vertexCount > std::numeric_limits<uint16_t>::max())
		return nullptr;
	const auto vertexCount = static_cast<uint16_t>(u32_vertexCount);

	auto retval = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
	retval->setIndexing(IPolygonGeometryBase::TriangleList());

	// Create indices
	using index_t = uint16_t;
	{
		constexpr uint32_t RowCount = 2u;
		const auto IndexCount = RowCount * 3 * sideCount;
		auto indexView = createIndexView<index_t>(IndexCount, vertexCount - 1);
		auto u = reinterpret_cast<index_t*>(indexView.src.buffer->getPointer());

		for (uint16_t i = 0u, j = 0u; i < halfIx; ++i)
		{
			u[j++] = i;
			u[j++] = (i + 1u) != halfIx ? (i + 1u):0u;
			u[j++] = i + halfIx;
			u[j++] = i + halfIx;
			u[j++] = (i + 1u)!= halfIx ? (i + 1u):0u;
			u[j++] = (i + 1u)!= halfIx ? (i + 1u + halfIx) : halfIx;
		}

		retval->setIndexView(std::move(indexView));
	}

	// Create vertex attributes with NONE usage because we have no clue how they'll be used
	hlsl::float32_t3* positions;

  {
    shapes::AABB<4, float32_t> aabb;
    aabb.maxVx = float32_t4(radius, radius, length, 0.0f);
    aabb.minVx = float32_t4(-radius, -radius, 0.0f, 0.0f);
    auto positionView = createPositionView(vertexCount, aabb);
    positions = reinterpret_cast<decltype(positions)>(positionView.src.buffer->getPointer());
    retval->setPositionView(std::move(positionView));
  }

	const float invSideCount = 1.f / static_cast<float>(sideCount);
	const float step = 2.f * numbers::pi<float32_t> * invSideCount;
	for (uint32_t i = 0u; i < sideCount; ++i)
	{
		const auto f_i = static_cast<float>(i);
		hlsl::float32_t3 p(std::cos(f_i * step), std::sin(f_i * step), 0.f);
		p *= radius;

		positions[i] = { p.x, p.y, p.z };

		positions[i + halfIx] = { p.x, p.y, length };
	}

	CPolygonGeometryManipulator::recomputeContentHashes(retval.get());
	return retval;
}

core::smart_refctd_ptr<ICPUGeometryCollection> CGeometryCreator::createArrow(
	const uint16_t tesselationCylinder,
	const uint16_t tesselationCone,
	const float height,
	const float cylinderHeight,
	const float width0,
	const float width1
) const
{
	assert(height > cylinderHeight);

	auto cylinder = createCylinder(width0, cylinderHeight, tesselationCylinder);
	auto cone = createCone(width1, height-cylinderHeight, tesselationCone);

	auto collection = core::make_smart_refctd_ptr<ICPUGeometryCollection>();
	auto* geometries = collection->getGeometries();
	geometries->push_back({
		.geometry = cylinder
	});
	const auto coneRotation = hlsl::math::quaternion<hlsl::float32_t>::create(hlsl::float32_t3(1.f, 0.f, 0.f), hlsl::numbers::pi<hlsl::float32_t> * -0.5f);
	const auto coneTransform = hlsl::math::linalg::promote_affine<3, 4>(hlsl::_static_cast<hlsl::float32_t3x3>(coneRotation));
	geometries->push_back({
		.transform = hlsl::math::linalg::promote_affine<3, 4>(coneTransform),
		.geometry = cone
	});
	return collection;

}

core::smart_refctd_ptr<ICPUPolygonGeometry> CGeometryCreator::createRectangle(const hlsl::float32_t2 size) const
{
	using namespace hlsl;

	auto retval = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
	retval->setIndexing(IPolygonGeometryBase::TriangleList());
	
	// Create indices
	{
		using index_t = uint16_t;
		/*
		0---1
		| / |
		3---2
		*/
		const index_t indices[] = {0,3,1,1,3,2};
		auto indexView = createIndexView<index_t>(std::size(indices), 3);
		memcpy(indexView.src.buffer->getPointer(), indices, sizeof(indices));
		retval->setIndexView(std::move(indexView));
	}

	constexpr auto VertexCount = 4;
	// Create vertices
	{
		{
			const hlsl::float32_t2 positions[VertexCount] = {
				hlsl::float32_t2(-size.x, size.y),
				hlsl::float32_t2( size.x, size.y),
				hlsl::float32_t2( size.x,-size.y),
				hlsl::float32_t2(-size.x,-size.y)
			};
			shapes::AABB<4,float32_t> aabb;
			aabb.minVx = float32_t4(-size,0.f,0.f);
			aabb.maxVx = float32_t4( size,0.f,0.f);
			auto positionView = createPositionView<2>(VertexCount, aabb);
			memcpy(positionView.src.buffer->getPointer(), positions, sizeof(positions));
			retval->setPositionView(std::move(positionView));
		}
		{
			const hlsl::vector<int8_t,4> normals[VertexCount] = {
				snorm_positive_z,
				snorm_positive_z,
				snorm_positive_z,
				snorm_positive_z,
			};
			shapes::AABB<4,int8_t> aabb;
			aabb.maxVx = snorm_positive_z;
			aabb.minVx = snorm_normal_t(0, 0, 0, 0);
			auto normalView = createSnormNormalView(VertexCount, aabb);
			memcpy(normalView.src.buffer->getPointer(), normals, sizeof(normals));
			retval->setNormalView(std::move(normalView));
		}
		{
			using uv_element_t = uint8_t;
			constexpr auto MaxUvVal = std::numeric_limits<uv_element_t>::max();
			const hlsl::vector<uv_element_t, 2> uvsData[VertexCount] = {
				hlsl::vector<uv_element_t,2>(  0, MaxUvVal),
				hlsl::vector<uv_element_t,2>(MaxUvVal, MaxUvVal),
				hlsl::vector<uv_element_t,2>(MaxUvVal,  0),
				hlsl::vector<uv_element_t,2>(  0,  0)
			};
			hlsl::vector<uv_element_t, 2>* uvs;
			auto uvView = createUvView<uv_element_t>(VertexCount);
			uvs = reinterpret_cast<decltype(uvs)>(uvView.src.buffer->getPointer());
			memcpy(uvs, uvsData, sizeof(uvsData));
			retval->getAuxAttributeViews()->push_back(std::move(uvView));
		}
	}

	CPolygonGeometryManipulator::recomputeContentHashes(retval.get());
	return retval;
}

core::smart_refctd_ptr<ICPUPolygonGeometry> CGeometryCreator::createDisk(const float radius, const uint32_t tesselation) const
{
	// need at least 120 external angles in the fan
	if (tesselation<2)
		return nullptr;

	using namespace hlsl;

	auto retval = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
	retval->setIndexing(IPolygonGeometryBase::TriangleFan());

	// without index buffer
	const size_t vertexCount = 2u + tesselation;

	float32_t2* positions;

	// for now because no reliable RGB10A2 encode and scant support for 24-bit UTB formats
	snorm_normal_t* normals;
	//
	using uv_element_t = uint16_t;
	constexpr uint16_t UnityUV = std::numeric_limits<uv_element_t>::max();
	hlsl::vector<uv_element_t, 2>* uvs;
	{
		{
			shapes::AABB<4,float32_t> aabb;
			aabb.maxVx = float32_t4(radius,radius, 0.f, 0.f);
			aabb.minVx = -aabb.maxVx;
			auto positionView = createPositionView<2>(vertexCount, aabb);
			positions = reinterpret_cast<decltype(positions)>(positionView.src.buffer->getPointer());
			retval->setPositionView(std::move(positionView));
		}
		{
			constexpr auto AttrSize = sizeof(decltype(*normals));
			auto buff = ICPUBuffer::create({AttrSize*vertexCount,IBuffer::EUF_NONE});
			shapes::AABB<4,int8_t> aabb;
			aabb.maxVx = snorm_positive_z;
			aabb.minVx = -aabb.maxVx;
			auto normalView = createSnormNormalView(vertexCount, aabb);
			normals = reinterpret_cast<decltype(normals)>(normalView.src.buffer->getPointer());
			retval->setNormalView(std::move(normalView));
		}
		{
			auto uvView = createUvView<uv_element_t>(vertexCount);
			uvs = reinterpret_cast<decltype(uvs)>(uvView.src.buffer->getPointer());
			retval->getAuxAttributeViews()->push_back(std::move(uvView));
		}
	}

	// populate data
	{
		const float angle = 360.f / static_cast<float>(tesselation);
		// center
		*(positions++) = float32_t2(0.f,0.f);
		*(uvs++) = uint16_t2(0,UnityUV);
		// last
		positions[tesselation] = float32_t3(0.f,radius,0.f);
		uvs[tesselation] = uint16_t2(UnityUV,0);
		for (auto i=0; i<tesselation; i++)
		{
			const float t = float(i)/float(tesselation);
			const float rad = t * 2.f * hlsl::numbers::pi<float>;
			*(positions++) = float32_t2(hlsl::sin(rad),hlsl::cos(rad))*radius;
			*(uvs++) = uint16_t2(t*UnityUV+0.5f,0);
		}
	}
	std::fill_n(normals,vertexCount, snorm_positive_z);

	CPolygonGeometryManipulator::recomputeContentHashes(retval.get());
	return retval;
}

/*
	Helpful Icosphere class implementation used to compute
	and create icopshere's vertices and indecies.

	Polyhedron subdividing icosahedron (20 tris) by N-times iteration
		The icosphere with N=1 (default) has 80 triangles by subdividing a triangle
		of icosahedron into 4 triangles. If N=0, it is identical to icosahedron.
*/

class Icosphere
{
public:
	using index_t = uint32_t;

	Icosphere(float radius = 1.0f, int subdivision = 1, bool smooth = false) : radius(radius), subdivision(subdivision), smooth(smooth)
	{
		if (smooth)
			buildVerticesSmooth();
		else
			buildVerticesFlat();
	}

	~Icosphere() {}

	unsigned int getVertexCount() const { return (unsigned int)vertices.size() / 3; }
	unsigned int getIndexCount() const { return (unsigned int)indices.size(); }
	unsigned int getLineIndexCount() const { return (unsigned int)lineIndices.size(); }
	unsigned int getTriangleCount() const { return getIndexCount() / 3; }

	unsigned int getPositionSize() const { return (unsigned int)vertices.size() * sizeof(float); }   // # of bytes
	unsigned int getNormalSize() const { return (unsigned int)normals.size() * sizeof(float); }
	unsigned int getTexCoordSize() const { return (unsigned int)texCoords.size() * sizeof(float); }
	unsigned int getIndexSize() const { return (unsigned int)indices.size() * sizeof(index_t); }
	unsigned int getLineIndexSize() const { return (unsigned int)lineIndices.size() * sizeof(unsigned int); }

	const float* getPositions() const { return vertices.data(); }
	const float* getNormals() const { return normals.data(); }
	const float* getTexCoords() const { return texCoords.data(); }
	const unsigned int* getIndices() const { return indices.data(); }
	const unsigned int* getLineIndices() const { return lineIndices.data(); }

protected:

private:

	/*
		return face normal (4th param) of a triangle v1-v2-v3
		if a triangle has no surface (normal length = 0), then return a zero vector
	*/

	static inline void computeFaceNormal(const float v1[3], const float v2[3], const float v3[3], float normal[3])
	{
		constexpr float EPSILON = 0.000001f;

		// default return value (0, 0, 0)
		normal[0] = normal[1] = normal[2] = 0;

		// find 2 edge vectors: v1-v2, v1-v3
		float ex1 = v2[0] - v1[0];
		float ey1 = v2[1] - v1[1];
		float ez1 = v2[2] - v1[2];
		float ex2 = v3[0] - v1[0];
		float ey2 = v3[1] - v1[1];
		float ez2 = v3[2] - v1[2];

		// cross product: e1 x e2
		float nx, ny, nz;
		nx = ey1 * ez2 - ez1 * ey2;
		ny = ez1 * ex2 - ex1 * ez2;
		nz = ex1 * ey2 - ey1 * ex2;

		// normalize only if the length is > 0
		float length = sqrtf(nx * nx + ny * ny + nz * nz);
		if (length > EPSILON)
		{
			// normalize
			float lengthInv = 1.0f / length;
			normal[0] = nx * lengthInv;
			normal[1] = ny * lengthInv;
			normal[2] = nz * lengthInv;
		}
	}

	/*
		return vertex normal (2nd param) by mormalizing the vertex vector
	*/

	static inline void computeVertexNormal(const float v[3], float normal[3])
	{
		// normalize
		float scale = Icosphere::computeScaleForLength(v, 1);
		normal[0] = v[0] * scale;
		normal[1] = v[1] * scale;
		normal[2] = v[2] * scale;
	}

	/*
		get the scale factor for vector to resize to the given length of vector
	*/

	static inline float computeScaleForLength(const float v[3], float length)
	{
		// and normalize the vector then re-scale to new radius
		return length / sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	}

	/*
		find middle point of 2 vertices
		NOTE: new vertex must be resized, so the length is equal to the given length
	*/

	static inline void computeHalfVertex(const float v1[3], const float v2[3], float length, float newV[3])
	{
		newV[0] = v1[0] + v2[0];
		newV[1] = v1[1] + v2[1];
		newV[2] = v1[2] + v2[2];
		float scale = Icosphere::computeScaleForLength(newV, length);
		newV[0] *= scale;
		newV[1] *= scale;
		newV[2] *= scale;
	}

	/*
		find middle texcoords of 2 tex coords and return new coord (3rd param)
	*/

	static inline void computeHalfTexCoord(const float t1[2], const float t2[2], float newT[2])
	{
		newT[0] = (t1[0] + t2[0]) * 0.5f;
		newT[1] = (t1[1] + t2[1]) * 0.5f;
	}

	/*
		This function used 20 non-shared line segments to determine if the given
		texture coordinate is shared or no. If it is on the line segments, it is also
		non-shared point

			 00  01  02  03  04         
			 /\  /\  /\  /\  /\         
			/  \/  \/  \/  \/  \        
		 05  06  07  08  09   \       
			 \   10  11  12  13  14     
			\  /\  /\  /\  /\  /      
			 \/  \/  \/  \/  \/       
				15  16  17  18  19      
	*/

	static inline bool isSharedTexCoord(const float t[2])
	{
		// 20 non-shared line segments
		const float S = 1.0f / 11;  // texture steps
		const float T = 1.0f / 3;
		static float segments[] = { S, 0,       0, T,			// 00 - 05
									S, 0,       S * 2, T,		// 00 - 06
									S * 3, 0,     S * 2, T,     // 01 - 06
									S * 3, 0,     S * 4, T,     // 01 - 07
									S * 5, 0,     S * 4, T,     // 02 - 07
									S * 5, 0,     S * 6, T,     // 02 - 08
									S * 7, 0,     S * 6, T,     // 03 - 08
									S * 7, 0,     S * 8, T,     // 03 - 09
									S * 9, 0,     S * 8, T,     // 04 - 09
									S * 9, 0,     1, T * 2,     // 04 - 14
									0, T,       S * 2, 1,		// 05 - 15
									S * 3, T * 2,   S * 2, 1,   // 10 - 15
									S * 3, T * 2,   S * 4, 1,   // 10 - 16
									S * 5, T * 2,   S * 4, 1,   // 11 - 16
									S * 5, T * 2,   S * 6, 1,   // 11 - 17
									S * 7, T * 2,   S * 6, 1,   // 12 - 17
									S * 7, T * 2,   S * 8, 1,   // 12 - 18
									S * 9, T * 2,   S * 8, 1,   // 13 - 18
									S * 9, T * 2,   S * 10, 1,  // 13 - 19
									1, T * 2,     S * 10, 1 };  // 14 - 19

		// check the point with all 20 line segments
		// if it is on the line segment, it is non-shared
		int32_t count = (sizeof(segments) / sizeof(segments[0]));
		for (int32_t i = 0, j = 2; i < count; i += 4, j += 4)
		{
			if (Icosphere::isOnLineSegment(&segments[i], &segments[j], t))
				return false;   // not shared
		}

		return true;
	}

	/*
		determine a point c is on the line segment a-b
	*/

	static inline bool isOnLineSegment(const float a[2], const float b[2], const float c[2])
	{
		constexpr float EPSILON = 0.0001f;

		// cross product must be 0 if c is on the line
		float cross = ((b[0] - a[0]) * (c[1] - a[1])) - ((b[1] - a[1]) * (c[0] - a[0]));
		if (cross > EPSILON || cross < -EPSILON)
			return false;

		// c must be within a-b
		if ((c[0] > a[0] && c[0] > b[0]) || (c[0] < a[0] && c[0] < b[0]))
			return false;
		if ((c[1] > a[1] && c[1] > b[1]) || (c[1] < a[1] && c[1] < b[1]))
			return false;

		return true;    // all passed, it is on the line segment
	}

	void updateRadius()
	{
		float scale = computeScaleForLength(&vertices[0], radius);

		std::size_t i, j;
		std::size_t count = vertices.size();
		for (i = 0, j = 0; i < count; i += 3, j += 8)
		{
			vertices[i] *= scale;
			vertices[i + 1] *= scale;
			vertices[i + 2] *= scale;
		}
	}

	/*
		compute 12 vertices of icosahedron using spherical coordinates
		The north pole is at (0, 0, r) and the south pole is at (0,0,-r).
		5 vertices are placed by rotating 72 deg at elevation 26.57 deg (=atan(1/2))
		5 vertices are placed by rotating 72 deg at elevation -26.57 deg
	*/

	core::vector<float> computeIcosahedronVertices()		
	{
		const float PI = acos(-1);
		const float H_ANGLE = PI / 180 * 72;    // 72 degree = 360 / 5
		const float V_ANGLE = atanf(1.0f / 2);  // elevation = 26.565 degree

		core::vector<float> vertices(12 * 3);    // 12 vertices
		int32_t i1, i2;                             // indices
		float z, xy;                            // coords
		float hAngle1 = -PI / 2 - H_ANGLE / 2;  // start from -126 deg at 2nd row
		float hAngle2 = -PI / 2;                // start from -90 deg at 3rd row

		// the first top vertex (0, 0, r)
		vertices[0] = 0;
		vertices[1] = 0;
		vertices[2] = radius;

		// 10 vertices at 2nd and 3rd rows
		for (auto i = 1; i <= 5; ++i)
		{
			i1 = i * 3;         // for 2nd row
			i2 = (i + 5) * 3;   // for 3rd row

			z = radius * sinf(V_ANGLE);             // elevaton
			xy = radius * cosf(V_ANGLE);

			vertices[i1] = xy * cosf(hAngle1);      // x
			vertices[i2] = xy * cosf(hAngle2);
			vertices[i1 + 1] = xy * sinf(hAngle1);  // x
			vertices[i2 + 1] = xy * sinf(hAngle2);
			vertices[i1 + 2] = z;                   // z
			vertices[i2 + 2] = -z;

			// next horizontal angles
			hAngle1 += H_ANGLE;
			hAngle2 += H_ANGLE;
		}

		// the last bottom vertex (0, 0, -r)
		i1 = 11 * 3;
		vertices[i1] = 0;
		vertices[i1 + 1] = 0;
		vertices[i1 + 2] = -radius;

		return vertices;
	}
	/*
		generate vertices with flat shading
		each triangle is independent (no shared vertices)
	*/

	void buildVerticesFlat()
	{
		//const float S_STEP = 1 / 11.0f;         // horizontal texture step
		//const float T_STEP = 1 / 3.0f;          // vertical texture step
		const float S_STEP = 186 / 2048.0f;		  // horizontal texture step
		const float T_STEP = 322 / 1024.0f;		  // vertical texture step

		// compute 12 vertices of icosahedron
		core::vector<float> tmpVertices = computeIcosahedronVertices();

		// clear memory of prev arrays
		core::vector<float>().swap(vertices);
		core::vector<float>().swap(normals);
		core::vector<float>().swap(texCoords);
		core::vector<uint32_t>().swap(indices);
		core::vector<uint32_t>().swap(lineIndices);

		const float* v0, * v1, * v2, * v3, * v4, * v11;          // vertex positions
		float n[3];												 // face normal
		float t0[2], t1[2], t2[2], t3[2], t4[2], t11[2];         // texCoords
		uint32_t index = 0;

		// compute and add 20 tiangles of icosahedron first
		v0 = &tmpVertices[0];       // 1st vertex
		v11 = &tmpVertices[11 * 3]; // 12th vertex
		for (auto i = 1; i <= 5; ++i)
		{
			// 4 vertices in the 2nd row
			v1 = &tmpVertices[i * 3];
			if (i < 5)
				v2 = &tmpVertices[(i + 1) * 3];
			else
				v2 = &tmpVertices[3];

			v3 = &tmpVertices[(i + 5) * 3];
			if ((i + 5) < 10)
				v4 = &tmpVertices[(i + 6) * 3];
			else
				v4 = &tmpVertices[6 * 3];

			// texture coords
			t0[0] = (2 * i - 1) * S_STEP;   t0[1] = 0;
			t1[0] = (2 * i - 2) * S_STEP;   t1[1] = T_STEP;
			t2[0] = (2 * i - 0) * S_STEP;   t2[1] = T_STEP;
			t3[0] = (2 * i - 1) * S_STEP;   t3[1] = T_STEP * 2;
			t4[0] = (2 * i + 1) * S_STEP;   t4[1] = T_STEP * 2;
			t11[0] = 2 * i * S_STEP;         t11[1] = T_STEP * 3;

			// add a triangle in 1st row
			Icosphere::computeFaceNormal(v0, v1, v2, n);
			addVertices(v0, v1, v2);
			addNormals(n, n, n);
			addTexCoords(t0, t1, t2);
			addIndices(index, index + 1, index + 2);

			// add 2 triangles in 2nd row
			Icosphere::computeFaceNormal(v1, v3, v2, n);
			addVertices(v1, v3, v2);
			addNormals(n, n, n);
			addTexCoords(t1, t3, t2);
			addIndices(index + 3, index + 4, index + 5);

			Icosphere::computeFaceNormal(v2, v3, v4, n);
			addVertices(v2, v3, v4);
			addNormals(n, n, n);
			addTexCoords(t2, t3, t4);
			addIndices(index + 6, index + 7, index + 8);

			// add a triangle in 3rd row
			Icosphere::computeFaceNormal(v3, v11, v4, n);
			addVertices(v3, v11, v4);
			addNormals(n, n, n);
			addTexCoords(t3, t11, t4);
			addIndices(index + 9, index + 10, index + 11);

			// add 6 edge lines per iteration
			//  i
			//  /   /   /   /   /       : (i, i+1)                              
			// /__ /__ /__ /__ /__                                              
			// \  /\  /\  /\  /\  /     : (i+3, i+4), (i+3, i+5), (i+4, i+5)    
			//  \/__\/__\/__\/__\/__                                            
			//   \   \   \   \   \      : (i+9,i+10), (i+9, i+11)               
			//    \   \   \   \   \                                             
			lineIndices.push_back(index);		  // (i, i+1)
			lineIndices.push_back(index + 1);     // (i, i+1)
			lineIndices.push_back(index + 3);     // (i+3, i+4)
			lineIndices.push_back(index + 4);
			lineIndices.push_back(index + 3);     // (i+3, i+5)
			lineIndices.push_back(index + 5);
			lineIndices.push_back(index + 4);     // (i+4, i+5)
			lineIndices.push_back(index + 5);
			lineIndices.push_back(index + 9);     // (i+9, i+10)
			lineIndices.push_back(index + 10);
			lineIndices.push_back(index + 9);     // (i+9, i+11)
			lineIndices.push_back(index + 11);

			// next index
			index += 12;
		}

		// subdivide icosahedron
		subdivideVerticesFlat();
	}

	/*
		 generate vertices with smooth shading
		 NOTE: The north and south pole vertices cannot be shared for smooth shading
		 because they have same position and normal, but different texcoords per face
		 And, the first vertex on each row is also not shared.
	*/

	void buildVerticesSmooth()
	{
		//const float S_STEP = 1 / 11.0f;            // horizontal texture step
		//const float T_STEP = 1 / 3.0f;		     // vertical texture step
		const float S_STEP = 186 / 2048.0f;			 // horizontal texture step
		const float T_STEP = 322 / 1024.0f;			 // vertical texture step

		// compute 12 vertices of icosahedron
		// NOTE: v0 (top), v11(bottom), v1, v6(first vert on each row) cannot be
		// shared for smooth shading (they have different texcoords)
		core::vector<float> tmpVertices = computeIcosahedronVertices();

		// clear memory of prev arrays
		core::vector<float>().swap(vertices);
		core::vector<float>().swap(normals);
		core::vector<float>().swap(texCoords);
		core::vector<uint32_t>().swap(indices);
		core::vector<uint32_t>().swap(lineIndices);
		std::map<std::pair<float, float>, uint32_t>().swap(sharedIndices);

		float v[3];                             // vertex
		float n[3];                             // normal
		float scale;                            // scale factor for normalization

		// smooth icosahedron has 14 non-shared (0 to 13) and
		// 8 shared vertices (14 to 21) (total 22 vertices)
		//  00  01  02  03  04          
		//  /\  /\  /\  /\  /\          
		// /  \/  \/  \/  \/  \         
		//10--14--15--16--17--11        
		// \  /\  /\  /\  /\  /\        
		//  \/  \/  \/  \/  \/  \       
		//  12--18--19--20--21--13      
		//   \  /\  /\  /\  /\  /       
		//    \/  \/  \/  \/  \/        
		//    05  06  07  08  09        
		// add 14 non-shared vertices first (index from 0 to 13)

		addVertex(tmpVertices[0], tmpVertices[1], tmpVertices[2]);      // v0 (top)
		addNormal(0, 0, 1);
		addTexCoord(S_STEP, 0);

		addVertex(tmpVertices[0], tmpVertices[1], tmpVertices[2]);      // v1
		addNormal(0, 0, 1);
		addTexCoord(S_STEP * 3, 0);

		addVertex(tmpVertices[0], tmpVertices[1], tmpVertices[2]);      // v2
		addNormal(0, 0, 1);
		addTexCoord(S_STEP * 5, 0);

		addVertex(tmpVertices[0], tmpVertices[1], tmpVertices[2]);      // v3
		addNormal(0, 0, 1);
		addTexCoord(S_STEP * 7, 0);

		addVertex(tmpVertices[0], tmpVertices[1], tmpVertices[2]);      // v4
		addNormal(0, 0, 1);
		addTexCoord(S_STEP * 9, 0);

		addVertex(tmpVertices[33], tmpVertices[34], tmpVertices[35]);   // v5 (bottom)
		addNormal(0, 0, -1);
		addTexCoord(S_STEP * 2, T_STEP * 3);

		addVertex(tmpVertices[33], tmpVertices[34], tmpVertices[35]);   // v6
		addNormal(0, 0, -1);
		addTexCoord(S_STEP * 4, T_STEP * 3);

		addVertex(tmpVertices[33], tmpVertices[34], tmpVertices[35]);   // v7
		addNormal(0, 0, -1);
		addTexCoord(S_STEP * 6, T_STEP * 3);

		addVertex(tmpVertices[33], tmpVertices[34], tmpVertices[35]);   // v8
		addNormal(0, 0, -1);
		addTexCoord(S_STEP * 8, T_STEP * 3);

		addVertex(tmpVertices[33], tmpVertices[34], tmpVertices[35]);   // v9
		addNormal(0, 0, -1);
		addTexCoord(S_STEP * 10, T_STEP * 3);

		v[0] = tmpVertices[3];  v[1] = tmpVertices[4];  v[2] = tmpVertices[5];  // v10 (left)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(0, T_STEP);

		addVertex(v[0], v[1], v[2]);                                            // v11 (right)
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 10, T_STEP);

		v[0] = tmpVertices[18]; v[1] = tmpVertices[19]; v[2] = tmpVertices[20]; // v12 (left)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP, T_STEP * 2);

		addVertex(v[0], v[1], v[2]);                                            // v13 (right)
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 11, T_STEP * 2);

		// add 8 shared vertices to array (index from 14 to 21)
		v[0] = tmpVertices[6];  v[1] = tmpVertices[7];  v[2] = tmpVertices[8];  // v14 (shared)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 2, T_STEP);
		sharedIndices[std::make_pair(S_STEP * 2, T_STEP)] = texCoords.size() / 2 - 1;

		v[0] = tmpVertices[9];  v[1] = tmpVertices[10]; v[2] = tmpVertices[11]; // v15 (shared)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 4, T_STEP);
		sharedIndices[std::make_pair(S_STEP * 4, T_STEP)] = texCoords.size() / 2 - 1;

		v[0] = tmpVertices[12]; v[1] = tmpVertices[13]; v[2] = tmpVertices[14]; // v16 (shared)
		scale = Icosphere::computeScaleForLength(v, 1);
		n[0] = v[0] * scale;    n[1] = v[1] * scale;    n[2] = v[2] * scale;
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 6, T_STEP);
		sharedIndices[std::make_pair(S_STEP * 6, T_STEP)] = texCoords.size() / 2 - 1;

		v[0] = tmpVertices[15]; v[1] = tmpVertices[16]; v[2] = tmpVertices[17]; // v17 (shared)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 8, T_STEP);
		sharedIndices[std::make_pair(S_STEP * 8, T_STEP)] = texCoords.size() / 2 - 1;

		v[0] = tmpVertices[21]; v[1] = tmpVertices[22]; v[2] = tmpVertices[23]; // v18 (shared)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 3, T_STEP * 2);
		sharedIndices[std::make_pair(S_STEP * 3, T_STEP * 2)] = texCoords.size() / 2 - 1;

		v[0] = tmpVertices[24]; v[1] = tmpVertices[25]; v[2] = tmpVertices[26]; // v19 (shared)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 5, T_STEP * 2);
		sharedIndices[std::make_pair(S_STEP * 5, T_STEP * 2)] = texCoords.size() / 2 - 1;

		v[0] = tmpVertices[27]; v[1] = tmpVertices[28]; v[2] = tmpVertices[29]; // v20 (shared)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 7, T_STEP * 2);
		sharedIndices[std::make_pair(S_STEP * 7, T_STEP * 2)] = texCoords.size() / 2 - 1;

		v[0] = tmpVertices[30]; v[1] = tmpVertices[31]; v[2] = tmpVertices[32]; // v21 (shared)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 9, T_STEP * 2);
		sharedIndices[std::make_pair(S_STEP * 9, T_STEP * 2)] = texCoords.size() / 2 - 1;

		// build index list for icosahedron (20 triangles)
		addIndices(0, 10, 14);      // 1st row (5 tris)
		addIndices(1, 14, 15);
		addIndices(2, 15, 16);
		addIndices(3, 16, 17);
		addIndices(4, 17, 11);
		addIndices(10, 12, 14);      // 2nd row (10 tris)
		addIndices(12, 18, 14);
		addIndices(14, 18, 15);
		addIndices(18, 19, 15);
		addIndices(15, 19, 16);
		addIndices(19, 20, 16);
		addIndices(16, 20, 17);
		addIndices(20, 21, 17);
		addIndices(17, 21, 11);
		addIndices(21, 13, 11);
		addIndices(5, 18, 12);      // 3rd row (5 tris)
		addIndices(6, 19, 18);
		addIndices(7, 20, 19);
		addIndices(8, 21, 20);
		addIndices(9, 13, 21);

		// add edge lines of icosahedron
		lineIndices.push_back(0);   lineIndices.push_back(10);       // 00 - 10
		lineIndices.push_back(1);   lineIndices.push_back(14);       // 01 - 14
		lineIndices.push_back(2);   lineIndices.push_back(15);       // 02 - 15
		lineIndices.push_back(3);   lineIndices.push_back(16);       // 03 - 16
		lineIndices.push_back(4);   lineIndices.push_back(17);       // 04 - 17
		lineIndices.push_back(10);  lineIndices.push_back(14);       // 10 - 14
		lineIndices.push_back(14);  lineIndices.push_back(15);       // 14 - 15
		lineIndices.push_back(15);  lineIndices.push_back(16);       // 15 - 16
		lineIndices.push_back(16);  lineIndices.push_back(17);       // 10 - 14
		lineIndices.push_back(17);  lineIndices.push_back(11);       // 17 - 11
		lineIndices.push_back(10);  lineIndices.push_back(12);       // 10 - 12
		lineIndices.push_back(12);  lineIndices.push_back(14);       // 12 - 14
		lineIndices.push_back(14);  lineIndices.push_back(18);       // 14 - 18
		lineIndices.push_back(18);  lineIndices.push_back(15);       // 18 - 15
		lineIndices.push_back(15);  lineIndices.push_back(19);       // 15 - 19
		lineIndices.push_back(19);  lineIndices.push_back(16);       // 19 - 16
		lineIndices.push_back(16);  lineIndices.push_back(20);       // 16 - 20
		lineIndices.push_back(20);  lineIndices.push_back(17);       // 20 - 17
		lineIndices.push_back(17);  lineIndices.push_back(21);       // 17 - 21
		lineIndices.push_back(21);  lineIndices.push_back(11);       // 21 - 11
		lineIndices.push_back(12);  lineIndices.push_back(18);       // 12 - 18
		lineIndices.push_back(18);  lineIndices.push_back(19);       // 18 - 19
		lineIndices.push_back(19);  lineIndices.push_back(20);       // 19 - 20
		lineIndices.push_back(20);  lineIndices.push_back(21);       // 20 - 21
		lineIndices.push_back(21);  lineIndices.push_back(13);       // 21 - 13
		lineIndices.push_back(5);   lineIndices.push_back(12);       // 05 - 12
		lineIndices.push_back(6);   lineIndices.push_back(18);       // 06 - 18
		lineIndices.push_back(7);   lineIndices.push_back(19);       // 07 - 19
		lineIndices.push_back(8);   lineIndices.push_back(20);       // 08 - 20
		lineIndices.push_back(9);   lineIndices.push_back(21);       // 09 - 21

		// subdivide icosahedron
		subdivideVerticesSmooth();

	}
	/*
		divide a trinage into 4 sub triangles and repeat N times
		If subdivision=0, do nothing.
	*/

	void subdivideVerticesFlat()
	{
		core::vector<float> tmpVertices;
		core::vector<float> tmpTexCoords;
		core::vector<unsigned int> tmpIndices;
		int32_t indexCount;
		const float* v1, * v2, * v3;			// ptr to original vertices of a triangle
		const float* t1, * t2, * t3;			// ptr to original texcoords of a triangle
		float newV1[3], newV2[3], newV3[3];		// new vertex positions
		float newT1[2], newT2[2], newT3[2];		// new texture coords
		float normal[3];						// new face normal
		uint32_t index = 0;					// new index value
		int32_t i, j;

		// iteration
		for (i = 1; i <= subdivision; ++i)
		{
			// copy prev arrays
			tmpVertices = vertices;
			tmpTexCoords = texCoords;
			tmpIndices = indices;

			// clear prev arrays
			vertices.clear();
			normals.clear();
			texCoords.clear();
			indices.clear();
			lineIndices.clear();

			index = 0;
			indexCount = (int)tmpIndices.size();
			for (j = 0; j < indexCount; j += 3)
			{
				// get 3 vertice and texcoords of a triangle
				v1 = &tmpVertices[tmpIndices[j] * 3];
				v2 = &tmpVertices[tmpIndices[j + 1] * 3];
				v3 = &tmpVertices[tmpIndices[j + 2] * 3];
				t1 = &tmpTexCoords[tmpIndices[j] * 2];
				t2 = &tmpTexCoords[tmpIndices[j + 1] * 2];
				t3 = &tmpTexCoords[tmpIndices[j + 2] * 2];

				// get 3 new vertices by spliting half on each edge
				computeHalfVertex(v1, v2, radius, newV1);
				computeHalfVertex(v2, v3, radius, newV2);
				computeHalfVertex(v1, v3, radius, newV3);
				computeHalfTexCoord(t1, t2, newT1);
				computeHalfTexCoord(t2, t3, newT2);
				computeHalfTexCoord(t1, t3, newT3);

				// add 4 new triangles
				addVertices(v1, newV1, newV3);
				addTexCoords(t1, newT1, newT3);
				computeFaceNormal(v1, newV1, newV3, normal);
				addNormals(normal, normal, normal);
				addIndices(index, index + 1, index + 2);

				addVertices(newV1, v2, newV2);
				addTexCoords(newT1, t2, newT2);
				computeFaceNormal(newV1, v2, newV2, normal);
				addNormals(normal, normal, normal);
				addIndices(index + 3, index + 4, index + 5);

				addVertices(newV1, newV2, newV3);
				addTexCoords(newT1, newT2, newT3);
				computeFaceNormal(newV1, newV2, newV3, normal);
				addNormals(normal, normal, normal);
				addIndices(index + 6, index + 7, index + 8);

				addVertices(newV3, newV2, v3);
				addTexCoords(newT3, newT2, t3);
				computeFaceNormal(newV3, newV2, v3, normal);
				addNormals(normal, normal, normal);
				addIndices(index + 9, index + 10, index + 11);

				// add new line indices per iteration
				addSubLineIndices(index, index + 1, index + 4, index + 5, index + 11, index + 9); //CCW

				// next index
				index += 12;
			}
		}
	}

	/*
		divide a trianlge (v1-v2-v3) into 4 sub triangles by adding middle vertices
		(newV1, newV2, newV3) and repeat N times
		If subdivision=0, do nothing.

				 v1           
				/ \           
		 newV1 *---* newV3    
				/ \ / \         
			v2---*---v3       
				newV2         
	*/

	void subdivideVerticesSmooth()
	{
		core::vector<uint32_t> tmpIndices;
		int32_t indexCount;
		uint32_t i1, i2, i3;            // indices from original triangle
		const float* v1, * v2, * v3;          // ptr to original vertices of a triangle
		const float* t1, * t2, * t3;          // ptr to original texcoords of a triangle
		float newV1[3], newV2[3], newV3[3]; // new subdivided vertex positions
		float newN1[3], newN2[3], newN3[3]; // new subdivided normals
		float newT1[2], newT2[2], newT3[2]; // new subdivided texture coords
		uint32_t newI1, newI2, newI3;   // new subdivided indices
		int32_t i, j;

		// iteration for subdivision
		for (i = 1; i <= subdivision; ++i)
		{
			// copy prev indices
			tmpIndices = indices;

			// clear prev arrays
			indices.clear();
			lineIndices.clear();

			indexCount = tmpIndices.size();
			for (j = 0; j < indexCount; j += 3)
			{
				// get 3 indices of each triangle
				i1 = tmpIndices[j];
				i2 = tmpIndices[j + 1];
				i3 = tmpIndices[j + 2];

				// get 3 vertex attribs from prev triangle
				v1 = &vertices[i1 * 3];
				v2 = &vertices[i2 * 3];
				v3 = &vertices[i3 * 3];
				t1 = &texCoords[i1 * 2];
				t2 = &texCoords[i2 * 2];
				t3 = &texCoords[i3 * 2];

				// get 3 new vertex attribs by spliting half on each edge
				computeHalfVertex(v1, v2, radius, newV1);
				computeHalfVertex(v2, v3, radius, newV2);
				computeHalfVertex(v1, v3, radius, newV3);
				computeHalfTexCoord(t1, t2, newT1);
				computeHalfTexCoord(t2, t3, newT2);
				computeHalfTexCoord(t1, t3, newT3);
				computeVertexNormal(newV1, newN1);
				computeVertexNormal(newV2, newN2);
				computeVertexNormal(newV3, newN3);

				// add new vertices/normals/texcoords to arrays
				// It will check if it is shared/non-shared and return index
				newI1 = addSubVertexAttribs(newV1, newN1, newT1);
				newI2 = addSubVertexAttribs(newV2, newN2, newT2);
				newI3 = addSubVertexAttribs(newV3, newN3, newT3);

				// add 4 new triangle indices
				addIndices(i1, newI1, newI3);
				addIndices(newI1, i2, newI2);
				addIndices(newI1, newI2, newI3);
				addIndices(newI3, newI2, i3);

				// add new line indices
				addSubLineIndices(i1, newI1, i2, newI2, i3, newI3); //CCW
			}
		}
	}

	/*
		generate interleaved vertices: V/N/T
		stride must be 32 bytes
	*/

	void addVertex(float x, float y, float z)
	{
		vertices.push_back(x);
		vertices.push_back(y);
		vertices.push_back(z);
	}

	void addVertices(const float v1[3], const float v2[3], const float v3[3])
	{
		vertices.push_back(v1[0]);  // x
		vertices.push_back(v1[1]);  // y
		vertices.push_back(v1[2]);  // z
		vertices.push_back(v2[0]);
		vertices.push_back(v2[1]);
		vertices.push_back(v2[2]);
		vertices.push_back(v3[0]);
		vertices.push_back(v3[1]);
		vertices.push_back(v3[2]);
	}

	void addNormal(float nx, float ny, float nz)
	{
		normals.push_back(nx);
		normals.push_back(ny);
		normals.push_back(nz);
	}

	void addNormals(const float n1[3], const float n2[3], const float n3[3])
	{
		normals.push_back(n1[0]);  // nx
		normals.push_back(n1[1]);  // ny
		normals.push_back(n1[2]);  // nz
		normals.push_back(n2[0]);
		normals.push_back(n2[1]);
		normals.push_back(n2[2]);
		normals.push_back(n3[0]);
		normals.push_back(n3[1]);
		normals.push_back(n3[2]);
	}

	void addTexCoord(float s, float t)
	{
		texCoords.push_back(s);
		texCoords.push_back(t);
	}

	void addTexCoords(const float t1[2], const float t2[2], const float t3[2])
	{
		texCoords.push_back(t1[0]); // s
		texCoords.push_back(t1[1]); // t
		texCoords.push_back(t2[0]);
		texCoords.push_back(t2[1]);
		texCoords.push_back(t3[0]);
		texCoords.push_back(t3[1]);
	}

	void addIndices(unsigned int i1, unsigned int i2, unsigned int i3)
	{
		indices.push_back(i1);
		indices.push_back(i2);
		indices.push_back(i3);
	}

	/*
		add 7 sub edge lines per triangle to array using 6 indices (CCW)           
			 i1                                                                     
			 /            : (i1, i2)                                                
			 i2---i6        : (i2, i6)												  
			 / \  /         : (i2, i3), (i2, i4), (i6, i4)							  
		 i3---i4---i5     : (i3, i4), (i4, i5)									  
	*/

	void addSubLineIndices(unsigned int i1, unsigned int i2, unsigned int i3, unsigned int i4, unsigned int i5, unsigned int i6)
	{
		lineIndices.push_back(i1);      // i1 - i2
		lineIndices.push_back(i2);
		lineIndices.push_back(i2);      // i2 - i6
		lineIndices.push_back(i6);
		lineIndices.push_back(i2);      // i2 - i3
		lineIndices.push_back(i3);
		lineIndices.push_back(i2);      // i2 - i4
		lineIndices.push_back(i4);
		lineIndices.push_back(i6);      // i6 - i4
		lineIndices.push_back(i4);
		lineIndices.push_back(i3);      // i3 - i4
		lineIndices.push_back(i4);
		lineIndices.push_back(i4);      // i4 - i5
		lineIndices.push_back(i5);
	}

	/*
		add a subdivided vertex attribs (vertex, normal, texCoord) to arrays, then
		return its index value
		If it is a shared vertex, remember its index, so it can be re-used
	*/

	unsigned int addSubVertexAttribs(const float v[3], const float n[3], const float t[2])
	{
		unsigned int index; // return value;

		// check if is shared vertex or not first
		if (Icosphere::isSharedTexCoord(t))
		{
			// find if it does already exist in sharedIndices map using (s,t) key
			// if not in the list, add the vertex attribs to arrays and return its index
			// if exists, return the current index
			std::pair<float, float> key = std::make_pair(t[0], t[1]);
			std::map<std::pair<float, float>, unsigned int>::iterator iter = sharedIndices.find(key);
			if (iter == sharedIndices.end())
			{
				addVertex(v[0], v[1], v[2]);
				addNormal(n[0], n[1], n[2]);
				addTexCoord(t[0], t[1]);
				index = texCoords.size() / 2 - 1;
				sharedIndices[key] = index;
			}
			else
			{
				index = iter->second;
			}
		}
		// not shared
		else
		{
			addVertex(v[0], v[1], v[2]);
			addNormal(n[0], n[1], n[2]);
			addTexCoord(t[0], t[1]);
			index = texCoords.size() / 2 - 1;
		}

		return index;
	}

	float radius;													// circumscribed radius
	uint32_t subdivision;
	bool smooth;
	core::vector<float> vertices;
	core::vector<float> normals;
	core::vector<float> texCoords;
	core::vector<uint32_t> indices;
	core::vector<uint32_t> lineIndices;
	std::map<std::pair<float, float>, uint32_t> sharedIndices;   // indices of shared vertices, key is tex coord (s,t)


};

core::smart_refctd_ptr<ICPUPolygonGeometry> CGeometryCreator::createIcoSphere(float radius, uint32_t subdivision, bool smooth) const
{

	Icosphere icosphere(radius, subdivision, smooth);

	auto retval = core::make_smart_refctd_ptr<ICPUPolygonGeometry>();
	retval->setIndexing(IPolygonGeometryBase::TriangleList());

	using namespace hlsl;

	// Create indices
	{
		auto indexView = createIndexView<Icosphere::index_t>(icosphere.getIndexCount(), icosphere.getVertexCount() - 1);
		memcpy(indexView.src.buffer->getPointer(), icosphere.getIndices(), icosphere.getIndexSize());
		retval->setIndexView(std::move(indexView));
	}

	{
		{
			shapes::AABB<4, float32_t> aabb;
			aabb.maxVx = float32_t4(radius, radius, radius, 0.f);
			aabb.minVx = -aabb.maxVx;
			auto positionView = createPositionView(icosphere.getVertexCount(), aabb);
			memcpy(positionView.src.buffer->getPointer(), icosphere.getPositions(), icosphere.getPositionSize());
			retval->setPositionView(std::move(positionView));
		}
		{
			using normal_t = float32_t3;
			constexpr auto AttrSize = sizeof(normal_t);
			auto buff = ICPUBuffer::create({icosphere.getNormalSize(), IBuffer::EUF_NONE});
			const auto normals = reinterpret_cast<normal_t*>(buff->getPointer());
			memcpy(normals, icosphere.getNormals(), icosphere.getNormalSize());
			shapes::AABB<4,float32_t> aabb;
			aabb.maxVx = float32_t4(1, 1, 1, 0.f);
			aabb.minVx = -aabb.maxVx;
			retval->setNormalView({
				.composed = {
					.encodedDataRange = {.f32 = aabb},
					.stride = AttrSize,
					.format = EF_R32G32B32_SFLOAT,
					.rangeFormat = IGeometryBase::EAABBFormat::F32
				},
				.src = {.offset = 0,.size = buff->getSize(),.buffer = std::move(buff)},
			});
		}
		{
			using uv_element_t = uint16_t;
			hlsl::vector<uv_element_t, 2>* uvs;
			auto uvView = createUvView<uv_element_t>(icosphere.getVertexCount());
			uvs = reinterpret_cast<decltype(uvs)>(uvView.src.buffer->getPointer());

			for (auto uv_i = 0u; uv_i < icosphere.getVertexCount(); uv_i++)
			{
				const auto texCoords = icosphere.getTexCoords();
				encodeUv(uvs + uv_i, float32_t2(texCoords[2 * uv_i], texCoords[(2 * uv_i) + 1]));
			}

			retval->getAuxAttributeViews()->push_back(std::move(uvView));
		}
	}

	CPolygonGeometryManipulator::recomputeContentHashes(retval.get());
	return retval;
}

} // end namespace nbl::asset


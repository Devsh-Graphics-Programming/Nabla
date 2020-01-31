// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// Code for this scene node has been contributed by Anders la Cour-Harbo (alc)

#include "CSkyDomeSceneNode.h"

#include "ICameraSceneNode.h"
#include "ISceneManager.h"

namespace irr
{
namespace scene
{


/* horiRes and vertRes:
	Controls the number of faces along the horizontal axis (30 is a good value)
	and the number of faces along the vertical axis (8 is a good value).

	texturePercentage:
	Only the top texturePercentage of the image is used, e.g. 0.8 uses the top 80% of the image,
	1.0 uses the entire image. This is useful as some landscape images have a small banner
	at the bottom that you don't want.

	spherePercentage:
	This controls how far around the sphere the sky dome goes. For value 1.0 you get exactly the upper
	hemisphere, for 1.1 you get slightly more, and for 2.0 you get a full sphere. It is sometimes useful
	to use a value slightly bigger than 1 to avoid a gap between some ground place and the sky. This
	parameters stretches the image to fit the chosen "sphere-size". */

CSkyDomeSceneNode::CSkyDomeSceneNode(core::smart_refctd_ptr<video::IVirtualTexture>&& texture, uint32_t horiRes, uint32_t vertRes,
		float texturePercentage, float spherePercentage, float radius,
		IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id)
	: ISceneNode(parent, mgr, id), Buffer(nullptr),
	  HorizontalResolution(horiRes), VerticalResolution(vertRes),
	  TexturePercentage(texturePercentage),
	  SpherePercentage(spherePercentage), Radius(radius)
{
	#ifdef _IRR_DEBUG
	setDebugName("CSkyDomeSceneNode");
	#endif

	setAutomaticCulling(scene::EAC_OFF);

	Buffer = new video::IGPUMeshBuffer();
	Buffer->getMaterial().ZBuffer = video::ECFN_NEVER;
	Buffer->getMaterial().BackfaceCulling = false;
	Buffer->getMaterial().ZWriteEnable = false;
	Buffer->getMaterial().setTexture(0, std::move(texture));
	BoundingBox.MaxEdge.set(0,0,0);
	BoundingBox.MinEdge.set(0,0,0);

	// regenerate the mesh
	generateMesh();
}

CSkyDomeSceneNode::CSkyDomeSceneNode(CSkyDomeSceneNode* other,
		IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id)
	: ISceneNode(parent, mgr, id), Buffer(0),
	  HorizontalResolution(other->HorizontalResolution), VerticalResolution(other->VerticalResolution),
	  TexturePercentage(other->TexturePercentage),
	  SpherePercentage(other->SpherePercentage), Radius(other->Radius)
{
	#ifdef _IRR_DEBUG
	setDebugName("CSkyDomeSceneNode");
	#endif

	setAutomaticCulling(scene::EAC_OFF);

	Buffer = other->Buffer;
	Buffer->grab();
}


CSkyDomeSceneNode::~CSkyDomeSceneNode()
{
	if (Buffer)
		Buffer->drop();
}


void CSkyDomeSceneNode::generateMesh()
{
	float azimuth;
	uint32_t k;


	const float azimuth_step = (core::PI<float>() * 2.f) / float(HorizontalResolution);
	if (SpherePercentage < 0.f)
		SpherePercentage = -SpherePercentage;
	if (SpherePercentage > 2.f)
		SpherePercentage = 2.f;
	const float elevation_step = SpherePercentage * core::HALF_PI<float>() / float(VerticalResolution);


    size_t numOfIndices = 3 * (2*VerticalResolution - 1) * HorizontalResolution;
	uint16_t* indices = (uint16_t*)_IRR_ALIGNED_MALLOC(numOfIndices*2,_IRR_SIMD_ALIGNMENT);

	size_t numberOfVertices = (HorizontalResolution + 1) * (VerticalResolution + 1);
	float* vertices = (float*)_IRR_ALIGNED_MALLOC(4*numberOfVertices*(3+2),_IRR_SIMD_ALIGNMENT);


	const float tcV = TexturePercentage / VerticalResolution;
	size_t vxIx = 0;
	for (k = 0, azimuth = 0; k <= HorizontalResolution; ++k)
	{
		float elevation = core::HALF_PI<float>();
		const float tcU = (float)k / (float)HorizontalResolution;
		const float sinA = sinf(azimuth);
		const float cosA = cosf(azimuth);
		for (uint32_t j = 0; j <= VerticalResolution; ++j)
		{
			const float cosEr = Radius * cosf(elevation);
			vertices[vxIx*(3+2)+0] = cosEr*sinA;
			vertices[vxIx*(3+2)+1] = Radius*sinf(elevation);
			vertices[vxIx*(3+2)+2] = cosEr*cosA;
			vertices[vxIx*(3+2)+3] = tcU;
			vertices[vxIx*(3+2)+4] = j*tcV;

			vxIx++;
			elevation -= elevation_step;
		}
		azimuth += azimuth_step;
	}

    size_t ixIx = 0;
	for (k = 0; k < HorizontalResolution; ++k)
	{
		indices[ixIx++] = VerticalResolution + 2 + (VerticalResolution + 1)*k;
		indices[ixIx++] = 1 + (VerticalResolution + 1)*k;
		indices[ixIx++] = 0 + (VerticalResolution + 1)*k;

		for (uint32_t j = 1; j < VerticalResolution; ++j)
		{
            indices[ixIx++] = VerticalResolution + 2 + (VerticalResolution + 1)*k + j;
			indices[ixIx++] = 1 + (VerticalResolution + 1)*k + j;
			indices[ixIx++] = 0 + (VerticalResolution + 1)*k + j;

			indices[ixIx++] = VerticalResolution + 1 + (VerticalResolution + 1)*k + j;
			indices[ixIx++] = VerticalResolution + 2 + (VerticalResolution + 1)*k + j;
			indices[ixIx++] = 0 + (VerticalResolution + 1)*k + j;
		}
	}

    auto vao = SceneManager->getVideoDriver()->createGPUMeshDataFormatDesc();

	video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
	reqs.vulkanReqs.size = numOfIndices*sizeof(uint16_t);
	reqs.vulkanReqs.alignment = 2;
	reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
	reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
	reqs.mappingCapability = video::IDriverMemoryAllocation::EMCAF_NO_MAPPING_ACCESS;
	reqs.prefersDedicatedAllocation = true;
	reqs.requiresDedicatedAllocation = true;
	{
		auto indexBuf = core::smart_refctd_ptr<video::IGPUBuffer>(SceneManager->getVideoDriver()->createGPUBufferOnDedMem(reqs, true), core::dont_grab);
		indexBuf->updateSubRange(video::IDriverMemoryAllocation::MemoryRange(0, reqs.vulkanReqs.size), indices);
		_IRR_ALIGNED_FREE(indices);
		vao->setIndexBuffer(std::move(indexBuf));
		Buffer->setIndexType(asset::EIT_16BIT);
		Buffer->setIndexCount(numOfIndices);
	}

	reqs.vulkanReqs.size = 4*numberOfVertices*(3+2);
	reqs.vulkanReqs.alignment = 4;
	{
		auto vAttr = core::smart_refctd_ptr<video::IGPUBuffer>(SceneManager->getVideoDriver()->createGPUBufferOnDedMem(reqs, true),core::dont_grab);
		vAttr->updateSubRange(video::IDriverMemoryAllocation::MemoryRange(0, reqs.vulkanReqs.size), vertices);
		_IRR_ALIGNED_FREE(vertices);
		vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(vAttr), asset::EVAI_ATTR0, asset::EF_R32G32B32_SFLOAT, 4 * (3 + 2), 0);
		vao->setVertexAttrBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(vAttr), asset::EVAI_ATTR2, asset::EF_R32G32_SFLOAT, 4 * (3 + 2), 4 * 3);
	}

	Buffer->setMeshDataAndFormat(std::move(vao));
}


//! renders the node.
void CSkyDomeSceneNode::render()
{
	video::IVideoDriver* driver = SceneManager->getVideoDriver();
	scene::ICameraSceneNode* camera = SceneManager->getActiveCamera();

	if (!camera || !driver)
		return;

	if ( !camera->getProjectionMatrix().isOrthogonal() && canProceedPastFence() ) // check this actually works!
	{
		core::matrix3x4SIMD mat;
		mat.set(AbsoluteTransformation);
		mat.setTranslation(core::vectorSIMDf().set(camera->getAbsolutePosition()));

		driver->setTransform(video::E4X3TS_WORLD, mat);

		driver->setMaterial(Buffer->getMaterial());
		driver->drawMeshBuffer(Buffer);
	}

	// for debug purposes only:
	if ( DebugDataVisible )
	{
		video::SGPUMaterial m;

		// show mesh
		if ( DebugDataVisible & scene::EDS_MESH_WIRE_OVERLAY )
		{
			m.Wireframe = true;
			driver->setMaterial(m);

			driver->drawMeshBuffer(Buffer);
		}
	}
}


//! returns the axis aligned bounding box of this node
const core::aabbox3d<float>& CSkyDomeSceneNode::getBoundingBox()
{
	return BoundingBox;
}


void CSkyDomeSceneNode::OnRegisterSceneNode()
{
	if (IsVisible)
	{
		SceneManager->registerNodeForRendering(this, ESNRP_SKY_BOX );
	}

	ISceneNode::OnRegisterSceneNode();
}


} // namespace scene
} // namespace irr

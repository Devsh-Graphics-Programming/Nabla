// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CNullDriver.h"
#include "os.h"
#include "IAnimatedMeshSceneNode.h"
#include "irr/asset/CMeshManipulator.h"
#include "CMeshSceneNodeInstanced.h"

#include <new>
#include "IrrlichtDevice.h"

namespace irr
{
namespace video
{

FW_AtomicCounter CNullDriver::ReallocationCounter(0);
int32_t CNullDriver::incrementAndFetchReallocCounter()
{
// omg this has to be rewritten
#if defined(FW_MUTEX_H_CXX11_IMPL)
	return ReallocationCounter += 1;
#elif _MSC_VER && !__INTEL_COMPILER
    return InterlockedIncrement(&ReallocationCounter);
#elif defined(__GNUC__)
    return __sync_add_and_fetch(&ReallocationCounter,int32_t(1));
#endif // _MSC_VER
}

//! constructor
CNullDriver::CNullDriver(IrrlichtDevice* dev, io::IFileSystem* io, const core::dimension2d<uint32_t>& screenSize)
: IVideoDriver(dev), FileSystem(io), ViewPort(0,0,0,0), ScreenSize(screenSize), 
	PrimitivesDrawn(0), TextureCreationFlags(0),
	OverrideMaterial2DEnabled(false),
	matrixModifiedBits(0)
{
	#ifdef _IRR_DEBUG
	setDebugName("CNullDriver");
	#endif

    for (size_t i = 0; i < EQOT_COUNT; i++)
        currentQuery[i] = nullptr;

	setTextureCreationFlag(ETCF_ALWAYS_32_BIT, true);
	setTextureCreationFlag(ETCF_CREATE_MIP_MAPS, true);

	ViewPort = core::rect<int32_t>(core::position2d<int32_t>(0,0), core::dimension2di(screenSize));


	if (FileSystem)
		FileSystem->grab();

    MaxTextureSizes[ITexture::ETT_1D][0] = 0x80u;
    MaxTextureSizes[ITexture::ETT_1D][1] = 0x1u;
    MaxTextureSizes[ITexture::ETT_1D][2] = 0x1u;

    MaxTextureSizes[ITexture::ETT_2D][0] = 0x80u;
    MaxTextureSizes[ITexture::ETT_2D][1] = 0x80u;
    MaxTextureSizes[ITexture::ETT_2D][2] = 0x1u;

    MaxTextureSizes[ITexture::ETT_3D][0] = 0x80u;
    MaxTextureSizes[ITexture::ETT_3D][1] = 0x80u;
    MaxTextureSizes[ITexture::ETT_3D][2] = 0x80u;

    MaxTextureSizes[ITexture::ETT_1D_ARRAY][0] = 0x80u;
    MaxTextureSizes[ITexture::ETT_1D_ARRAY][1] = 0x1u;
    MaxTextureSizes[ITexture::ETT_1D_ARRAY][2] = 0x800u;

    MaxTextureSizes[ITexture::ETT_2D_ARRAY][0] = 0x80u;
    MaxTextureSizes[ITexture::ETT_2D_ARRAY][1] = 0x80u;
    MaxTextureSizes[ITexture::ETT_2D_ARRAY][2] = 0x800u;

    MaxTextureSizes[ITexture::ETT_CUBE_MAP][0] = 0x80u;
    MaxTextureSizes[ITexture::ETT_CUBE_MAP][1] = 0x80u;
    MaxTextureSizes[ITexture::ETT_CUBE_MAP][2] = 0x6u;

    MaxTextureSizes[ITexture::ETT_CUBE_MAP_ARRAY][0] = 0x80u;
    MaxTextureSizes[ITexture::ETT_CUBE_MAP_ARRAY][1] = 0x80u;
    MaxTextureSizes[ITexture::ETT_CUBE_MAP_ARRAY][2] = 0x800u*6;


	// set ExposedData to 0
	memset(&ExposedData, 0, sizeof(ExposedData));
}


//! destructor
CNullDriver::~CNullDriver()
{
	if (FileSystem)
		FileSystem->drop();
}

//! applications must call this method before performing any rendering. returns false if failed.
bool CNullDriver::beginScene(bool backBuffer, bool zBuffer, SColor color,
		const SExposedVideoData& videoData, core::rect<int32_t>* sourceRect)
{
	scene::CMeshSceneNodeInstanced::recullOrder = 0;

	PrimitivesDrawn = 0;
	return true;
}


//! applications must call this method after performing any rendering. returns false if failed.
bool CNullDriver::endScene()
{
	FPSCounter.registerFrame(std::chrono::high_resolution_clock::now(), PrimitivesDrawn);

	return true;
}


//! sets transformation
void CNullDriver::setTransform(const E_4X3_TRANSFORMATION_STATE& state, const core::matrix4x3& mat)
{
    if (state>E4X3TS_WORLD)
        return;


	const uint32_t commonBits = (0x1u<<E4X3TS_WORLD_VIEW)|(0x1u<<E4X3TS_WORLD_VIEW_INVERSE)|(0x1u<<E4X3TS_NORMAL_MATRIX)|(0x1u<<(E4X3TS_COUNT+EPTS_PROJ_VIEW_WORLD))|(0x1u<<(E4X3TS_COUNT+EPTS_PROJ_VIEW_WORLD_INVERSE));
    uint32_t modifiedBit;
    switch (state)
    {
        case E4X3TS_VIEW:
            modifiedBit = (0x1u<<E4X3TS_VIEW)|(0x1u<<E4X3TS_VIEW_INVERSE)|(0x1u<<(E4X3TS_COUNT+EPTS_PROJ_VIEW))|(0x1u<<(E4X3TS_COUNT+EPTS_PROJ_VIEW_INVERSE))|commonBits;
            break;
        case E4X3TS_WORLD:
            modifiedBit = (0x1u<<E4X3TS_WORLD)|(0x1u<<E4X3TS_WORLD_INVERSE)|commonBits;
            break;
    }

    //if all bits marked as modified and matrices dont change
    if ((matrixModifiedBits&modifiedBit)==modifiedBit)
        TransformationMatrices[state] = mat;
    else
    {
        if (TransformationMatrices[state]==mat)
            return;
        matrixModifiedBits |= modifiedBit;
        TransformationMatrices[state] = mat;
    }
}

//! sets transformation
void CNullDriver::setTransform(const E_PROJECTION_TRANSFORMATION_STATE& state, const core::matrix4SIMD& mat)
{
    if (state>EPTS_PROJ)
        return;


	const uint32_t modifiedBit = ((0x1u<<EPTS_PROJ)|(0x1u<<EPTS_PROJ_VIEW)|(0x1u<<EPTS_PROJ_VIEW_WORLD)|(0x1u<<EPTS_PROJ_INVERSE)|(0x1u<<EPTS_PROJ_VIEW_INVERSE)|(0x1u<<EPTS_PROJ_VIEW_WORLD_INVERSE))<<E4X3TS_COUNT;

    //if all bits marked as modified and matrices dont change
    if ((matrixModifiedBits&modifiedBit)==modifiedBit)
        ProjectionMatrices[state] = mat;
    else
    {
        if (ProjectionMatrices[state]==mat)
            return;
        matrixModifiedBits |= modifiedBit;
        ProjectionMatrices[state] = mat;
    }
}


//! Returns the transformation set by setTransform
const core::matrix4x3& CNullDriver::getTransform(const E_4X3_TRANSFORMATION_STATE& state)
{
    const uint32_t stateBit = 0x1u<<state;

	if (matrixModifiedBits&stateBit)
    {
        switch (state)
        {
            case E4X3TS_WORLD:
            case E4X3TS_VIEW:
                break;
            case E4X3TS_WORLD_VIEW:
                TransformationMatrices[E4X3TS_WORLD_VIEW] = concatenateBFollowedByA(TransformationMatrices[E4X3TS_VIEW],TransformationMatrices[E4X3TS_WORLD]);
                break;
            case E4X3TS_VIEW_INVERSE:
                TransformationMatrices[E4X3TS_VIEW].getInverse(TransformationMatrices[E4X3TS_VIEW_INVERSE]);
                break;
            case E4X3TS_WORLD_INVERSE:
                TransformationMatrices[E4X3TS_WORLD].getInverse(TransformationMatrices[E4X3TS_WORLD_INVERSE]);
                break;
            case E4X3TS_WORLD_VIEW_INVERSE:
                if (matrixModifiedBits&(0x1u<<E4X3TS_WORLD_VIEW))
                {
                    TransformationMatrices[E4X3TS_WORLD_VIEW] = concatenateBFollowedByA(TransformationMatrices[E4X3TS_VIEW],TransformationMatrices[E4X3TS_WORLD]);
                    matrixModifiedBits &= ~(0x1u<<E4X3TS_WORLD_VIEW);
                }

                TransformationMatrices[E4X3TS_WORLD_VIEW].getInverse(TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE]);
                break;
            case E4X3TS_NORMAL_MATRIX:
                if (matrixModifiedBits&(0x1u<<E4X3TS_WORLD_VIEW_INVERSE))
                {
                    if (matrixModifiedBits&(0x1u<<E4X3TS_WORLD_VIEW))
                    {
                        TransformationMatrices[E4X3TS_WORLD_VIEW] = concatenateBFollowedByA(TransformationMatrices[E4X3TS_VIEW],TransformationMatrices[E4X3TS_WORLD]);
                        matrixModifiedBits &= ~(0x1u<<E4X3TS_WORLD_VIEW);
                    }

                    TransformationMatrices[E4X3TS_WORLD_VIEW].getInverse(TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE]);
                    matrixModifiedBits &= ~(0x1u<<E4X3TS_WORLD_VIEW_INVERSE);
                }

                TransformationMatrices[E4X3TS_NORMAL_MATRIX](0,0) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](0,0);
                TransformationMatrices[E4X3TS_NORMAL_MATRIX](0,1) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](1,0);
                TransformationMatrices[E4X3TS_NORMAL_MATRIX](0,2) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](2,0);
                TransformationMatrices[E4X3TS_NORMAL_MATRIX](1,0) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](0,1);
                TransformationMatrices[E4X3TS_NORMAL_MATRIX](1,1) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](1,1);
                TransformationMatrices[E4X3TS_NORMAL_MATRIX](1,2) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](2,1);
                TransformationMatrices[E4X3TS_NORMAL_MATRIX](2,0) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](0,2);
                TransformationMatrices[E4X3TS_NORMAL_MATRIX](2,1) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](1,2);
                TransformationMatrices[E4X3TS_NORMAL_MATRIX](2,2) = TransformationMatrices[E4X3TS_WORLD_VIEW_INVERSE](2,2);
                break;
        }

        matrixModifiedBits &= ~stateBit;
    }

    return TransformationMatrices[state];
}

//! Returns the transformation set by setTransform
const core::matrix4SIMD& CNullDriver::getTransform(const E_PROJECTION_TRANSFORMATION_STATE& state)
{
    const uint32_t stateBit = 0x1u<<(state+E4X3TS_COUNT);

	if (matrixModifiedBits&stateBit)
    {
        switch (state)
        {
            case EPTS_PROJ:
                break;
            case EPTS_PROJ_VIEW:
                ProjectionMatrices[EPTS_PROJ_VIEW] = core::concatenateBFollowedByA(ProjectionMatrices[EPTS_PROJ],TransformationMatrices[E4X3TS_VIEW]);
                break;
            case EPTS_PROJ_VIEW_WORLD:
                if (matrixModifiedBits&(0x1u<<E4X3TS_WORLD_VIEW))
                {
                    TransformationMatrices[E4X3TS_WORLD_VIEW] = concatenateBFollowedByA(TransformationMatrices[E4X3TS_VIEW],TransformationMatrices[E4X3TS_WORLD]);
                    ///TransformationMatrices[E4X3TS_WORLD_VIEW] = concatenatePreciselyBFollowedByA(TransformationMatrices[E4X3TS_VIEW],TransformationMatrices[E4X3TS_WORLD]);
                    matrixModifiedBits &= ~(0x1u<<E4X3TS_WORLD_VIEW);
                }
                ProjectionMatrices[EPTS_PROJ_VIEW_WORLD] = concatenateBFollowedByA(ProjectionMatrices[EPTS_PROJ],TransformationMatrices[E4X3TS_WORLD_VIEW]);
                break;
            case EPTS_PROJ_INVERSE:
                ProjectionMatrices[EPTS_PROJ].getInverseTransform(ProjectionMatrices[EPTS_PROJ]);
                break;
            case EPTS_PROJ_VIEW_INVERSE:
                if (matrixModifiedBits&(0x1u<<(EPTS_PROJ_VIEW+E4X3TS_COUNT)))
                {
                    ProjectionMatrices[EPTS_PROJ_VIEW] = concatenateBFollowedByA(ProjectionMatrices[EPTS_PROJ],TransformationMatrices[E4X3TS_VIEW]);
                    matrixModifiedBits &= ~(0x1u<<(EPTS_PROJ_VIEW+E4X3TS_COUNT));
                }
                ProjectionMatrices[EPTS_PROJ_VIEW].getInverseTransform(ProjectionMatrices[EPTS_PROJ_VIEW_INVERSE]);
                break;
            case EPTS_PROJ_VIEW_WORLD_INVERSE:
                if (matrixModifiedBits&(0x1u<<(EPTS_PROJ_VIEW_WORLD+E4X3TS_COUNT)))
                {
                    if (matrixModifiedBits&(0x1u<<E4X3TS_WORLD_VIEW))
                    {
                        TransformationMatrices[E4X3TS_WORLD_VIEW] = concatenateBFollowedByA(TransformationMatrices[E4X3TS_VIEW],TransformationMatrices[E4X3TS_WORLD]);
                        ///TransformationMatrices[E4X3TS_WORLD_VIEW] = concatenatePreciselyBFollowedByA(TransformationMatrices[E4X3TS_VIEW],TransformationMatrices[E4X3TS_WORLD]);
                        matrixModifiedBits &= ~(0x1u<<E4X3TS_WORLD_VIEW);
                    }
                    ProjectionMatrices[EPTS_PROJ_VIEW_WORLD] = concatenateBFollowedByA(ProjectionMatrices[EPTS_PROJ],TransformationMatrices[E4X3TS_WORLD_VIEW]);
                    matrixModifiedBits &= ~(0x1u<<(EPTS_PROJ_VIEW_WORLD+E4X3TS_COUNT));
                }
                ProjectionMatrices[EPTS_PROJ_VIEW_WORLD].getInverseTransform(ProjectionMatrices[EPTS_PROJ_VIEW_WORLD_INVERSE]);
                break;
        }

        matrixModifiedBits &= ~stateBit;
    }
    return ProjectionMatrices[state];
}

void CNullDriver::removeFrameBuffer(IFrameBuffer* framebuf)
{
}

void CNullDriver::removeAllFrameBuffers()
{
}

core::smart_refctd_ptr<ITexture> CNullDriver::createGPUTexture(const ITexture::E_TEXTURE_TYPE& type, const uint32_t* size, uint32_t mipmapLevels, asset::E_FORMAT format)
{
    if (type != ITexture::ETT_2D)
        return nullptr;

    return core::make_smart_refctd_ptr<SDummyTexture>("");
}


//! returns a device dependent texture from parameters
//! THIS METHOD HAS TO BE OVERRIDDEN BY DERIVED DRIVERS WITH OWN TEXTURES
core::smart_refctd_ptr<ITexture> CNullDriver::createDeviceDependentTexture(const ITexture::E_TEXTURE_TYPE& type, const uint32_t* size, uint32_t mipmapLevels,
			const io::path& name, asset::E_FORMAT format)
{
    //better safe than sorry
    if (type!=ITexture::ETT_2D)
        return nullptr;

	return core::make_smart_refctd_ptr<SDummyTexture>(name);
}

void CNullDriver::bindDescriptorSets_generic(const IGPUPipelineLayout* _newLayout, uint32_t _first, uint32_t _count, const IGPUDescriptorSet** _descSets, const IGPUPipelineLayout** _destPplnLayouts)
{
    uint32_t compatibilityLimits[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT]{}; //actually more like "compatibility limit + 1" (i.e. 0 mean not comaptible at all)
    for (uint32_t i = 0u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
    {
        const uint32_t lim = _destPplnLayouts[i] ? //if no descriptor set bound at this index
            _destPplnLayouts[i]->isCompatibleForSet(IGPUPipelineLayout::DESCRIPTOR_SET_COUNT - 1u, _newLayout) : 0u;

        compatibilityLimits[i] = (lim == IGPUPipelineLayout::DESCRIPTOR_SET_COUNT) ? 0u : (lim + 1u);
    }

    /*
    https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#descriptorsets-compatibility
    When binding a descriptor set (see Descriptor Set Binding) to set number N, if the previously bound descriptor sets for sets zero through N-1 were all bound using compatible pipeline layouts, then performing this binding does not disturb any of the lower numbered sets.
    */
    for (uint32_t i = 0u; i < _first; i++)
        if (compatibilityLimits[i] <= i)
            _destPplnLayouts[i] = nullptr;

    /*
    If, additionally, the previous bound descriptor set for set N was bound using a pipeline layout compatible for set N, then the bindings in sets numbered greater than N are also not disturbed.
    */
    if (compatibilityLimits[_first] <= _first)
        for (uint32_t i = _first + _count; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
            _destPplnLayouts = nullptr;
}


//! sets a render target
bool CNullDriver::setRenderTarget(video::IFrameBuffer* texture, bool setNewViewport)
{
	return false;
}


//! sets a viewport
void CNullDriver::setViewPort(const core::rect<int32_t>& area)
{
}


//! gets the area of the current viewport
const core::rect<int32_t>& CNullDriver::getViewPort() const
{
	return ViewPort;
}


//! returns color format
asset::E_FORMAT CNullDriver::getColorFormat() const
{
	return asset::EF_B5G6R5_UNORM_PACK16;
}


//! returns screen size
const core::dimension2d<uint32_t>& CNullDriver::getScreenSize() const
{
	return ScreenSize;
}


//! returns the current render target size,
//! or the screen size if render targets are not implemented
const core::dimension2d<uint32_t>& CNullDriver::getCurrentRenderTargetSize() const
{
	return ScreenSize;
}


// returns current frames per second value
int32_t CNullDriver::getFPS() const
{
	return FPSCounter.getFPS();
}



//! returns amount of primitives (mostly triangles) were drawn in the last frame.
//! very useful method for statistics.
uint32_t CNullDriver::getPrimitiveCountDrawn( uint32_t param ) const
{
	return (0 == param) ? FPSCounter.getPrimitive() : (1 == param) ? FPSCounter.getPrimitiveAverage() : FPSCounter.getPrimitiveTotal();
}



//! \return Returns the name of the video driver. Example: In case of the DIRECT3D8
//! driver, it would return "Direct3D8".

const wchar_t* CNullDriver::getName() const
{
	return L"Irrlicht NullDevice";
}



//! Returns the maximum amount of primitives (mostly vertices) which
//! the device is able to render with one drawIndexedTriangleList
//! call.
uint32_t CNullDriver::getMaximalIndicesCount() const
{
	return 0xFFFFFFFF;
}


//! Enables or disables a texture creation flag.
void CNullDriver::setTextureCreationFlag(E_TEXTURE_CREATION_FLAG flag, bool enabled)
{
	if (enabled && ((flag == ETCF_ALWAYS_16_BIT) || (flag == ETCF_ALWAYS_32_BIT)))
	{
		// disable other formats
		setTextureCreationFlag(ETCF_ALWAYS_16_BIT, false);
		setTextureCreationFlag(ETCF_ALWAYS_32_BIT, false);
	}

	// set flag
	TextureCreationFlags = (TextureCreationFlags & (~flag)) |
		((((uint32_t)!enabled)-1) & flag);
}


//! Returns if a texture creation flag is enabled or disabled.
bool CNullDriver::getTextureCreationFlag(E_TEXTURE_CREATION_FLAG flag) const
{
	return (TextureCreationFlags & flag)!=0;
}

//!
void CNullDriver::drawMeshBuffer(const IGPUMeshBuffer* mb)
{
	if (!mb)
		return;

    uint32_t increment = mb->getInstanceCount();
    switch (mb->getPipeline()->getPrimitiveAssemblyParams().primitiveType)
    {
        case asset::EPT_POINT_LIST:
            increment *= mb->getIndexCount();
            break;
        case asset::EPT_LINE_STRIP:
            increment *= mb->getIndexCount()-1;
            break;
        case asset::EPT_LINE_LIST:
            increment *= mb->getIndexCount()/2;
            break;
        case asset::EPT_TRIANGLE_STRIP:
            increment *= mb->getIndexCount()-2;
            break;
        case asset::EPT_TRIANGLE_FAN:
            increment *= mb->getIndexCount()-2;
            break;
        case asset::EPT_TRIANGLE_LIST:
            increment *= mb->getIndexCount()/3;
            break;
    }
    PrimitivesDrawn += increment;
}


//! Indirect Draw
void CNullDriver::drawArraysIndirect(const asset::IMeshDataFormatDesc<video::IGPUBuffer>* vao,
                                     const asset::E_PRIMITIVE_TYPE& mode,
                                     const IGPUBuffer* indirectDrawBuff,
                                     const size_t& offset, const size_t& count, const size_t& stride)
{
}

void CNullDriver::drawIndexedIndirect(  const asset::IMeshDataFormatDesc<video::IGPUBuffer>* vao,
                                        const asset::E_PRIMITIVE_TYPE& mode,
                                        const asset::E_INDEX_TYPE& type,
                                        const IGPUBuffer* indirectDrawBuff,
                                        const size_t& offset, const size_t& count, const size_t& stride)
{
}


void CNullDriver::beginQuery(IQueryObject* query)
{
    if (!query)
        return; //error

    if (currentQuery[query->getQueryObjectType()])
        return; //error

    query->grab();
    currentQuery[query->getQueryObjectType()][0] = query;
}
void CNullDriver::endQuery(IQueryObject* query)
{
    if (!query)
        return; //error
    if (currentQuery[query->getQueryObjectType()]!=query)
        return; //error

    if (currentQuery[query->getQueryObjectType()])
        currentQuery[query->getQueryObjectType()]->drop();
    currentQuery[query->getQueryObjectType()][0] = NULL;
}


//! Only used by the internal engine. Used to notify the driver that
//! the window was resized.
void CNullDriver::OnResize(const core::dimension2d<uint32_t>& size)
{
	ScreenSize = size;
}


//! Returns driver and operating system specific data about the IVideoDriver.
const SExposedVideoData& CNullDriver::getExposedVideoData()
{
	return ExposedData;
}


//! Returns type of video driver
E_DRIVER_TYPE CNullDriver::getDriverType() const
{
	return EDT_NULL;
}

/*
//! Returns pointer to the IGPUProgrammingServices interface.
IGPUProgrammingServices* CNullDriver::getGPUProgrammingServices()
{
	return this;
}

int32_t CNullDriver::addHighLevelShaderMaterial(
    const char* vertexShaderProgram,
    const char* controlShaderProgram,
    const char* evaluationShaderProgram,
    const char* geometryShaderProgram,
    const char* pixelShaderProgram,
    uint32_t patchVertices,
    E_MATERIAL_TYPE baseMaterial,
    IShaderConstantSetCallBack* callback,
    const char** xformFeedbackOutputs,
    const uint32_t& xformFeedbackOutputCount,
    int32_t userData,
    const char* vertexShaderEntryPointName,
    const char* controlShaderEntryPointName,
    const char* evaluationShaderEntryPointName,
    const char* geometryShaderEntryPointName,
    const char* pixelShaderEntryPointName)
{
	os::Printer::log("High level shader materials not available (yet) in this driver, sorry");
	return -1;
}

bool CNullDriver::replaceHighLevelShaderMaterial(const int32_t &materialIDToReplace,
    const char* vertexShaderProgram,
    const char* controlShaderProgram,
    const char* evaluationShaderProgram,
    const char* geometryShaderProgram,
    const char* pixelShaderProgram,
    uint32_t patchVertices,
    E_MATERIAL_TYPE baseMaterial,
    IShaderConstantSetCallBack* callback,
    const char** xformFeedbackOutputs,
    const uint32_t& xformFeedbackOutputCount,
    int32_t userData,
    const char* vertexShaderEntryPointName,
    const char* controlShaderEntryPointName,
    const char* evaluationShaderEntryPointName,
    const char* geometryShaderEntryPointName,
    const char* pixelShaderEntryPointName)
{
    if (materialIDToReplace<0 || MaterialRenderers.size()<static_cast<uint32_t>(materialIDToReplace))
        return false;

    int32_t nr = addHighLevelShaderMaterial(vertexShaderProgram,controlShaderProgram,evaluationShaderProgram,geometryShaderProgram,pixelShaderProgram,
                                        3,baseMaterial,callback,xformFeedbackOutputs,xformFeedbackOutputCount,userData,
                                        vertexShaderEntryPointName,controlShaderEntryPointName,evaluationShaderEntryPointName,geometryShaderEntryPointName,pixelShaderEntryPointName);
    if (nr==-1)
        return false;

	MaterialRenderers[materialIDToReplace].Renderer->drop();
	MaterialRenderers[materialIDToReplace] = MaterialRenderers[MaterialRenderers.size()-1];

	MaterialRenderers.resize(MaterialRenderers.size()-1);

    return true;
}

int32_t CNullDriver::addHighLevelShaderMaterialFromFiles(
    const io::path& vertexShaderProgramFileName,
    const io::path& controlShaderProgramFileName,
    const io::path& evaluationShaderProgramFileName,
    const io::path& geometryShaderProgramFileName,
    const io::path& pixelShaderProgramFileName,
    uint32_t patchVertices,
    E_MATERIAL_TYPE baseMaterial,
    IShaderConstantSetCallBack* callback,
    const char** xformFeedbackOutputs,
    const uint32_t& xformFeedbackOutputCount,
    int32_t userData,
    const char* vertexShaderEntryPointName,
    const char* controlShaderEntryPointName,
    const char* evaluationShaderEntryPointName,
    const char* geometryShaderEntryPointName,
    const char* pixelShaderEntryPointName)
{
	io::IReadFile* vsfile = 0;
	io::IReadFile* gsfile = 0;
	io::IReadFile* ctsfile = 0;
	io::IReadFile* etsfile = 0;
	io::IReadFile* psfile = 0;

	if (vertexShaderProgramFileName.size() )
	{
		vsfile = FileSystem->createAndOpenFile(vertexShaderProgramFileName);
		if (!vsfile)
		{
			os::Printer::log("Could not open vertex shader program file",
				vertexShaderProgramFileName.c_str(), ELL_WARNING);
		}
	}

	if (controlShaderProgramFileName.size() )
	{
		ctsfile = FileSystem->createAndOpenFile(controlShaderProgramFileName);
		if (!ctsfile)
		{
			os::Printer::log("Could not open control shader program file",
				controlShaderProgramFileName.c_str(), ELL_WARNING);
		}
	}

	if (evaluationShaderProgramFileName.size() )
	{
		etsfile = FileSystem->createAndOpenFile(evaluationShaderProgramFileName);
		if (!etsfile)
		{
			os::Printer::log("Could not open evaluation shader program file",
				evaluationShaderProgramFileName.c_str(), ELL_WARNING);
		}
	}

	if (geometryShaderProgramFileName.size() )
	{
		gsfile = FileSystem->createAndOpenFile(geometryShaderProgramFileName);
		if (!gsfile)
		{
			os::Printer::log("Could not open geometry shader program file",
				geometryShaderProgramFileName.c_str(), ELL_WARNING);
		}
	}

	if (pixelShaderProgramFileName.size() )
	{
		psfile = FileSystem->createAndOpenFile(pixelShaderProgramFileName);
		if (!psfile)
		{
			os::Printer::log("Could not open pixel shader program file",
				pixelShaderProgramFileName.c_str(), ELL_WARNING);
		}
	}

	int32_t result = addHighLevelShaderMaterialFromFiles(
		vsfile, ctsfile, etsfile, gsfile, psfile,
		patchVertices, baseMaterial, callback,
		xformFeedbackOutputs,xformFeedbackOutputCount, userData,
		vertexShaderEntryPointName, controlShaderEntryPointName,
		evaluationShaderEntryPointName, geometryShaderEntryPointName,
		pixelShaderEntryPointName);

	if (psfile)
		psfile->drop();

	if (ctsfile)
		ctsfile->drop();

	if (etsfile)
		etsfile->drop();

	if (gsfile)
		gsfile->drop();

	if (vsfile)
		vsfile->drop();

	return result;
}

int32_t CNullDriver::addHighLevelShaderMaterialFromFiles(
    io::IReadFile* vertexShaderProgram,
    io::IReadFile* controlShaderProgram,
    io::IReadFile* evaluationShaderProgram,
    io::IReadFile* geometryShaderProgram,
    io::IReadFile* pixelShaderProgram,
    uint32_t patchVertices,
    E_MATERIAL_TYPE baseMaterial,
    IShaderConstantSetCallBack* callback,
    const char** xformFeedbackOutputs,
    const uint32_t& xformFeedbackOutputCount,
    int32_t userData,
    const char* vertexShaderEntryPointName,
    const char* controlShaderEntryPointName,
    const char* evaluationShaderEntryPointName,
    const char* geometryShaderEntryPointName,
    const char* pixelShaderEntryPointName)
{
	char* vs = 0;
	char* cts = 0;
	char* ets = 0;
	char* gs = 0;
	char* ps = 0;

	if (vertexShaderProgram)
	{
		const long size = vertexShaderProgram->getSize();
		if (size)
		{
			vs = new char[size+1];
			vertexShaderProgram->read(vs, size);
			vs[size] = 0;
		}
	}

	if (pixelShaderProgram)
	{
		const long size = pixelShaderProgram->getSize();
		if (size)
		{
			// if both handles are the same we must reset the file
			if (pixelShaderProgram==vertexShaderProgram)
				pixelShaderProgram->seek(0);
			ps = new char[size+1];
			pixelShaderProgram->read(ps, size);
			ps[size] = 0;
		}
	}

	if (geometryShaderProgram)
	{
		const long size = geometryShaderProgram->getSize();
		if (size)
		{
			// if both handles are the same we must reset the file
			if ((geometryShaderProgram==vertexShaderProgram) ||
					(geometryShaderProgram==pixelShaderProgram))
				geometryShaderProgram->seek(0);
			gs = new char[size+1];
			geometryShaderProgram->read(gs, size);
			gs[size] = 0;
		}
	}

	if (controlShaderProgram)
	{
		const long size = controlShaderProgram->getSize();
		if (size)
		{
			// if both handles are the same we must reset the file
			if ((controlShaderProgram==vertexShaderProgram) ||
					(controlShaderProgram==pixelShaderProgram) ||
                        (controlShaderProgram==geometryShaderProgram))
				controlShaderProgram->seek(0);
			cts = new char[size+1];
			controlShaderProgram->read(cts, size);
			cts[size] = 0;
		}
	}

	if (evaluationShaderProgram)
	{
		const long size = evaluationShaderProgram->getSize();
		if (size)
		{
			// if both handles are the same we must reset the file
			if ((evaluationShaderProgram==vertexShaderProgram) ||
					(evaluationShaderProgram==pixelShaderProgram) ||
                        (evaluationShaderProgram==geometryShaderProgram) ||
                            (evaluationShaderProgram==controlShaderProgram))
				evaluationShaderProgram->seek(0);
			ets = new char[size+1];
			evaluationShaderProgram->read(ets, size);
			ets[size] = 0;
		}
	}


	int32_t result = this->addHighLevelShaderMaterial(
		vs, cts, ets, gs, ps,patchVertices,
		baseMaterial, callback,
		xformFeedbackOutputs,xformFeedbackOutputCount, userData,
		vertexShaderEntryPointName,
		controlShaderEntryPointName,
		evaluationShaderEntryPointName,
		geometryShaderEntryPointName,
		pixelShaderEntryPointName);

    if (vs)
        delete [] vs;
    if (ps)
        delete [] ps;
    if (gs)
        delete [] gs;
    if (cts)
        delete [] cts;
    if (ets)
        delete [] ets;

	return result;
}
*/

void CNullDriver::blitRenderTargets(IFrameBuffer* in, IFrameBuffer* out, bool copyDepth, bool copyStencil,
									core::recti srcRect, core::recti dstRect,
									bool bilinearFilter)
{
}


//! Clears the ZBuffer.
void CNullDriver::clearZBuffer(const float &depth)
{
}
void CNullDriver::clearStencilBuffer(const int32_t &stencil)
{
}
void CNullDriver::clearZStencilBuffers(const float &depth, const int32_t &stencil)
{
}
void CNullDriver::clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const int32_t* vals)
{
}
void CNullDriver::clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const uint32_t* vals)
{
}
void CNullDriver::clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const float* vals)
{
}
void CNullDriver::clearScreen(const E_SCREEN_BUFFERS &buffer, const float* vals)
{
}
void CNullDriver::clearScreen(const E_SCREEN_BUFFERS &buffer, const uint32_t* vals)
{
}


// prints renderer version
void CNullDriver::printVersion()
{
	core::stringw namePrint = L"Using renderer: ";
	namePrint += getName();
	os::Printer::log(namePrint.c_str(), ELL_INFORMATION);
}


//! creates a video driver
IVideoDriver* createNullDriver(IrrlichtDevice* dev, io::IFileSystem* io, const core::dimension2d<uint32_t>& screenSize)
{
	CNullDriver* nullDriver = new CNullDriver(dev, io, screenSize);

	return nullDriver;
}



//! Enable/disable a clipping plane.
void CNullDriver::enableClipPlane(uint32_t index, bool enable)
{
	// not necessary
}


const uint32_t* CNullDriver::getMaxTextureSize(const ITexture::E_TEXTURE_TYPE& type) const
{
    return MaxTextureSizes[type];
}

} // end namespace
} // end namespace

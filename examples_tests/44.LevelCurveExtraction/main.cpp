#define _IRR_STATIC_LIB_
#include "InputEventReciever.h"
#include "../common/QToQuitEventReceiver.h"
#include "../../ext/FullScreenTriangle/FullScreenTriangle.h"
#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"


const char* vertShaderCode = R"===(
#version 430 core
layout(location = 0) in vec4 vPos;
layout( push_constant, row_major ) uniform Block {
	mat4 modelViewProj;
} PushConstants;

void main()
{
    gl_Position = PushConstants.modelViewProj * vPos;
}
)===";
const char* fragShaderCode = R"===(
#version 430 core
layout(location = 0) out vec4 pixelColor;
void main()
{
    pixelColor = vec4(0,1,0,1);
}
)===";

const char* geometryShaderCode = R"===(
#version 440 core

#define FLT_MAX 3.402823466e+38
struct DrawIndirectArrays_t
{
    uint  count;
    uint  instanceCount;
    uint  first;
    uint  baseInstance;
};



layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
layout(location = 0) in vec3 LocalPos[];
layout(location = 1) in vec3 ViewPos[];
layout(location = 2) in vec3 Normal[];
layout(location = 0) out vec3 fragLocalPos;
layout(location = 1) out vec3 fragViewPos;
layout(location = 2) out vec3 fragNormal;
#ifndef _NO_UV
layout(location = 3) in vec2 UV[];
layout(location = 3) out vec2 fragUV;
#endif

layout(set = 0, binding = 0) coherent buffer LineCount
{
    DrawIndirectArrays_t lineDraw;
};
layout(set = 0, binding = 1) writeonly buffer Lines
{
    float linePoints[]; // 6 floats decribe a line, 3d start, 3d end
};
layout(set = 0, binding = 2) uniform LevelCurveSettings
{
    float intersectionPlaneSpacing;
    vec3 intersectionPlaneNormal;
};
void main() {
    const float levelPlanesDistance = intersectionPlaneSpacing;
    const vec3 levelPlaneNormal = vec3(0.0,1.0,0.0);   
    uint numHorLines;
    float maxLevel = -FLT_MAX;
    float minLevel = FLT_MAX;
    uint i;
    float vertexPlaneDistance[3];
    for (i = 0; i < 3; i++)
    {
        vertexPlaneDistance[i] =dot(levelPlaneNormal, LocalPos[i]);
        maxLevel = max(maxLevel,vertexPlaneDistance[i]);
        minLevel = min(minLevel,vertexPlaneDistance[i]);
    }
    int sharedMxVtx = 0,sharedMnVtx = 0;
    for (i = 0; i < 3; i++)
    {
        if (vertexPlaneDistance[i] == maxLevel)
			sharedMxVtx++;
		if (vertexPlaneDistance[i] == minLevel)
			sharedMnVtx++;
    }
    if(sharedMnVtx <2)
		minLevel += 0.001f;
	if(sharedMxVtx < 2)
		maxLevel -= 0.001f;
  
    numHorLines = uint(floor(maxLevel/levelPlanesDistance)-ceil(minLevel/levelPlanesDistance-1));
    if(numHorLines>0)
    {
        uint outID = atomicAdd(lineDraw.count,2 * numHorLines) * 3;
        float beginLevel = ceil(minLevel/levelPlanesDistance)*levelPlanesDistance;

        const vec3 edgeVectors[3] = vec3[3](
            LocalPos[1]-LocalPos[0],
            LocalPos[2]-LocalPos[1],
            LocalPos[0]-LocalPos[2]);

        const float edgeMinMax[6] = float[6](
            dot(levelPlaneNormal,LocalPos[0]),
            dot(levelPlaneNormal,LocalPos[1]), 
            dot(levelPlaneNormal,LocalPos[1]),
            dot(levelPlaneNormal,LocalPos[2]), 
            dot(levelPlaneNormal,LocalPos[2]),
            dot(levelPlaneNormal,LocalPos[0]));
            
        const float edgePlaneDot[3] = float[3](
            dot(edgeVectors[0],levelPlaneNormal),
            dot(edgeVectors[1],levelPlaneNormal),
            dot(edgeVectors[2],levelPlaneNormal));

        for (i=0; i<numHorLines; i++)
        {
            float d = float(i) *levelPlanesDistance + beginLevel;
            for(int j = 0; j < 3; j++)
            {
                float mx= max(edgeMinMax[j*2],edgeMinMax[j*2+1]);
                float mn= min(edgeMinMax[j*2],edgeMinMax[j*2+1]);
                if(d>= mn && d <= mx)
                {
                    float w_y = edgePlaneDot[j];
                    float t;
                    if(w_y==0)
                        t=0;
                    else
                        t=(d-edgeMinMax[j*2])/w_y;

                    vec3 outputIntersection = LocalPos[j] + edgeVectors[j] * t;
                    linePoints[outID++] = outputIntersection.x;
                    linePoints[outID++] = outputIntersection.y;
                    linePoints[outID++] = outputIntersection.z;
                }
            }

        }

     
    }
    //passthrough part
    for (i = 0; i < gl_in.length(); i++)
    {
        fragLocalPos = LocalPos[i];
        fragViewPos = ViewPos[i];
        fragNormal = Normal[i];
        gl_Position = gl_in[i].gl_Position;
#ifndef _NO_UV
        fragUV = UV[i];
#endif
        EmitVertex();
    }
    EndPrimitive();

}
)===";

struct SLinesSettings {
public:
    float spacing;
    float x, y, z;
};

using namespace irr;
using namespace core;

int main()
{
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	auto device = createDeviceEx(params);

	if (!device)
		return 1; // could not create selected driver.


	//! Since our cursor will be enslaved, there will be no way to close the window
	//! So we listen for the "Q" key being pressed and exit the application
    //Also
    //Get input from page up and page down
    //Decrement spacing depending on it.
    ChgSpacingEventReciever receiver;
	device->setEventReceiver(&receiver);


	auto* driver = device->getVideoDriver();
	auto* smgr = device->getSceneManager();
    auto* am = device->getAssetManager();
    auto* fs = am->getFileSystem();

    //
    auto* qnc = am->getMeshManipulator()->getQuantNormalCache();
    //loading cache from file
    qnc->loadNormalQuantCacheFromFile<asset::CQuantNormalCache::E_CACHE_TYPE::ECT_2_10_10_10>(fs,"../../tmp/normalCache101010.sse", true);

    asset::IAssetLoader::SAssetLoadParams lp;
    asset::SAssetBundle meshes_bundle;
    pfd::message("Choose file to open", "Choose an OBJ file to open or press cancel to open a default scene.", pfd::choice::ok);
    while (true)
    {
        pfd::open_file file("Choose an OBJ file", "", { "OBJ files (.obj)", "*.obj" });
        if (!file.result().empty())
        {
            //lp.loaderFlags = asset::IAssetLoader::ELPF_DONT_COMPILE_GLSL;
            meshes_bundle = am->getAsset(file.result()[0], lp);
            if (meshes_bundle.isEmpty())
            {
                pfd::message("Choose file to open", "Chosen file could not be loaded. Choose another OBJ file to open or press cancel to open a default scene.", pfd::choice::ok);
                continue;
            }
            break;
        }
        else
        {

            fs->addFileArchive("../../media/sponza.zip");
            meshes_bundle = am->getAsset("sponza.obj", lp);
            if (meshes_bundle.isEmpty())
            { 
                std::cout << "Could not open Sponza.zip. Quitting program";
                return 1;
            }
            break;
        }
    }

    //! disable mouse cursor, since camera will force it to the middle
    //! and we don't want a jittery cursor in the middle distracting us
    device->getCursorControl()->setVisible(false);


    auto mesh = meshes_bundle.getContents().begin()[0];
    auto mesh_raw = static_cast<asset::ICPUMesh*>(mesh.get());

    //saving cache to file
    qnc->saveCacheToFile(asset::CQuantNormalCache::E_CACHE_TYPE::ECT_2_10_10_10,fs,"../../tmp/normalCache101010.sse");
  
    //copy the pipeline
    auto pipeline_cp = core::smart_refctd_ptr_static_cast<asset::ICPURenderpassIndependentPipeline>(mesh_raw->getMeshBuffer(0u)->getPipeline()->clone(3u));
    //get the simple geometry shader data and turn it into ICPUSpecializedShader
    auto unspecializedVertShader = driver->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(vertShaderCode));
    auto unspecializedFragShader = driver->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(fragShaderCode));
    auto unspecializedGeomShader = core::make_smart_refctd_ptr<asset::ICPUShader>(geometryShaderCode);


    auto vshader = driver->createGPUSpecializedShader(unspecializedVertShader.get(), asset::ISpecializedShader::SInfo({}, nullptr, "main", asset::ISpecializedShader::ESS_VERTEX));
    auto fshader = driver->createGPUSpecializedShader(unspecializedFragShader.get(), asset::ISpecializedShader::SInfo({}, nullptr, "main", asset::ISpecializedShader::ESS_FRAGMENT));
    auto geomShader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecializedGeomShader), asset::ISpecializedShader::SInfo({}, nullptr, "main", asset::ISpecializedShader::ESS_GEOMETRY));

    pipeline_cp->setShaderAtIndex(asset::ICPURenderpassIndependentPipeline::ESSI_GEOMETRY_SHADER_IX, geomShader.get());
    auto* layout = pipeline_cp->getLayout();
    core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> ds0layout;
    {
        asset::ICPUDescriptorSetLayout::SBinding b[3];
        b[0].binding = 0u;
        b[0].count = 1u;
        b[0].samplers = nullptr;
        b[0].stageFlags = asset::ISpecializedShader::ESS_GEOMETRY;
        b[0].type = asset::EDT_STORAGE_BUFFER;

        b[1].binding = 1u;
        b[1].count = 1u;
        b[1].samplers = nullptr;
        b[1].stageFlags = asset::ISpecializedShader::ESS_GEOMETRY;
        b[1].type = asset::EDT_STORAGE_BUFFER;

        b[2].binding = 2u;
        b[2].count = 1u;
        b[2].samplers = nullptr;
        b[2].stageFlags = asset::ISpecializedShader::ESS_GEOMETRY;
        b[2].type = asset::EDT_UNIFORM_BUFFER;
        ds0layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(b,b+3);
    }

    //set up draw indirect pipeline
    asset::SPrimitiveAssemblyParams assemblyParams = { asset::EPT_LINE_LIST,false,2u };
    asset::SVertexInputParams inputParams;
    inputParams.enabledAttribFlags = 0b11u;
    inputParams.enabledBindingFlags = 0b1u;
    inputParams.attributes[0].binding = 0u;
    inputParams.attributes[0].format = asset::EF_R32G32B32_SFLOAT;
    inputParams.attributes[0].relativeOffset = 0u;
    inputParams.bindings[0].stride = sizeof(float)*3;
    inputParams.bindings[0].inputRate = asset::EVIR_PER_VERTEX;

    asset::SPushConstantRange pcRange[1] = { asset::ISpecializedShader::ESS_VERTEX,0,sizeof(core::matrix4SIMD)
         };
    auto pLayout = driver->createGPUPipelineLayout(pcRange, pcRange + 1u, nullptr, nullptr, nullptr, nullptr);
    video::IGPUSpecializedShader* shaders[2] = { vshader.get(),fshader.get() };
    asset::SBlendParams blendParams;
    blendParams.logicOpEnable = false;
    blendParams.logicOp = asset::ELO_NO_OP;
    for (size_t i = 1ull; i < asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
        blendParams.blendParams[i].attachmentEnabled = false;
    asset::SRasterizationParams rasterParams;
    rasterParams.polygonMode = asset::EPM_LINE;
    rasterParams.depthBiasConstantFactor = 1;
    rasterParams.depthBiasSlopeFactor = 1;

    auto drawIndirect_pipeline = driver->createGPURenderpassIndependentPipeline(nullptr, std::move(pLayout), shaders, shaders + sizeof(shaders) / sizeof(void*), inputParams, blendParams, assemblyParams, rasterParams);


    
    layout->setDescriptorSetLayout(0,core::smart_refctd_ptr(ds0layout));
    auto gpuds0layout = driver->getGPUObjectsFromAssets(&ds0layout, &ds0layout + 1)->front();

    for (size_t i = 0; i < mesh_raw->getMeshBufferCount(); i++)
    {
        mesh_raw->getMeshBuffer(i)->setPipeline(core::smart_refctd_ptr(pipeline_cp));
    }


    //create buffers for the geometry shader
    asset::DrawArraysIndirectCommand_t drawArraysIndirectCmd;
    drawArraysIndirectCmd.instanceCount = 1u;
    drawArraysIndirectCmd.baseInstance = 0u;
    drawArraysIndirectCmd.count = 0u;
    drawArraysIndirectCmd.first = 0u;
    //auto lineCountBuffer = driver->createDeviceLocalGPUBufferOnDedMem(roundUp(sizeof(irr::asset::DrawArraysIndirectCommand_t),16ull));
    auto lineCountBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(roundUp(sizeof(irr::asset::DrawArraysIndirectCommand_t), 16ull), &drawArraysIndirectCmd);
    uint32_t triangleCount;
    if (!asset::IMeshManipulator::getPolyCount(triangleCount, mesh_raw))
        assert(false);
    float levelCurveSpacing = 10.0f;
    float planeX = 0, planeY = 1, planeZ = 0;
    SLinesSettings lineSettings;
    lineSettings.spacing = levelCurveSpacing;
    lineSettings.x = planeX;
    lineSettings.y = planeY;
    lineSettings.z = planeZ;

    auto linesBuffer = driver->createDeviceLocalGPUBufferOnDedMem(triangleCount * 6 * sizeof(float));
    auto uniformLinesSettingsBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(roundUp(4 * sizeof(float), 16ull), &lineSettings);
    
    auto gpuds0 = driver->createGPUDescriptorSet(std::move(gpuds0layout));
    {
        video::IGPUDescriptorSet::SWriteDescriptorSet w[3];
        video::IGPUDescriptorSet::SDescriptorInfo i[3];
        w[0].arrayElement = 0;
        w[0].binding = 0u;
        w[0].count = 1u;
        w[0].descriptorType = asset::EDT_STORAGE_BUFFER;
        w[0].dstSet = gpuds0.get();
        w[0].info = i;
        i[0].desc = lineCountBuffer;
        i[0].buffer.offset = 0;
        i[0].buffer.size = lineCountBuffer->getSize();

        w[1].arrayElement = 0;
        w[1].binding = 1u;
        w[1].count = 1u;
        w[1].descriptorType = asset::EDT_STORAGE_BUFFER;
        w[1].dstSet = gpuds0.get();
        w[1].info = i+1;
        i[1].desc = linesBuffer;
        i[1].buffer.offset = 0;
        i[1].buffer.size = linesBuffer->getSize();

        w[2].arrayElement = 0;
        w[2].binding = 2u;
        w[2].count = 1u;
        w[2].descriptorType = asset::EDT_UNIFORM_BUFFER;
        w[2].dstSet = gpuds0.get();
        w[2].info = i + 2;
        i[2].desc = uniformLinesSettingsBuffer;
        i[2].buffer.offset = 0;
        i[2].buffer.size = uniformLinesSettingsBuffer->getSize();
        driver->updateDescriptorSets(3u, w, 0u, nullptr);
    }

    asset::SBufferBinding<video::IGPUBuffer> bufferBinding[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
    bufferBinding[0].offset = 0;
    bufferBinding[0].buffer = linesBuffer; 
   for (size_t i = 1; i < video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; i++)
    {
        bufferBinding[i].offset = 0;
        bufferBinding[i].buffer = nullptr;
    }

    


    //we can safely assume that all meshbuffers within mesh loaded from OBJ has same DS1 layout (used for camera-specific data)
    //so we can create just one DS
    asset::ICPUDescriptorSetLayout* ds1layout = mesh_raw->getMeshBuffer(0u)->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
    uint32_t ds1UboBinding = 0u;
    for (const auto& bnd : ds1layout->getBindings())
        if (bnd.type==asset::EDT_UNIFORM_BUFFER)
        {
            ds1UboBinding = bnd.binding;
            break;
        }

    size_t neededDS1UBOsz = 0ull;
    {
        auto pipelineMetadata = static_cast<const asset::IPipelineMetadata*>(mesh_raw->getMeshBuffer(0u)->getPipeline()->getMetadata());
        for (const auto& shdrIn : pipelineMetadata->getCommonRequiredInputs())
            if (shdrIn.descriptorSection.type==asset::IPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set==1u && shdrIn.descriptorSection.uniformBufferObject.binding==ds1UboBinding)
                neededDS1UBOsz = std::max<size_t>(neededDS1UBOsz, shdrIn.descriptorSection.uniformBufferObject.relByteoffset+shdrIn.descriptorSection.uniformBufferObject.bytesize);
    }

    auto gpuds1layout = driver->getGPUObjectsFromAssets(&ds1layout, &ds1layout+1)->front();

    auto gpuubo = driver->createDeviceLocalGPUBufferOnDedMem(neededDS1UBOsz);
    auto gpuds1 = driver->createGPUDescriptorSet(std::move(gpuds1layout));
    {
        video::IGPUDescriptorSet::SWriteDescriptorSet write;
        write.dstSet = gpuds1.get();
        write.binding = ds1UboBinding;
        write.count = 1u;
        write.arrayElement = 0u;
        write.descriptorType = asset::EDT_UNIFORM_BUFFER;
        video::IGPUDescriptorSet::SDescriptorInfo info;
        {
            info.desc = gpuubo;
            info.buffer.offset = 0ull;
            info.buffer.size = neededDS1UBOsz;
        }
        write.info = &info;
        driver->updateDescriptorSets(1u, &write, 0u, nullptr);
    }
    //DescriptorSetLayout is null
    auto gpumesh = driver->getGPUObjectsFromAssets(&mesh_raw, &mesh_raw+1)->front();

	//! we want to move around the scene and view it from different angles
	scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0,100.0f,0.5f);

	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(1.f);
	camera->setFarValue(5000.0f);

    smgr->setActiveCamera(camera);
	uint64_t lastFPSTime = 0;
    
	while(device->run() && receiver.keepOpen())
	{
        driver->beginScene(true, true, video::SColor(255,128,128,128) );

        if (levelCurveSpacing != receiver.getSpacing())
        {
            levelCurveSpacing = receiver.getSpacing();
            driver->updateBufferRangeViaStagingBuffer(uniformLinesSettingsBuffer.get(), 0, sizeof(float), &levelCurveSpacing);
        }

        
      
        // zero out buffer LineCount
        driver->fillBuffer(lineCountBuffer.get(), offsetof(irr::asset::DrawArraysIndirectCommand_t, count), sizeof(uint32_t), 0u);

        //emit "memory barrier" of type GL_SHADER_STORAGE_BITS before scene is drawn - same as pre render? or post invoking render but before it finishes?
        //did you mean GL_SHADER_STORAGE_BARRIER_BIT?
        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        //! This animates (moves) the camera and sets the transforms
		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());


		camera->render();


        core::vector<uint8_t> uboData(gpuubo->getSize());
        auto pipelineMetadata = static_cast<const asset::IPipelineMetadata*>(mesh_raw->getMeshBuffer(0u)->getPipeline()->getMetadata());
        for (const auto& shdrIn : pipelineMetadata->getCommonRequiredInputs())
        {
            if (shdrIn.descriptorSection.type==asset::IPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set==1u && shdrIn.descriptorSection.uniformBufferObject.binding==ds1UboBinding)
            {
                switch (shdrIn.type)
                {
                case asset::IPipelineMetadata::ECSI_WORLD_VIEW_PROJ:
                {
                    core::matrix4SIMD mvp = camera->getConcatenatedMatrix();
                    memcpy(uboData.data()+shdrIn.descriptorSection.uniformBufferObject.relByteoffset, mvp.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                }
                break;
                case asset::IPipelineMetadata::ECSI_WORLD_VIEW:
                {
                    core::matrix3x4SIMD MV = camera->getViewMatrix();
                    memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, MV.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                }
                break;
                case asset::IPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE:
                {
                    core::matrix3x4SIMD MV = camera->getViewMatrix();
                    memcpy(uboData.data()+shdrIn.descriptorSection.uniformBufferObject.relByteoffset, MV.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                }
                break;
                }
            }
        }       
        driver->updateBufferRangeViaStagingBuffer(gpuubo.get(), 0ull, gpuubo->getSize(), uboData.data());

        for (uint32_t i = 0u; i < gpumesh->getMeshBufferCount(); ++i)
        {
            video::IGPUMeshBuffer* gpumb = gpumesh->getMeshBuffer(i);
            const video::IGPURenderpassIndependentPipeline* pipeline = gpumb->getPipeline();  
            const video::IGPUDescriptorSet* ds3 = gpumb->getAttachedDescriptorSet();

            driver->bindGraphicsPipeline(pipeline);
            driver->bindDescriptorSets(video::EPBP_GRAPHICS, pipeline->getLayout(), 0u, 1u, &gpuds0.get(), nullptr);
            const video::IGPUDescriptorSet* gpuds1_ptr = gpuds1.get();
            driver->bindDescriptorSets(video::EPBP_GRAPHICS, pipeline->getLayout(), 1u, 1u, &gpuds1_ptr, nullptr);
            const video::IGPUDescriptorSet* gpuds3_ptr = gpumb->getAttachedDescriptorSet();
            if (gpuds3_ptr)
                driver->bindDescriptorSets(video::EPBP_GRAPHICS, pipeline->getLayout(), 3u, 1u, &gpuds3_ptr, nullptr);
            driver->pushConstants(pipeline->getLayout(), video::IGPUSpecializedShader::ESS_FRAGMENT, 0u, gpumb->MAX_PUSH_CONSTANT_BYTESIZE, gpumb->getPushConstantsDataPtr());
            driver->drawMeshBuffer(gpumb);
        }

        //emit "memory barrier" of type GL_ALL_BARRIER_BITS after the entire scene finishes drawing
        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_ALL_BARRIER_BITS);
        
        
        driver->bindGraphicsPipeline(drawIndirect_pipeline.get());
        driver->pushConstants(drawIndirect_pipeline->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), camera->getConcatenatedMatrix().pointer());


        //invoke drawIndirect and use linesBuffer
        driver->drawArraysIndirect(bufferBinding, asset::EPT_LINE_LIST, lineCountBuffer.get(), 0u, 1u, sizeof(asset::DrawArraysIndirectCommand_t));
		driver->endScene();
      
		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Level Curve Extraction Demo - IrrlichtBAW Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}

	return 0;
}
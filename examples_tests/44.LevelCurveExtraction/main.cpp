// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_

#include "InputEventReciever.h"
#include "../source/Nabla/COpenGLExtensionHandler.h"
#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"

#include "common.glsl"


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
    gl_FragDepth = gl_FragCoord.z*1.002;
    pixelColor = vec4(0,1,0,1);
}
)===";

const char* geometryShaderCode = R"===(
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

#include "../common.glsl"

void main()
{
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
        uint outID = atomicAdd(lineDraw[mdiIndex].count,2*numHorLines)*3;
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

using namespace nbl;
using namespace core;


void SaveBufferToCSV(video::IVideoDriver* driver, size_t mdiIndex, video::IGPUBuffer* drawIndirectBuffer, video::IGPUBuffer* lineBuffer)
{
    constexpr uint64_t timeoutInNanoSeconds = 15000000000u;
    const uint32_t alignment = sizeof(float);
    auto downloadStagingArea = driver->getDefaultDownStreamingBuffer();
    auto downBuffer = downloadStagingArea->getBuffer();

    auto getData = [&alignment,timeoutInNanoSeconds,downloadStagingArea,driver, downBuffer](video::IGPUBuffer* buf, const uint32_t origOffset, const uint32_t bytelen) -> void*
    {
        uint32_t address = std::remove_pointer<decltype(downloadStagingArea)>::type::invalid_address;
        auto unallocatedSize = downloadStagingArea->multi_alloc(1u, &address, &bytelen, &alignment);
        if (unallocatedSize)
        {
            os::Printer::log("Could not download the buffer from the GPU!", ELL_ERROR);
            return nullptr;
        }
        driver->copyBuffer(buf, downBuffer, origOffset, address, bytelen);
        auto downloadFence = driver->placeFence(true);
        auto result = downloadFence->waitCPU(timeoutInNanoSeconds, true);
        if (result==video::E_DRIVER_FENCE_RETVAL::EDFR_TIMEOUT_EXPIRED || result==video::E_DRIVER_FENCE_RETVAL::EDFR_FAIL)
        {
            os::Printer::log("Could not download the buffer from the GPU, fence not signalled!", ELL_ERROR);
            downloadStagingArea->multi_free(1u, &address, &bytelen, nullptr);
            return nullptr;
        }
        if (downloadStagingArea->needsManualFlushOrInvalidate())
            driver->invalidateMappedMemoryRanges({ {downloadStagingArea->getBuffer()->getBoundMemory(),address,bytelen} });
        // this is abuse of the API, the memory should be freed AFTER the pointer is finished being used, however no one else is using this staging buffer so we can allow it without experiencing data corruption
        downloadStagingArea->multi_free(1u, &address, &bytelen, nullptr);
        return reinterpret_cast<uint8_t*>(downloadStagingArea->getBufferPointer())+address;
    };

    // get the line count
    auto vertexCount = *reinterpret_cast<const uint32_t*>(getData(drawIndirectBuffer,sizeof(asset::DrawArraysIndirectCommand_t)*mdiIndex+offsetof(nbl::asset::DrawArraysIndirectCommand_t,count),sizeof(asset::DrawArraysIndirectCommand_t::count)));

    // get the lines
    auto* data = reinterpret_cast<float*>(getData(lineBuffer,0u,vertexCount*sizeof(float)*3u));
    std::ofstream csvFile ("../linesbuffer_content.csv");
    csvFile << "A_x;A_y;A_z;B_x;B_y;B_z\n";
    for (uint32_t i = 0; i<vertexCount*3u; i+=6u)
    {
        csvFile << data[i + 0] << ";";
        csvFile << data[i + 1] << ";";
        csvFile << data[i + 2] << ";";
        csvFile << data[i + 3] << ";";
        csvFile << data[i + 4] << ";";
        csvFile << data[i + 5];
        csvFile << "\n";
    }
    csvFile.close();
    os::Printer::log("Saved linesbuffer contents to a csv file");
}

int main()
{
    constexpr auto linesBufferSize = LINE_VERTEX_LIMIT*3u*sizeof(float); // 128 MB, max for Intel HD Graphics
    constexpr auto maxLineCount = linesBufferSize/(sizeof(float)*6u);

	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check nbl::SIrrlichtCreationParameters
	nbl::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
    params.StreamingDownloadBufferSize = linesBufferSize;
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


    // prepate geometry shaders for modified OBJ pipelines
    auto unspecializedGeomShaderUV = core::make_smart_refctd_ptr<asset::ICPUShader>((std::string("#version 440 core\n") + geometryShaderCode).c_str());
    auto unspecializedGeomShaderNOUV = core::make_smart_refctd_ptr<asset::ICPUShader>((std::string("#version 440 core\n#define _NO_UV\n") + geometryShaderCode).c_str());
    auto geomShaderUV = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecializedGeomShaderUV), asset::ISpecializedShader::SInfo({}, nullptr, "main", asset::ISpecializedShader::ESS_GEOMETRY));
    auto geomShaderNOUV = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecializedGeomShaderNOUV), asset::ISpecializedShader::SInfo({}, nullptr, "main", asset::ISpecializedShader::ESS_GEOMETRY));


    // create buffers for draw indirect structs
    asset::DrawArraysIndirectCommand_t drawArraysIndirectCmd[2] = { { 0u,1u,0u,0u }, { 0u,1u,0u,0u } };
    auto lineCountBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(core::alignUp(sizeof(drawArraysIndirectCmd), 16ull), drawArraysIndirectCmd);

    //create buffers for the geometry shader
    auto linesBuffer = driver->createDeviceLocalGPUBufferOnDedMem(linesBufferSize);

    auto uniformLinesSettingsBuffer = driver->createDeviceLocalGPUBufferOnDedMem(roundUp(sizeof(GlobalUniforms),16ull));

    // prepare line pipeline
    core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> drawIndirect_pipeline;
    {
        // set up the shaders
        auto lineVertShader = driver->createShader(core::make_smart_refctd_ptr<asset::ICPUShader>(vertShaderCode));
        auto lineFragShader = driver->createShader(core::make_smart_refctd_ptr<asset::ICPUShader>(fragShaderCode));
        auto linevshader = driver->createSpecializedShader(lineVertShader.get(), asset::ISpecializedShader::SInfo({}, nullptr, "main", asset::ISpecializedShader::ESS_VERTEX));
        auto linefshader = driver->createSpecializedShader(lineFragShader.get(), asset::ISpecializedShader::SInfo({}, nullptr, "main", asset::ISpecializedShader::ESS_FRAGMENT));

        // the layout
        asset::SPushConstantRange pcRange[1] = { asset::ISpecializedShader::ESS_VERTEX,0,sizeof(core::matrix4SIMD) };
        auto linePipelineLayout = driver->createPipelineLayout(pcRange, pcRange+1u, nullptr, nullptr, nullptr, nullptr);

        //set up the pipeline
        asset::SPrimitiveAssemblyParams assemblyParams = { asset::EPT_LINE_LIST,false,2u };
        asset::SVertexInputParams inputParams;
        inputParams.enabledAttribFlags = 0b11u;
        inputParams.enabledBindingFlags = 0b1u;
        inputParams.attributes[0].binding = 0u;
        inputParams.attributes[0].format = asset::EF_R32G32B32_SFLOAT;
        inputParams.attributes[0].relativeOffset = 0u;
        inputParams.bindings[0].stride = sizeof(float) * 3;
        inputParams.bindings[0].inputRate = asset::EVIR_PER_VERTEX;

        video::IGPUSpecializedShader* shaders[2] = { linevshader.get(),linefshader.get() };
        asset::SBlendParams blendParams; // defaults are fine
        asset::SRasterizationParams rasterParams;
        rasterParams.polygonMode = asset::EPM_LINE;

        drawIndirect_pipeline = driver->createRenderpassIndependentPipeline(nullptr, std::move(linePipelineLayout), shaders, shaders + sizeof(shaders) / sizeof(void*), inputParams, blendParams, assemblyParams, rasterParams);
    }

    // prepare global descriptor set layout
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
        ds0layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(b, b+3);
    }
    auto gpuds0layout = driver->getGPUObjectsFromAssets(&ds0layout,&ds0layout+1)->front();

    // and the actual descriptor set
    core::smart_refctd_ptr<video::IGPUDescriptorSet> gpuds[4] = {};
    gpuds[0] = driver->createDescriptorSet(smart_refctd_ptr(gpuds0layout));
    {
        video::IGPUDescriptorSet::SWriteDescriptorSet write[3];
        video::IGPUDescriptorSet::SDescriptorInfo info[3];
        write[0].arrayElement = 0;
        write[0].binding = 0u;
        write[0].count = 1u;
        write[0].descriptorType = asset::EDT_STORAGE_BUFFER;
        write[0].dstSet = gpuds[0].get();
        write[0].info = info;
        info[0].desc = lineCountBuffer;
        info[0].buffer.offset = 0;
        info[0].buffer.size = lineCountBuffer->getSize();

        write[1].arrayElement = 0;
        write[1].binding = 1u;
        write[1].count = 1u;
        write[1].descriptorType = asset::EDT_STORAGE_BUFFER;
        write[1].dstSet = gpuds[0].get();
        write[1].info = info+1;
        info[1].desc = linesBuffer;
        info[1].buffer.offset = 0;
        info[1].buffer.size = linesBuffer->getSize();

        write[2].arrayElement = 0;
        write[2].binding = 2u;
        write[2].count = 1u;
        write[2].descriptorType = asset::EDT_UNIFORM_BUFFER;
        write[2].dstSet = gpuds[0].get();
        write[2].info = info+2;
        info[2].desc = uniformLinesSettingsBuffer;
        info[2].buffer.offset = 0;
        info[2].buffer.size = uniformLinesSettingsBuffer->getSize();
        driver->updateDescriptorSets(3u, write, 0u, nullptr);
    }

    // finally the vertex input bindings for line shader
    asset::SBufferBinding<video::IGPUBuffer> bufferBinding[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT] = {};
    bufferBinding[0].offset = 0;
    bufferBinding[0].buffer = linesBuffer;


    // set up compute shader to clamp line count and bind it persistently
    {
        auto layout = driver->createPipelineLayout(nullptr, nullptr, std::move(gpuds0layout), nullptr, nullptr, nullptr);
        core::smart_refctd_ptr<video::IGPUSpecializedShader> compShader;
        {
            asset::IAssetLoader::SAssetLoadParams lp;
            auto cs_bundle = am->getAsset("../compute.comp", lp);
            auto cs = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*cs_bundle.getContents().begin());

            auto cs_rawptr = cs.get();
            compShader = driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
        }
        auto compPipeline = driver->createComputePipeline(nullptr, std::move(layout), std::move(compShader));
        driver->bindComputePipeline(compPipeline.get());
        driver->bindDescriptorSets(video::EPBP_COMPUTE, compPipeline->getLayout(), 0u, 1u, &gpuds[0].get(), nullptr);
    }


    //
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
            if (meshes_bundle.getContents().empty())
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
            if (meshes_bundle.getContents().empty())
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

    // process mesh slightly
    auto mesh = meshes_bundle.getContents().begin()[0];
    auto mesh_raw = static_cast<asset::ICPUMesh*>(mesh.get());

    uint32_t triangleCount;
    if (!asset::IMeshManipulator::getPolyCount(triangleCount, mesh_raw))
        assert(false);
  
    const auto meta = meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>();
    const asset::CMTLMetadata::CRenderpassIndependentPipeline* pipelineMetadata = nullptr;
    core::map<const asset::ICPURenderpassIndependentPipeline*,core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>> modifiedPipelines;
    for (auto mb : mesh_raw->getMeshBuffers())
    {
        auto* pipeline = mb->getPipeline();
        auto found = modifiedPipelines.find(pipeline);
        if (found==modifiedPipelines.end())
        {
            pipelineMetadata = static_cast<const asset::CMTLMetadata::CRenderpassIndependentPipeline*>(meta->getAssetSpecificMetadata(pipeline));

            // new pipeline to modify, copy the pipeline
            auto pipeline_cp = core::smart_refctd_ptr_static_cast<asset::ICPURenderpassIndependentPipeline>(pipeline->clone(1u));

            // insert a geometry shader into the pipeline
            pipeline_cp->setShaderAtIndex(asset::ICPURenderpassIndependentPipeline::ESSI_GEOMETRY_SHADER_IX,(pipelineMetadata->usesShaderWithUVs() ? geomShaderUV:geomShaderNOUV).get());

            // add descriptor set layout with one that has an SSBO and UBO
            auto* layout = pipeline_cp->getLayout();
            layout->setDescriptorSetLayout(0, core::smart_refctd_ptr(ds0layout));

            // cache the result
            found = modifiedPipelines.emplace(pipeline,std::move(pipeline_cp)).first;
        }
        mb->setPipeline(core::smart_refctd_ptr(found->second));
    }
    assert(pipelineMetadata);


    //we can safely assume that all meshbuffers within mesh loaded from OBJ has same DS1 layout (used for camera-specific data)
    //so we can create just one DS
    asset::ICPUDescriptorSetLayout* ds1layout = mesh_raw->getMeshBuffers().begin()[0]->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
    uint32_t ds1UboBinding = 0u;
    for (const auto& bnd : ds1layout->getBindings())
        if (bnd.type==asset::EDT_UNIFORM_BUFFER)
        {
            ds1UboBinding = bnd.binding;
            break;
        }

    size_t neededDS1UBOsz = 0ull;
    {
        for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
            if (shdrIn.descriptorSection.type==asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set==1u && shdrIn.descriptorSection.uniformBufferObject.binding==ds1UboBinding)
                neededDS1UBOsz = std::max<size_t>(neededDS1UBOsz, shdrIn.descriptorSection.uniformBufferObject.relByteoffset+shdrIn.descriptorSection.uniformBufferObject.bytesize);
    }


    auto gpuubo = driver->createDeviceLocalGPUBufferOnDedMem(neededDS1UBOsz);

    auto gpuds1layout = driver->getGPUObjectsFromAssets(&ds1layout,&ds1layout+1)->front();
    gpuds[1] = driver->createDescriptorSet(std::move(gpuds1layout));
    {
        video::IGPUDescriptorSet::SWriteDescriptorSet write;
        write.dstSet = gpuds[1].get();
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

    // finally get our GPU mesh
    auto gpumesh = driver->getGPUObjectsFromAssets(&mesh_raw, &mesh_raw+1)->front();

    auto startingCameraSpeed = gpumesh->getBoundingBox().getExtent().getLength() * 0.0005f;
    float cameraSpeed = 1;

    auto boundingBoxSize = gpumesh->getBoundingBox().getExtent().getLength();
	//! we want to move around the scene and view it from different angles
    scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0, 100.0f, startingCameraSpeed);

	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(1.f);
	camera->setFarValue(5000.0f);

    smgr->setActiveCamera(camera);
	uint64_t lastFPSTime = 0;
    GlobalUniforms globalUniforms = { 0u,10.f };
	while(device->run() && receiver.keepOpen())
	{
        if (receiver.resetCameraPosition())
        {
            camera->setPosition(core::vector3df(-4, 0, 0));
            camera->setTarget(core::vector3df(0, 0, 0));
        }
        if (cameraSpeed != receiver.getCameraSpeed())
        {
            cameraSpeed = receiver.getCameraSpeed();
            auto pos = camera->getPosition();
            auto rot = camera->getRotation();
            smgr->addToDeletionQueue(camera);
            camera = smgr->addCameraSceneNodeFPS(0, 100.0f, startingCameraSpeed * cameraSpeed);
            smgr->setActiveCamera(camera);
            camera->setPosition(pos);
            camera->setRotation(rot);
            camera->setNearValue(1.f);
            camera->setFarValue(5000.0f);
        }


        driver->beginScene(true, true, video::SColor(255,128,128,128) );

        // always update cause of mdiIndex
        globalUniforms.intersectionPlaneSpacing = receiver.getSpacing();
        driver->updateBufferRangeViaStagingBuffer(uniformLinesSettingsBuffer.get(), 0, sizeof(globalUniforms), &globalUniforms);


        //! This animates (moves) the camera and sets the transforms
		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();


        core::vector<uint8_t> uboData(gpuubo->getSize());
        for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
        {
            if (shdrIn.descriptorSection.type==asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set==1u && shdrIn.descriptorSection.uniformBufferObject.binding==ds1UboBinding)
            {
                switch (shdrIn.type)
                {
                case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_PROJ:
                {
                    core::matrix4SIMD mvp = camera->getConcatenatedMatrix();
                    memcpy(uboData.data()+shdrIn.descriptorSection.uniformBufferObject.relByteoffset, mvp.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                }
                break;
                case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW:
                {
                    core::matrix3x4SIMD MV = camera->getViewMatrix();
                    memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, MV.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                }
                break;
                case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE:
                {
                    core::matrix3x4SIMD MV = camera->getViewMatrix();
                    memcpy(uboData.data()+shdrIn.descriptorSection.uniformBufferObject.relByteoffset, MV.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                }
                break;
                }
            }
        }       
        driver->updateBufferRangeViaStagingBuffer(gpuubo.get(), 0ull, gpuubo->getSize(), uboData.data());

        // draw the meshbuffers and compute lines
        for (auto gpumb : gpumesh->getMeshBuffers())
        {
            const video::IGPURenderpassIndependentPipeline* pipeline = gpumb->getPipeline();  
            const video::IGPUDescriptorSet* ds3 = gpumb->getAttachedDescriptorSet();

            driver->bindGraphicsPipeline(pipeline);
            driver->bindDescriptorSets(video::EPBP_GRAPHICS, pipeline->getLayout(), 0u, 2u, reinterpret_cast<const video::IGPUDescriptorSet**>(gpuds), nullptr);
            const video::IGPUDescriptorSet* gpuds3_ptr = gpumb->getAttachedDescriptorSet();
            if (gpuds3_ptr)
                driver->bindDescriptorSets(video::EPBP_GRAPHICS, pipeline->getLayout(), 3u, 1u, &gpuds3_ptr, nullptr);
            driver->pushConstants(pipeline->getLayout(), video::IGPUSpecializedShader::ESS_FRAGMENT|video::IGPUSpecializedShader::ESS_VERTEX, 0u, gpumb->MAX_PUSH_CONSTANT_BYTESIZE, gpumb->getPushConstantsDataPtr());
            driver->drawMeshBuffer(gpumb);
        }

        // emit "memory barrier" of type GL_SHADER_STORAGE_BARRIER_BIT after the entire scene finishes drawing because we'll use the outputs as SSBOs
        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        driver->dispatch(1u,1u,1u);
        // emit another memory barrier telling that we'll use the previously async written SSBO as MDI source, vertex data source, possibly to download and finally we'll re-use in the geometry shader as a clear buffer (cleared to 0 lines)
        const bool willDownloadThisFrame = receiver.doBufferSave();
        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_COMMAND_BARRIER_BIT|GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT|(willDownloadThisFrame ? GL_BUFFER_UPDATE_BARRIER_BIT:0u)|GL_SHADER_STORAGE_BARRIER_BIT);

        driver->bindGraphicsPipeline(drawIndirect_pipeline.get());
        driver->pushConstants(drawIndirect_pipeline->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), camera->getConcatenatedMatrix().pointer());
        //invoke drawIndirect and use linesBuffer
        driver->drawArraysIndirect(bufferBinding, asset::EPT_LINE_LIST, lineCountBuffer.get(), sizeof(asset::DrawArraysIndirectCommand_t)* globalUniforms.mdiIndex, 1u, sizeof(asset::DrawArraysIndirectCommand_t));
		driver->endScene();
        
        if (willDownloadThisFrame)
            SaveBufferToCSV(driver,globalUniforms.mdiIndex,lineCountBuffer.get(),linesBuffer.get());
        globalUniforms.mdiIndex ^= 0x1u;

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
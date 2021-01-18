// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_SKINNED_MESH_SCENE_NODE_H_INCLUDED__
#define __NBL_C_SKINNED_MESH_SCENE_NODE_H_INCLUDED__

#include "ISkinnedMeshSceneNode.h"
#include "CSkinningStateManager.h"


namespace nbl
{
namespace scene
{

	class CSkinnedMeshSceneNode : public ISkinnedMeshSceneNode
	{
            void buildFrameNr(const uint32_t& deltaTimeMs);
            core::smart_refctd_ptr<video::IGPUSkinnedMesh> mesh;
            CSkinningStateManager* boneStateManager;

            //core::vector<video::SGPUMaterial> Materials;
            core::aabbox3d<float> Box;
            IAnimationEndCallBack<ISkinnedMeshSceneNode>* LoopCallBack;

            float FramesPerSecond;
            float CurrentFrameNr;
            float StartFrame,EndFrame;
            float desiredUpdateFrequency;
            uint32_t LastTimeMs;
            bool Looping;

            int32_t PassCount;
        protected:
            //! Destructor
            virtual ~CSkinnedMeshSceneNode()
            {
                if (boneStateManager)
                    boneStateManager->drop();

                if (LoopCallBack)
                    LoopCallBack->drop();
            }

        public:

            //! Constructor
            CSkinnedMeshSceneNode(core::smart_refctd_ptr<video::IGPUSkinnedMesh>&& _mesh, const ISkinningStateManager::E_BONE_UPDATE_MODE& boneControlMode, IDummyTransformationSceneNode* parent, ISceneManager* mgr,	int32_t id,
                    const core::vector3df& position = core::vector3df(0.f), const core::vector3df& rotation = core::vector3df(0.f),
                    const core::vector3df& scale = core::vector3df(1.f))
                : ISkinnedMeshSceneNode(parent, mgr, id, position, rotation, scale), mesh(), boneStateManager(NULL),
                LoopCallBack(NULL), FramesPerSecond(0.025f), desiredUpdateFrequency(1000.f/120.f), StartFrame(0.f), EndFrame(0.f), CurrentFrameNr(0.f), LastTimeMs(0),
                Looping(true), PassCount(0)
            {
                #ifdef _NBL_DEBUG
                setDebugName("CSkinnedMeshSceneNode");
                #endif
                setMesh(std::move(_mesh),boneControlMode);
            }

            //!
            virtual bool supportsDriverFence() const {return true;}

            const void* getRawBoneData() {return boneStateManager->getRawBoneData();}

            virtual video::IGPUBufferView* getBonePoseTBO() const override
            {
                if (!boneStateManager)
                    return nullptr;

                return boneStateManager->getBoneDataTBO();
            }

            //! Sets the current frame number.
            virtual void setCurrentFrame(const float& frame)
            {
                // if you pass an out of range value, we just clamp it
                CurrentFrameNr = core::clamp ( frame, (float)StartFrame, (float)EndFrame );
            }

            //! Sets the frame numbers between the animation is looped.
            virtual bool setFrameLoop(const float& begin, const float& end);

            //! Sets the speed with which the animation is played.
            virtual void setAnimationSpeed(const float&  framesPerSecond) {FramesPerSecond = framesPerSecond*0.001f;}

            //! Gets the speed with which the animation is played.
            virtual float getAnimationSpeed() const {return FramesPerSecond*1000.f;}

            //! only for EBUM_NONE and EBUM_READ, it dictates what is the actual frequency we want to bother updating the mesh
            //! because we don't want to waste CPU time if we can tolerate the bones updating at 120Hz or similar
            virtual void setDesiredUpdateFrequency(const float& hertz) {desiredUpdateFrequency = 1000.f/hertz;}

            virtual float getDesiredUpdateFrequency() const {return 1000.f/desiredUpdateFrequency;}

            virtual size_t getBoneCount() const { return boneStateManager->getBoneCount(); }

            //! frame
            virtual void OnRegisterSceneNode();

            //! OnAnimate() is called just before rendering the whole scene.
            virtual void OnAnimate(uint32_t timeMs);

            //! renders the node.
            virtual void render();

            virtual void setBoundingBox(const core::aabbox3d<float>& bbox) {Box = bbox;}
            //! returns the axis aligned bounding box of this node
            virtual const core::aabbox3d<float>& getBoundingBox() {return Box;}

            //! Get a pointer to a joint in the mesh.
            virtual ISkinningStateManager::IBoneSceneNode* getJointNode(const size_t& jointID)
            {
                if (!mesh || !boneStateManager || boneStateManager->getBoneUpdateMode()==ISkinningStateManager::EBUM_NONE || jointID>=mesh->getBoneReferenceHierarchy()->getBoneCount())
                    return NULL;

                ISkinningStateManager::IBoneSceneNode* tmpBone = boneStateManager->getBone(jointID,0);
                if (tmpBone)
                    return tmpBone;

                assert(boneStateManager->getBoneUpdateMode()==ISkinningStateManager::EBUM_READ);
                boneStateManager->createBones(0);
                return boneStateManager->getBone(jointID,0);
            }


            //! Returns the currently displayed frame number.
            virtual float getFrameNr() const {return CurrentFrameNr;}
            //! Returns the current start frame number.
            virtual float getStartFrame() const {return StartFrame;}
            //! Returns the current end frame number.
            virtual float getEndFrame() const {return EndFrame;}

            //! Sets looping mode which is on by default.
            /** If set to false, animations will not be played looped. */
            virtual void setLoopMode(bool playAnimationLooped) {Looping = playAnimationLooped;}

            //! returns the current loop mode
            /** When true the animations are played looped */
            virtual bool getLoopMode() const {return Looping;}

            //! Sets a callback interface which will be called if an animation playback has ended.
            /** Set this to 0 to disable the callback again.
            Please note that this will only be called when in non looped
            mode, see ISkinnedMeshSceneNode::setLoopMode(). */
            virtual void setAnimationEndCallback(IAnimationEndCallBack<ISkinnedMeshSceneNode>* callback=0);

            //! Sets a new mesh
            virtual void setMesh(core::smart_refctd_ptr<video::IGPUSkinnedMesh>&& inMesh, const ISkinningStateManager::E_BONE_UPDATE_MODE& boneControl=ISkinningStateManager::EBUM_NONE);

            //! Returns the current mesh
            virtual video::IGPUSkinnedMesh* getMesh(void) override {return mesh.get();}
            virtual const video::IGPUSkinnedMesh* getMesh(void) const override {return mesh.get();}

            //! animates the joints in the mesh based on the current frame.
            /** Also takes in to account transitions. */
            virtual void animateJoints()
            {
                if (!boneStateManager)
                    return;

                updateAbsolutePosition();
                boneStateManager->setFrame(getFrameNr(),0);

                boneStateManager->performBoning();
            }
	};


} // end namespace scene
} // end namespace nbl

#endif



// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_DUMMY_TRANSFORMATION_SCENE_NODE_H_INCLUDED__
#define __I_DUMMY_TRANSFORMATION_SCENE_NODE_H_INCLUDED__

#include "IReferenceCounted.h"
#include "ISceneNodeAnimator.h"
#include "irrList.h"
#include "matrix4x3.h"
#include "ESceneNodeTypes.h"

namespace irr
{
namespace scene
{

class ISceneManager;
class ISceneNodeAnimator;
class IDummyTransformationSceneNode;



	//! Typedef for list of scene nodes
	typedef core::list<IDummyTransformationSceneNode*> IDummyTransformationSceneNodeList;
	//! Typedef for list of scene node animators
	typedef core::list<ISceneNodeAnimator*> ISceneNodeAnimatorList;

//! Dummy scene node for adding additional transformations to the scene graph.
/** This scene node does not render itself, and does not respond to set/getPosition,
set/getRotation and set/getScale. Its just a simple scene node that takes a
matrix as relative transformation, making it possible to insert any transformation
anywhere into the scene graph.
This scene node is for example used by the IAnimatedMeshSceneNode for emulating
joint scene nodes when playing skeletal animations.
*/
class IDummyTransformationSceneNode : public IReferenceCounted
{
    protected:
        uint64_t lastTimeRelativeTransRead[5];

        uint64_t relativeTransChanged;
        bool relativeTransNeedsUpdate;
    public:

        //! Constructor
        IDummyTransformationSceneNode(IDummyTransformationSceneNode* parent,
				const core::vector3df& position = core::vector3df(0,0,0),
				const core::vector3df& rotation = core::vector3df(0,0,0),
				const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f)) :
                RelativeTranslation(position), RelativeRotation(rotation), RelativeScale(scale),
				Parent(0),  relativeTransChanged(1), relativeTransNeedsUpdate(true)
        {
            memset(lastTimeRelativeTransRead,0,sizeof(uint64_t)*5);

			if (parent)
				parent->addChild(this);

			updateAbsolutePosition();
        }

        virtual ~IDummyTransformationSceneNode()
        {
            removeAll();

			// delete all animators
			ISceneNodeAnimatorList::Iterator ait = Animators.begin();
			for (; ait != Animators.end(); ++ait)
				(*ait)->drop();
        }

        virtual const bool isISceneNode() const {return false;}

        //! Returns a reference to the current relative transformation matrix.
        /** This is the matrix, this scene node uses instead of scale, translation
        and rotation. */
        inline virtual const core::matrix4x3& getRelativeTransformationMatrix()
        {
            if (relativeTransNeedsUpdate)
            {
                RelativeTransformation.setRotationDegrees(RelativeRotation);
                RelativeTransformation.setTranslation(RelativeTranslation);
                //
                RelativeTransformation(0,0) *= RelativeScale.X;
                RelativeTransformation(1,0) *= RelativeScale.X;
                RelativeTransformation(2,0) *= RelativeScale.X;
                RelativeTransformation(0,1) *= RelativeScale.Y;
                RelativeTransformation(1,1) *= RelativeScale.Y;
                RelativeTransformation(2,1) *= RelativeScale.Y;
                RelativeTransformation(0,2) *= RelativeScale.Z;
                RelativeTransformation(1,2) *= RelativeScale.Z;
                RelativeTransformation(2,2) *= RelativeScale.Z;
                //
                relativeTransChanged++;
                relativeTransNeedsUpdate = false;
            }
            return RelativeTransformation;
        }

        //!
        inline void setRelativeTransformationMatrix(const core::matrix4x3& tform)
        {
            RelativeTransformation = tform;
            relativeTransChanged++;
            relativeTransNeedsUpdate = false;
        }

        inline const uint64_t& getRelativeTransChangedHint() const {return relativeTransChanged;}

        inline const uint64_t& getAbsoluteTransformLastRecomputeHint() const {return lastTimeRelativeTransRead[3];}

        inline const core::vector3df& getScale()
        {
            if (lastTimeRelativeTransRead[0]<relativeTransChanged)
            {
                const core::matrix4x3& rel = getRelativeTransformationMatrix();
                RelativeScale = rel.getScale();
                lastTimeRelativeTransRead[0] = relativeTransChanged;
            }
            return RelativeScale;
        }

        inline void setScale(const core::vector3df& scale)
        {
            RelativeScale = scale;
            relativeTransNeedsUpdate = true;
        }

        inline const core::vector3df& getRotation()
        {
            if (lastTimeRelativeTransRead[1]<relativeTransChanged)
            {
                const core::matrix4x3& rel = getRelativeTransformationMatrix();
                RelativeRotation = rel.getRotationDegrees();
                lastTimeRelativeTransRead[1] = relativeTransChanged;
            }
            return RelativeRotation;
        }

        inline void setRotation(const core::vector3df& rotation)
        {
            RelativeRotation = rotation;
            relativeTransNeedsUpdate = true;
        }

        inline const core::vector3df& getPosition()
        {
            if (lastTimeRelativeTransRead[2]<relativeTransChanged)
            {
                const core::matrix4x3& rel = getRelativeTransformationMatrix();
                RelativeTranslation = rel.getTranslation();
                lastTimeRelativeTransRead[2] = relativeTransChanged;
            }
            return RelativeTranslation;
        }

        inline void setPosition(const core::vector3df& newpos)
        {
            RelativeTranslation = newpos;
            relativeTransNeedsUpdate = true;
        }


        inline virtual bool needsAbsoluteTransformRecompute() const
        {
            if (relativeTransNeedsUpdate||lastTimeRelativeTransRead[3]<relativeTransChanged)
                return true;

            if (Parent)
                return lastTimeRelativeTransRead[4]<Parent->getAbsoluteTransformLastRecomputeHint();

            return false;
        }

        inline virtual size_t needsDeepAbsoluteTransformRecompute() const
        {
            const IDummyTransformationSceneNode* parentStack[1024];
            parentStack[0] = this;
            size_t stackSize=0;

            while (parentStack[stackSize])
            {
                parentStack[++stackSize] = parentStack[stackSize];
                if (stackSize>=1024)
                    break;
            }

            size_t maxStackSize = stackSize-1;
            while (--stackSize)
            {
                if (parentStack[stackSize]->relativeTransNeedsUpdate||parentStack[stackSize]->lastTimeRelativeTransRead[3]<parentStack[stackSize]->relativeTransChanged)
                    return stackSize;

                if (stackSize<maxStackSize)
                {
                    if (parentStack[stackSize]->lastTimeRelativeTransRead[4]<parentStack[stackSize+1]->getAbsoluteTransformLastRecomputeHint())
                        return stackSize;
                }
            }

            return 0xdeadbeefu;
        }

        inline const core::matrix4x3& getAbsoluteTransformation()
        {
            return AbsoluteTransformation;
        }

		//! Gets the absolute position of the node in world coordinates.
		/** If you want the position of the node relative to its parent,
		use getPosition() instead.
		NOTE: For speed reasons the absolute position is not
		automatically recalculated on each change of the relative
		position or by a position change of an parent. Instead the
		update usually happens once per frame in OnAnimate. You can enforce
		an update with updateAbsolutePosition().
		\return The current absolute position of the scene node (updated on last call of updateAbsolutePosition). */
		inline core::vector3df getAbsolutePosition() const
		{
			return AbsoluteTransformation.getTranslation();
		}

		//! Updates the absolute position based on the relative and the parents position
		/** Note: This does not recursively update the parents absolute positions, so if you have a deeper
			hierarchy you might want to update the parents first.*/
		inline virtual void updateAbsolutePosition()
		{
            bool recompute = relativeTransNeedsUpdate||lastTimeRelativeTransRead[3]<relativeTransChanged;

            if (Parent)
            {
                uint64_t parentAbsoluteHint = Parent->getAbsoluteTransformLastRecomputeHint();
                if (lastTimeRelativeTransRead[4] < parentAbsoluteHint)
                {
                    lastTimeRelativeTransRead[4] = parentAbsoluteHint;
                    recompute = true;
                }

                // recompute if local transform has changed
                if (recompute)
                {
                    const core::matrix4x3& rel = getRelativeTransformationMatrix();
                    AbsoluteTransformation = concatenateBFollowedByA(Parent->getAbsoluteTransformation(),rel);
                    lastTimeRelativeTransRead[3] = relativeTransChanged;
                }
            }
            else if (recompute)
            {
                AbsoluteTransformation = getRelativeTransformationMatrix();
                lastTimeRelativeTransRead[3] = relativeTransChanged;
            }
		}



		//! Returns a const reference to the list of all children.
		/** \return The list of all children of this node. */
		inline const core::list<IDummyTransformationSceneNode*>& getChildren() const
		{
			return Children;
		}


		//! Changes the parent of the scene node.
		/** \param newParent The new parent to be used. */
		virtual void setParent(IDummyTransformationSceneNode* newParent)
		{
		    if (newParent==Parent)
                return;

			grab();
			remove();

			Parent = newParent;

			if (Parent)
            {
				Parent->addChild(this);
				lastTimeRelativeTransRead[4] = 0;
            }

			drop();
		}


		//! Returns the parent of this scene node
		/** \return A pointer to the parent. */
		inline IDummyTransformationSceneNode* getParent() const
		{
			return Parent;
		}

		//! Adds a child to this scene node.
		/** If the scene node already has a parent it is first removed
		from the other parent.
		\param child A pointer to the new child. */
		virtual void addChild(IDummyTransformationSceneNode* child)
		{
			if (child && (child != this))
			{
				child->grab();
				child->remove(); // remove from old parent
				Children.push_back(child);
				child->Parent = this;
				child->lastTimeRelativeTransRead[4] = 0;
			}
		}


		//! Removes a child from this scene node.
		/** If found in the children list, the child pointer is also
		dropped and might be deleted if no other grab exists.
		\param child A pointer to the child which shall be removed.
		\return True if the child was removed, and false if not,
		e.g. because it couldn't be found in the children list. */
		virtual bool removeChild(IDummyTransformationSceneNode* child)
		{
			IDummyTransformationSceneNodeList::Iterator it = Children.begin();
			for (; it != Children.end(); ++it)
				if ((*it) == child)
				{
					(*it)->Parent = 0;
					(*it)->drop();
					Children.erase(it);
					return true;
				}

			_IRR_IMPLEMENT_MANAGED_MARSHALLING_BUGFIX;
			return false;
		}


		//! Removes all children of this scene node
		/** The scene nodes found in the children list are also dropped
		and might be deleted if no other grab exists on them.
		*/
		virtual void removeAll()
		{
			IDummyTransformationSceneNodeList::Iterator it = Children.begin();
			for (; it != Children.end(); ++it)
			{
				(*it)->Parent = 0;
				(*it)->drop();
			}

			Children.clear();
		}


		//! Removes this scene node from the scene
		/** If no other grab exists for this node, it will be deleted.
		*/
		virtual void remove()
		{
			if (Parent)
				Parent->removeChild(this);
		}


		//! Adds an animator which should animate this node.
		/** \param animator A pointer to the new animator. */
		virtual void addAnimator(ISceneNodeAnimator* animator)
		{
			if (animator)
			{
				Animators.push_back(animator);
				animator->grab();
			}
		}


		//! Get a list of all scene node animators.
		/** \return The list of animators attached to this node. */
		const core::list<ISceneNodeAnimator*>& getAnimators() const
		{
			return Animators;
		}


		//! Removes an animator from this scene node.
		/** If the animator is found, it is also dropped and might be
		deleted if not other grab exists for it.
		\param animator A pointer to the animator to be deleted. */
		virtual void removeAnimator(ISceneNodeAnimator* animator)
		{
			ISceneNodeAnimatorList::Iterator it = Animators.begin();
			for (; it != Animators.end(); ++it)
			{
				if ((*it) == animator)
				{
					(*it)->drop();
					Animators.erase(it);
					return;
				}
			}
		}


		//! Removes all animators from this scene node.
		/** The animators might also be deleted if no other grab exists
		for them. */
		virtual void removeAnimators()
		{
			ISceneNodeAnimatorList::Iterator it = Animators.begin();
			for (; it != Animators.end(); ++it)
				(*it)->drop();

			Animators.clear();
		}

        //! Returns type of the scene node
        virtual ESCENE_NODE_TYPE getType() const { return ESNT_DUMMY_TRANSFORMATION; }

        //! Creates a clone of this scene node and its children.
        virtual IDummyTransformationSceneNode* clone(IDummyTransformationSceneNode* newParent=0, ISceneManager* newManager=0)
        {
            if (!newParent)
                newParent = Parent;

            IDummyTransformationSceneNode* nb = new IDummyTransformationSceneNode(newParent);

            nb->cloneMembers(this, newManager);
            nb->setRelativeTransformationMatrix(RelativeTransformation);

            if ( newParent )
                nb->drop();
            return nb;
        }

	protected:
		//! Pointer to the parent
		IDummyTransformationSceneNode* Parent;

		//! List of all children of this node
		core::list<IDummyTransformationSceneNode*> Children;

		//! List of all animator nodes
		core::list<ISceneNodeAnimator*> Animators;

		//! Absolute transformation of the node.
		core::matrix4x3 AbsoluteTransformation;

        //! Relative transformation of the node.
        core::matrix4x3 RelativeTransformation;

		//! Relative translation of the scene node.
		core::vector3df RelativeTranslation;

		//! Relative rotation of the scene node.
		core::vector3df RelativeRotation;

		//! Relative scale of the scene node.
		core::vector3df RelativeScale;

		//! A clone function for the IDummy... members.
		/** This method can be used by clone() implementations of
		derived classes
		\param toCopyFrom The node from which the values are copied */
		virtual void cloneMembers(IDummyTransformationSceneNode* toCopyFrom, ISceneManager* newManager)
		{
			AbsoluteTransformation = toCopyFrom->AbsoluteTransformation;
			RelativeTranslation = toCopyFrom->RelativeTranslation;
			RelativeTranslation = toCopyFrom->RelativeTranslation;
			RelativeRotation = toCopyFrom->RelativeRotation;
			RelativeScale = toCopyFrom->RelativeScale;

			// clone children
			IDummyTransformationSceneNodeList::Iterator it = toCopyFrom->Children.begin();
			for (; it != toCopyFrom->Children.end(); ++it)
				(*it)->clone(this, newManager);

			// clone animators
			ISceneNodeAnimatorList::Iterator ait = toCopyFrom->Animators.begin();
			for (; ait != toCopyFrom->Animators.end(); ++ait)
			{
				ISceneNodeAnimator* anim = (*ait)->createClone(this, newManager);
				if (anim)
				{
					addAnimator(anim);
					anim->drop();
				}
			}
		}
};

} // end namespace scene
} // end namespace irr


#endif


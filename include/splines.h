// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_SPLINES_H_INCLUDED__
#define __NBL_SPLINES_H_INCLUDED__

#include "BuildConfigOptions.h"
#include <cmath>       /* sqrt */
#include "nbl/core/math/glslFunctions.tcc"

#include <vector>


namespace nbl
{
namespace core
{

// TODO @Przemog
// TODO: Refactor the base into an interpolator so we can use it for rotations too (in `core`)
    // control points
    // looping, finishing or pingponging (repeat, clamp or mirror)
    // do what `getPos_fromParameter` does but call it `getValue`
    // do what `getUnnormDirection_fromParameter` does but call it `getDerivativeAndTangent`
// TODO: Refactor the other functionality Follow Curve Animator (in `scene`)
    // need `getSplineLength` as `getCurveLength`
    // info about segments
    // info about arclength being precise (for many splines its not because arclength integrals are not fun)
    // for the Follow Spine Animators (derived from interpolator)
        // implement `_fromParameter` as simple passthroughs
        // remember that `getPos` and `getUnnormDirection` need to be implemented in arclength (so the object can follow at constant speed)
// TODO: Implement a FollowCircle or FollowEllipsoid derived from Follow Curve Animator
    // 
// TODO: Refactor the `BlockChange` stuff into an ext::baw::Animators
    // all the methods with `BlockChange` in the name
class ISpline// : public AllocationOverrideDefault
{
    public:
        virtual ~ISpline() {}

        // TODO: add pingpong option
        virtual bool        isLooping() const {return isLoop;}
        virtual size_t      getSegmentCount() const = 0;
        virtual float       getSplineLength() const = 0;

        //
        virtual void        getSegmentLengths(float* outSegLens) const = 0;
        virtual float       getSegmentLength(const uint32_t& segmentID) const = 0;
        virtual float       getSegmentParameterRange(const uint32_t& segmentID) const = 0;

        //get position
        //this function returns the id of the segment you might have moved into
        virtual uint32_t    getPos(vectorSIMDf& pos, float& distanceAlongSeg, const uint32_t& segmentID, float* paramHint=NULL, const float& accuracyThresh=0.00390625f) const = 0;
        virtual bool        getPos_fromParameter(vectorSIMDf& pos, const uint32_t& segmentID, const float& parameter) const = 0;

        //to get direction to look in
        virtual bool        getUnnormDirection(vectorSIMDf& tan, const uint32_t& segmentID, const float& distanceAlongSeg) const = 0;
        virtual bool        getUnnormDirection_fromParameter(vectorSIMDf& tan, const uint32_t& segmentID, const float& parameter) const = 0;

        //baw specific
        virtual bool      canGiveParameterUntilBlockChange() const {return false;}
        // pass in current position
        virtual float           getParameterUntilBlockChange(const uint32_t& segmentID, const float& param) = 0;
        virtual core::vector<float>   getBlockChangesInSegment(const uint32_t& segmentID, float startParam=0.f) = 0;

        //is the distance and parameter the same?
        virtual bool            isArcLengthPrecise() const = 0;
        ///virtual float           parameterToDistance(const float& param) const = 0;
        ///virtual float           distanceToParameter(const float& dist) const = 0;
    protected:
        ISpline(bool loop) : isLoop(loop) {}

        const bool isLoop;
    private:
};


class CLinearSpline : public ISpline
{
    public:
        CLinearSpline(vectorSIMDf* controlPoints, const size_t& count, const bool loop = false) : ISpline(loop)
        {
            //assert(count<0x80000000u && count);
            for (size_t i=1; i<count; i++)
                segments.push_back(Segment(controlPoints[i-1],controlPoints[i]));

            if (isLoop)
                segments.push_back(Segment(controlPoints[count-1],controlPoints[0]));
            finalize();
        }
        CLinearSpline(vectorSIMDf* controlPoints, float* customDistances, const size_t& count, const bool loop = false) : ISpline(loop)
        {
            //assert(count<0x80000000u);
            for (size_t i=1; i<count; i++)
                segments.push_back(Segment(controlPoints[i-1],controlPoints[i],customDistances[i-1]));

            if (isLoop)
                segments.push_back(Segment(controlPoints[count-1],controlPoints[0],customDistances[count-1]));
            finalize();
        }

        //
        virtual size_t      getSegmentCount() const
        {
            return segments.size();
        }
        virtual float       getSplineLength() const
        {
            return splineLen;
        }

        //
        virtual void        getSegmentLengths(float* outSegLens) const
        {
            for (size_t i=0; i<segments.size(); i++)
                outSegLens[i] = segments[i].length;
        }
        virtual float       getSegmentLength(const uint32_t& segmentID) const
        {
            //assert(segmentID<segments.size());
            return segments[segmentID].length;
        }
        virtual float       getSegmentParameterRange(const uint32_t& segmentID) const
        {
            return getSegmentLength(segmentID);
        }

        //get position
        //this function returns the id of the segment you might have moved into - 0xdeadbeefu is an error code
        virtual uint32_t    getPos(vectorSIMDf& pos, float& distanceAlongSeg, const uint32_t& segmentID, float* paramHint=NULL, const float& accuracyThresh=0.00390625f) const
        {
            if (distanceAlongSeg<0.f)
                return 0xdeadbeefu;

            uint32_t actualSeg;
            if (isLoop)
            {
                actualSeg = segmentID%segments.size();
                while (distanceAlongSeg>=segments[actualSeg].length)
                {
                    distanceAlongSeg -= segments[actualSeg].length;
                    actualSeg++;
                    if (actualSeg==segments.size())
                        actualSeg = 0;
                }
            }
            else
            {
                if (segmentID>=segments.size())
                    return 0xdeadbeefu;

                actualSeg = segmentID;
                while (distanceAlongSeg>=segments[actualSeg].length)
                {
                    distanceAlongSeg -= segments[actualSeg].length;
                    actualSeg++;
                    if (actualSeg==segments.size())
                        return 0xdeadbeefu;
                }
            }

            pos = segments[actualSeg].posHelper(distanceAlongSeg);
            if (paramHint)
                *paramHint = distanceAlongSeg;

            return actualSeg;
        }
        virtual bool        getPos_fromParameter(vectorSIMDf& pos, const uint32_t& segmentID, const float& parameter) const
        {
            if (segmentID>=segments.size()||parameter>segments[segmentID].length)
                return false;

            pos = segments[segmentID].posHelper(parameter);

            return true;
        }

        //to get direction to look in
        virtual bool        getUnnormDirection(vectorSIMDf& tan, const uint32_t& segmentID, const float& distanceAlongSeg) const
        {
            if (segmentID>=segments.size()||distanceAlongSeg>segments[segmentID].length)
                return false;

            tan = segments[segmentID].directionHelper(distanceAlongSeg);

            return true;
        }

        virtual bool        getUnnormDirection_fromParameter(vectorSIMDf& tan, const uint32_t& segmentID, const float& parameter) const
        {
            return getUnnormDirection(tan,segmentID,parameter);
        }

        //baw specific
        virtual bool      canGiveParameterUntilBlockChange() const {return true;}
        // pass in current position
        virtual float           getParameterUntilBlockChange(const uint32_t& segmentID, const float& param)
        {
            if (segmentID>=segments.size()||param>=segments[segmentID].length)
                return -1.f;

            return segments[segmentID].findNextBlockChange(param);
        }
        virtual core::vector<float>   getBlockChangesInSegment(const uint32_t& segmentID, float startParam=0.f)
        {
            core::vector<float> changes;
            if (segmentID>=segments.size())
                return changes;

            const Segment& seg = segments[segmentID];
            while (true)
            {
                float fnd = seg.findNextBlockChange(startParam);
                if (fnd<0.f)
                    return changes;

                startParam = fnd;
                reinterpret_cast<uint32_t&>(startParam)++;
                changes.push_back(fnd);
            }
        }


        virtual bool            isArcLengthPrecise() const {return true;}
        ///virtual float           parameterToDistance(const float& param) const {return param;}
        ///virtual float           distanceToParameter(const float& dist) const {return dist;}
    private:
        void finalize()
        {
            double lenDouble = 0;
            for (size_t i=0; i<segments.size(); i++)
            {
                lenDouble += segments[i].length;
            }
            splineLen = lenDouble;
        }


        struct Segment
        {
            Segment(const vectorSIMDf& startPt,const vectorSIMDf& endPt)
            {
                weights[0] = endPt-startPt;
                length = core::length(weights[0])[0];
                weights[0] /= length;

                weights[1] = startPt;
            }
            Segment(const vectorSIMDf& startPt,const vectorSIMDf& endPt, const float& customLen)
            {
                weights[0] = (endPt-startPt);
                length = customLen;
                weights[0] /= length;

                weights[1] = startPt;
            }

            inline vectorSIMDf posHelper(const float& distanceAlongSeg) const
            {
                return weights[0]*distanceAlongSeg+weights[1];
            }
            inline vectorSIMDf directionHelper(const float& distanceAlongSeg) const
            {
                return weights[0];
            }
            inline float findNextBlockChange(const float& param) const
            {
                vectorSIMDf startingNegFrac = posHelper(param);
                startingNegFrac = floor(startingNegFrac)-startingNegFrac;
                vectorSIMDf dir = directionHelper(param);
                float changes[3];
                for (uint32_t i=0; i<3; i++)
                    changes[i] = findChange(startingNegFrac.pointer[i],dir.pointer[i]);

                float smallest;
                if (reinterpret_cast<uint32_t*>(changes)[0]<=reinterpret_cast<uint32_t*>(changes)[1])
                {
                    if (reinterpret_cast<uint32_t*>(changes)[2]<=reinterpret_cast<uint32_t*>(changes)[0])
                        smallest = changes[2];
                    else
                        smallest = changes[0];
                }
                else if (reinterpret_cast<uint32_t*>(changes)[2]<=reinterpret_cast<uint32_t*>(changes)[1])
                {
                    smallest = changes[2];
                }
                else
                    smallest = changes[1];

                smallest += param;
                if (smallest<length)
                    return smallest;

                return -1.f;
            }
            inline float findChange(const float& currentNegFrac, const float& changePerParam) const
            {
                if (currentNegFrac==0.f||currentNegFrac==1.f)
                    return 0.f;

                if (changePerParam < -FLT_MIN)
                    return currentNegFrac/changePerParam;
                else if (changePerParam > FLT_MIN)
                    return (1.f+currentNegFrac)/changePerParam;

                return -1.f;
            }

            float length;
            vectorSIMDf weights[2];
        };

        core::vector<Segment> segments;
        float splineLen;
};


//! Loop code is wrong for now, unable to calculate A_0 so the gradients match all the way around the loop
class CQuadraticSpline : public ISpline
{
    public:
        CQuadraticSpline(vectorSIMDf* controlPoints, const size_t& count, const bool loop = false, float tightness = 1.1107f) : ISpline(loop)
        {
			assert(count>1ull);
            vectorSIMDf startGradient;
            if (isLoop)
				startGradient = controlPoints[1]-controlPoints[count-1ull];
            else
				startGradient = controlPoints[1]-controlPoints[0];
         
			float currentApproxLen = core::length(controlPoints[1]-controlPoints[0])[0];
			segments.push_back(Segment(controlPoints[0], controlPoints[1], startGradient, currentApproxLen, tightness));

            for (size_t i=2; i<count; i++)
            {
                vectorSIMDf startPt = controlPoints[i-1ull];
                vectorSIMDf endPt = controlPoints[i];
                currentApproxLen = core::length(endPt-startPt)[0];
                segments.push_back(Segment(segments.back(),startPt,endPt,currentApproxLen, tightness));
            }

			if (isLoop)
			{
				vectorSIMDf startPt = controlPoints[count-1ull];
				vectorSIMDf endPt = controlPoints[0];
				currentApproxLen = core::length(endPt-startPt)[0];
				segments.push_back(Segment(segments.back(),startPt,endPt,currentApproxLen, tightness));
			}

            finalize();
        }

        //
        virtual size_t      getSegmentCount() const
        {
            return segments.size();
        }
        virtual float       getSplineLength() const
        {
            return splineLen;
        }

        //
        virtual void        getSegmentLengths(float* outSegLens) const
        {
            for (size_t i=0; i<segments.size(); i++)
                outSegLens[i] = segments[i].length;
        }
        virtual float       getSegmentLength(const uint32_t& segmentID) const
        {
            //assert(segmentID<segments.size());
            return segments[segmentID].length;
        }
        virtual float       getSegmentParameterRange(const uint32_t& segmentID) const
        {
            return segments[segmentID].parameterLength;
        }

        //get position
        //this function returns the id of the segment you might have moved into - 0xdeadbeefu is an error code
        virtual uint32_t    getPos(vectorSIMDf& pos, float& distanceAlongSeg, const uint32_t& segmentID, float* paramHint=NULL, const float& accuracyThresh=0.00390625f) const
        {
            if (distanceAlongSeg<0.f)
                return 0xdeadbeefu;

            uint32_t actualSeg;
            if (isLoop)
            {
                actualSeg = segmentID%segments.size();
                while (distanceAlongSeg>=segments[actualSeg].length)
                {
                    distanceAlongSeg -= segments[actualSeg].length;
                    actualSeg++;
                    if (actualSeg==segments.size())
                        actualSeg = 0;
                }
            }
            else
            {
                if (segmentID>=segments.size())
                    return 0xdeadbeefu;

                actualSeg = segmentID;
                while (distanceAlongSeg>=segments[actualSeg].length)
                {
                    distanceAlongSeg -= segments[actualSeg].length;
                    actualSeg++;
                    if (actualSeg==segments.size())
                        return 0xdeadbeefu;
                }
            }

            if (paramHint)
            {
                if (actualSeg!=segmentID)
                    *paramHint = -1.f;

                *paramHint = segments[actualSeg].getParameterFromArcLen(distanceAlongSeg,*paramHint,accuracyThresh);
				//assert(*paramHint < segments[actualSeg].parameterLength);
                pos = segments[actualSeg].posHelper(*paramHint);
            }
            else
                pos = segments[actualSeg].posHelper(segments[actualSeg].getParameterFromArcLen(distanceAlongSeg,-1.f,accuracyThresh));

            return actualSeg;
        }
        virtual bool        getPos_fromParameter(vectorSIMDf& pos, const uint32_t& segmentID, const float& parameter) const
        {
            if (segmentID>=segments.size()||parameter>segments[segmentID].parameterLength)
                return false;

            pos = segments[segmentID].posHelper(parameter);

            return true;
        }

        //to get direction to look in
        virtual bool        getUnnormDirection(vectorSIMDf& tan, const uint32_t& segmentID, const float& distanceAlongSeg) const
        {
            if (segmentID>=segments.size()||distanceAlongSeg>segments[segmentID].length)
                return false;

            tan = segments[segmentID].directionHelper(segments[segmentID].getParameterFromArcLen(distanceAlongSeg,-1.f,0.00390625f));

            return true;
        }
        virtual bool        getUnnormDirection_fromParameter(vectorSIMDf& tan, const uint32_t& segmentID, const float& parameter) const
        {
            if (segmentID>=segments.size()||parameter>segments[segmentID].parameterLength)
                return false;

            tan = segments[segmentID].directionHelper(parameter);

            return true;
        }

        //baw specific -- to be implemented later
        virtual bool      canGiveParameterUntilBlockChange() const {return false;}
        // pass in current position
        virtual float           getParameterUntilBlockChange(const uint32_t& segmentID, const float& param)
        {
            //if (segmentID>=segments.size()||param>=segments[segmentID].parameterLength)
                return -1.f;

            //return segments[segmentID].findNextBlockChange(param);
        }
        virtual core::vector<float>   getBlockChangesInSegment(const uint32_t& segmentID, float startParam=0.f)
        {
            core::vector<float> changes;
            //if (segmentID>=segments.size())
                return changes;
/*
            const Segment& seg = segments[segmentID];
            while (true)
            {
                float fnd = seg.findNextBlockChange(startParam);
                if (fnd<0.f)
                    return changes;

                startParam = fnd;
                reinterpret_cast<uint32_t&>(startParam)++;
                changes.push_back(fnd);
            }*/
        }


        virtual bool            isArcLengthPrecise() const {return true;}/**
        virtual float           parameterToDistance(const float& param) const
        {
        }
        virtual float           distanceToParameter(const float& dist) const
        {
        }**/

    protected:
        CQuadraticSpline(bool loop) : ISpline(loop) {}

        void finalize()
        {
            double lenDouble = 0;
            for (size_t i=0; i<segments.size(); i++)
            {
                lenDouble += segments[i].length;
            }
            splineLen = lenDouble;
        }


        class Segment
        {
            public:
				Segment(const vectorSIMDf& startPt, const vectorSIMDf& endPt, const vectorSIMDf& startGradient, float currentApproxLen, float tightness)
				{
					/// ad^2+bd+y0 = y1
					weights[2] = startPt;
					/// ad+b = (y1-y0)/d = w
					auto approxMidGradient = (endPt-startPt)/currentApproxLen;

					float K;
					auto comparelambda = [&]() -> bool
					{
						/// The differential 2ad+b is continuous
						/// b = Kf'(0)
						weights[1] = startGradient*K;
						/// ad+Kf'(0) = w
						/// a = (w-Kf'(0))/d
						weights[0] = (approxMidGradient - weights[1]) / currentApproxLen;
						finalize(currentApproxLen);
						return this->length < tightness*currentApproxLen;
					};

					uint32_t& it = reinterpret_cast<uint32_t&>(K);
					uint32_t first = 0u;
					int32_t count = 0x47800000;
					while (count > 0)
					{
						it = first;
						uint32_t step = count / 2u;
						it += step;
						if (comparelambda())
						{
							first = ++it;
							count -= step + 1;
						}
						else
							count = step;
					}
					it = first;
					comparelambda();
				}
				Segment(const Segment& previousSeg, const vectorSIMDf& startPt,const vectorSIMDf& endPt, float currentApproxLen, float tightness) :
					Segment(startPt,endPt,previousSeg.weights[0]*previousSeg.parameterLength*2.f+previousSeg.weights[1],currentApproxLen, tightness)
				{
				}
				static Segment createForBSpline(const vectorSIMDf& startPt, const vectorSIMDf& midCtrl, const vectorSIMDf& endPt)
				{
					Segment seg;
					seg.weights[2] = startPt;
					seg.weights[1] = (midCtrl-startPt)*2.f;
					seg.weights[0] = endPt+startPt-midCtrl*2.f;

					float bLen = dot(seg.weights[1],seg.weights[1]).X;
					if (bLen<0.000001f)
					{
						seg.weights[1] = endPt-startPt;
						seg.weights[0].set(0.f,0.f,0.f);
					}
					else if (std::abs(dot(seg.weights[0],seg.weights[1]).X)>std::sqrt(bLen*dot(seg.weights[0],seg.weights[0]).X)*0.999999f)
					{
						seg.weights[1] = endPt-startPt;
						seg.weights[0].set(0.f,0.f,0.f);
					}

					seg.finalize(1.f);
					return seg;
				}

				inline void finalize(const float &currentApproxLen)
				{
					parameterLength = currentApproxLen;

					float a = 4.f*core::lengthsquared(weights[0])[0];
					float b = 4.f*core::dot(weights[0],weights[1])[0];
					float c = core::lengthsquared(weights[1])[0];

					integrationConstants[0] = 0.25f*b/a;
					integrationConstants[1] = sqrtf(c) * integrationConstants[0];
					integrationConstants[2] = 2.f * core::sqrt(core::max<float>(a*c, 0.f)) + b;
					integrationConstants[3] = 2.f * a;
					integrationConstants[4] = 2.f * sqrtf(a);
					integrationConstants[5] = b;
					integrationConstants[6] = (0.25f*b*b/a-c) / integrationConstants[4];

					arcLen2ParameterLinear = 0.f;
					length = getArcLenFromParameter(parameterLength);
					if (isinf(length))
					{
						integrationConstants[0] = INFINITY;
						integrationConstants[2] = 0.f;
						if (a < 1.0e-10)
							length = sqrtf(c)*parameterLength;
						else
							assert(false);
					}
					else
					{
#ifdef _NBL_DEBUG
						assert(integrationConstants[0] < 10000000.f);
						assert(integrationConstants[2] >= 0.f);
#endif
					}
					arcLen2ParameterLinear = parameterLength / length;
				}

				inline float getArcLenFromParameter(const float &parameter) const
				{
					if (integrationConstants[0] < 10000000.f && integrationConstants[2] > 1.0e-40)
					{
						auto differential = directionHelper(parameter);
						float differentialLen = core::length(differential)[0];
						
						float a = 4.f*core::lengthsquared(weights[0])[0];
						float b = 4.f*core::dot(weights[0],weights[1])[0];
						float c = core::lengthsquared(weights[1])[0];

						//float x = parameter;

						/// integral sqrt(a x^2 + b x + c) dx =
						/// ((b + 2 a x) Sqrt[c + x (b + a x)])/(4 a) - ((b^2 - 4 a c) Log[b + 2 a x + 2 Sqrt[a] Sqrt[c + x (b + a x)]])/(8 a^(3/2))
						/// ((2 a x + b) differentialLen)/(4 a) - ((b^2 - 4 a c) log(2 sqrt(a) differentialLen + 2 a x + b))/(8 a^(3/2))
						/// @ x=0
						/// (b sqrt(c))/(4 a) - ((b^2 - 4 a c) log(2 sqrt(a) sqrt(c) + b))/(8 a^(3/2))

						/// Non Log
						/// ((2 a x + b) differentialLen - b sqrt(c))/(4 a)
						//float non_log_term = ((2.f*a*x+b)*differentialLen-b*sqrtf(c))/(4.f*a);
						float non_log_term = (0.5f*parameter+integrationConstants[0])*differentialLen-integrationConstants[1];

						/// Log 
						/// (log(2 sqrt(a) sqrt(c) + b)) - log(2 sqrt(a) differentialLen + 2 a x + b)))     ((b^2 - 4 a c)/(8 a^(3/2))
						/// log((2 sqrt(a c) + b))/(2 sqrt(a) differentialLen + 2 a x + b)))     ((b^2 - 4 a c)/(8 a^(3/2))
						float log_argument = integrationConstants[2] / (integrationConstants[3] * parameter + integrationConstants[4] * differentialLen+integrationConstants[5]);
						float log_term = logf(log_argument)*integrationConstants[6];

						return non_log_term + log_term;
					}
					else
						return parameter/arcLen2ParameterLinear;
				}

				inline float getParameterFromArcLen(const float& arcLen, float parameterHint, const float& accuracyThresh) const
				{
					if (true)
					{
						if (arcLen<=accuracyThresh)
							return arcLen;
						if (arcLen>=length-accuracyThresh)
							return parameterLength;
						if (parameterHint<0.f||parameterHint>parameterLength)
							parameterHint = arcLen*arcLen2ParameterLinear;
						/// dist = IndefInt(param) - lowerIntVal
						/// IndefInt^-1(dist+lowerIntVal) = param
						/// Newton-Raphson      f = arcLen - getArcLenFromParameter(parameterHint);
						/// Newton-Raphson      f' = -getArcLenFromParameter'(parameterHint);
						float arcLenDiffAtParamGuess = arcLen-getArcLenFromParameter(parameterHint);
						for (size_t i=0; std::abs(arcLenDiffAtParamGuess)>accuracyThresh&&i<32; i++)
						{
							float differentialAtGuess = core::length(directionHelper(parameterHint))[0];
							parameterHint += arcLenDiffAtParamGuess/differentialAtGuess;
							arcLenDiffAtParamGuess = arcLen-getArcLenFromParameter(parameterHint);
						}
						return std::min<float>(parameterHint,parameterLength);
					}
					else
						return arcLen*arcLen2ParameterLinear;
				}

				inline vectorSIMDf posHelper(const float& parameter) const
				{
					return (weights[0]*parameter+weights[1])*parameter+weights[2];
				}
				inline vectorSIMDf directionHelper(const float& parameter) const
				{
					return weights[0]*parameter*2.f+weights[1];
				}/*
				inline float findNextBlockChange(const float& param) const
				{
					vectorSIMDf startingNegFrac = posHelper(param);
					startingNegFrac = floor(startingNegFrac)-startingNegFrac;
					vectorSIMDf dir = directionHelper(param);
					float changes[3];
					for (uint32_t i=0; i<3; i++)
						changes[i] = findChange(startingNegFrac.pointer[i],dir.pointer[i]);

					float smallest;
					if (reinterpret_cast<uint32_t*>(changes)[0]<=reinterpret_cast<uint32_t*>(changes)[1])
					{
						if (reinterpret_cast<uint32_t*>(changes)[2]<=reinterpret_cast<uint32_t*>(changes)[0])
							smallest = changes[2];
						else
							smallest = changes[0];
					}
					else if (reinterpret_cast<uint32_t*>(changes)[2]<=reinterpret_cast<uint32_t*>(changes)[1])
					{
						smallest = changes[2];
					}
					else
						smallest = changes[1];

					smallest_+= param;
					if (smallest<length)
						return smallest;

					return -1.f;
				}
				inline float findChange(const float& currentNegFrac, const float& changePerParam) const
				{
					if (currentNegFrac==0.f||currentNegFrac==1.f)
						return 0.f;

					if (changePerParam < -FLT_MIN)
						return currentNegFrac/changePerParam;
					else if (changePerParam > FLT_MIN)
						return (1.f+currentNegFrac)/changePerParam;

					return -1.f;
				}*/


				float arcLen2ParameterLinear;
				//
				float integrationConstants[7];
				//
				vectorSIMDf weights[3];

				//
				float length, parameterLength;

            private:
                Segment() {}
        };

        core::vector<Segment> segments;
        float splineLen;
};


//! Loop code is wrong for now, unable to calculate A_0 so the gradients match all the way around the loop
class CQuadraticBSpline : public CQuadraticSpline
{
    public:
        CQuadraticBSpline(vectorSIMDf* controlPoints, const size_t& count, const bool loop = false) : CQuadraticSpline(loop)
        {
            //assert(count<0x80000000u && count);
            vectorSIMDf firstMidpoint = (controlPoints[0]+controlPoints[1])*0.5f;
            if (isLoop)
            {
                segments.push_back(Segment::createForBSpline((controlPoints[count-1]+controlPoints[0])*0.5f,controlPoints[0],firstMidpoint));
            }
            else
            {
                segments.push_back(Segment::createForBSpline(controlPoints[0],controlPoints[0],firstMidpoint));
            }

            vectorSIMDf midpoint = firstMidpoint;
            vectorSIMDf lastMid = midpoint;
            for (size_t i=2; i<count; i++)
            {
                midpoint = (controlPoints[i-1]+controlPoints[i])*0.5f;
                segments.push_back(Segment::createForBSpline(lastMid,controlPoints[i-1],midpoint));
                lastMid = midpoint;
            }

            if (isLoop)
            {
                segments.push_back(Segment::createForBSpline(lastMid,controlPoints[count-1],segments[0].weights[2]));
            }
            else
            {
                segments.push_back(Segment::createForBSpline(lastMid,lastMid,controlPoints[count-1]));
            }

            finalize();
        }
};

}
}

#endif

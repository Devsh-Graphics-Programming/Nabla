#ifndef _NBL_EXT_HATCH_H_
#define _NBL_EXT_HATCH_H_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>

#include <nbl/builtin/hlsl/math/equations/cubic.hlsl>
#include <nbl/builtin/hlsl/math/equations/quartic.hlsl>
#include <nbl/builtin/hlsl/math/geometry.hlsl>
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>

#include <complex.h>
#include <tgmath.h>
#include <nbl/builtin/hlsl/shapes/util.hlsl>

namespace nbl::ext::csg2d
{
	
using namespace nbl::core;
using nbl::hlsl::shapes::QuadraticBezier;
using nbl::hlsl::shapes::Quadratic;

enum class SweepMajorAxis
{
	MAJOR_X = 0u,
	MAJOR_Y = 1u,
};

/**
 * @brief A geometric processing engine that decomposes closed Bezier polygons into fillable slabs.
 *
 * The HatchBuilder utilizes a algorithm (similar to sweep-line but for beziers) to process a collection of quadratic Bezier 
 * curves that form closed polygons. As the sweep-line moves along the defined MajorAxis, it 
 * splits and sorts the curves to resolve complex overlaps and self-intersections.
 * * The algorithm outputs these resolved areas as `CurveBox` structures (often called slabs or 
 * monotone regions). Each CurveBox represents a discrete, continuous region bounded by two 
 * minor-axis curves (the left and right boundaries if SweepMajorAxis is MAJOR_Y).
 * * Crucially, this decomposition is governed by the **XOR (Even-Odd) fill rule**. By identifying 
 * where curves begin, end, and intersect, the builder pairs the active boundaries so that the 
 * resulting CurveBoxes represent exactly the interior regions that should be filled to achieve a 
 * topologically correct XOR fill of the original polygons.
 *
 * @tparam float_t The floating-point precision type used for all internal geometric calculations (e.g., float, double).
 * @tparam MajorAxis The primary axis direction for the sweep-line to travel across. Defaults to SweepMajorAxis::MAJOR_Y.
 */
template <typename float_t, SweepMajorAxis MajorAxis = SweepMajorAxis::MAJOR_Y>
class HatchBuilder 
{
public:
	// Constants calculated at compile time
	static constexpr int MajorIdx = static_cast<int>(MajorAxis);
	static constexpr int MinorIdx = 1 - MajorIdx;

	using float_t2 = nbl::hlsl::portable_vector_t2<float_t>;
	using float_t3 = nbl::hlsl::portable_vector_t3<float_t>;
	using float_t4 = nbl::hlsl::portable_vector_t4<float_t>;
	using float_t2x2 = nbl::hlsl::portable_matrix_t2x2<float_t>;

	struct CurveBox 
	{
		QuadraticBezier<float_t> minCurve; // left curve if Major is Y
		QuadraticBezier<float_t> maxCurve; // right curve if Major is Y
		float_t2 aabbMin;
		float_t2 aabbMax;
	};
	
	struct Segment
	{
		const QuadraticBezier<float_t>* originalBezier = nullptr;
		// because beziers are broken down, depending on the type this is t_start or t_end
		float_t t_start;
		float_t t_end; // beziers get broken down
	};

	/**
	 * @brief Constructs a new HatchBuilder.
	 * @param logger An optional smart pointer to the Nabla system logger.
	 * @param majorAxis The primary axis for the sweep-line algorithm (0 for X, 1 for Y).
	 */
	HatchBuilder(nbl::system::logger_opt_smart_ptr logger) 
		: m_logger(std::move(logger))
	{
	}

	
	/**
	 * @brief Ensures a bezier is strictly monotonic along the major axis before adding it.
	 * Reorders control points if the curve is running backwards along the sweep.
	 * * @param bezier The pre-split, guaranteed monotonic bezier curve.
	 */
	void addMonotonicBezier(const QuadraticBezier<float_t>& bezier) 
	{
		QuadraticBezier<float_t> outputBezier = bezier;
		
		// Ensure strictly increasing along major axis
		if (outputBezier.P0[MajorIdx] > outputBezier.P2[MajorIdx]) 
		{
			outputBezier.P2 = bezier.P0;
			outputBezier.P0 = bezier.P2;
		}

		// Minor precision fix
		if (outputBezier.P1[MajorIdx] < outputBezier.P0[MajorIdx])
			outputBezier.P1[MajorIdx] = outputBezier.P0[MajorIdx];

#ifdef DEBUG_HATCH_VISUALLY
		if (debugOutput)
		{
			uint32_t bezierIdx = beziers.size();
			float32_t4 colors[5] = {
				float32_t4(33,150,243, 255) / float32_t4(255.0),
				float32_t4(29,233,182, 255) / float32_t4(255.0),
				float32_t4(238,255,65, 255) / float32_t4(255.0),
				float32_t4(244,81,30, 255) / float32_t4(255.0),
				float32_t4(211,47,47, 255) / float32_t4(255.0)
			};
			//drawDebugBezier(bezier, colors[bezierIdx % 5]);
		}
#endif

		m_beziers.push_back(outputBezier);
	}


	/**
	 * @brief Adds a single bezier curve to the builder's internal queue.
	 * * @param bezier The quadratic bezier curve to add.
	 */
	void addBezier(const QuadraticBezier<float_t>& bezier) 
	{
		std::array<QuadraticBezier<float_t>, 2> monotonicSegments;
		
		// Assuming Impl exists and provides this functionality
		bool isMonotonic = splitIntoMajorMonotonicSegments(bezier, monotonicSegments);

		if (isMonotonic) 
		{
			addMonotonicBezier(bezier);
#ifdef DEBUG_HATCH_VISUALLY
			//if (debugOutput)
				//drawDebugBezier(unsplitBezier, float32_t4(0.8, 0.8, 0.8, 1.0));
#endif
		} 
		else 
		{
			addMonotonicBezier(monotonicSegments[0]);
			addMonotonicBezier(monotonicSegments[1]);
#ifdef DEBUG_HATCH_VISUALLY
			//if (debugOutput)
			//{ drawDebugBezier(monotonicSegments.data()[0], float32_t4(1.0, 0.0, 0.0, 1.0)); drawDebugBezier(monotonicSegments.data()[1], float32_t4(0.0, 1.0, 0.0, 1.0)); }
#endif
		}
	}

	/**
	 * @brief Clears all queued segments. 
	 * Useful for reusing the builder instance without reallocating memory.
	 */
	void clear() 
	{
		m_beziers.clear();
	}

	// ========================================================================
	// REQUIRED DERIVED INTERFACE
	// ========================================================================
	/**
	 * @brief Called by process() whenever a completed monotonic region (Curve Box/Slab) is generated.
	 * * This pure virtual function must be implemented by the derived class. It acts as the 
	 * primary callback during the sweep-line algorithm, allowing the user to process, render, or store the resulting geometric slabs.
	 * @param curveBox The generated CurveBox 
	 */
	virtual void consumeCurveBox(const CurveBox& curveBox) = 0;

	/**
	 * @brief Executes the CSG/Sweep-line algorithm on all added beziers.
	 * As CurveBoxes are discovered, this function will automatically trigger `consumeCurveBox()`.
	 */
	void build() 
	{	
		// this threshsold is used to decide when to consider MinorIdx position to be 
		// the same and check tangents because intersection algorithms has rounding errors
		constexpr float_t MinorPositionComparisonThreshhold = 1e-3;
		constexpr float_t TangentComparisonThreshhold = 1e-7;

		std::stack<Segment> starts; // Next segments sorted by start points
		std::stack<float_t> ends; // Next end points
		std::priority_queue<float_t, std::vector<float_t>, std::greater<float_t> > intersections; // Next intersection points as MajorIdx coordinate
		float_t maxMajor;

#ifdef DEBUG_HATCH_VISUALLY

	/* old parameters int32_t* debugStepPtr, const std::function<void(CPolyline, LineStyleInfo)>& debugOutput */

	int32_t debugStepDefault = 0u;
	int32_t& debugStep = (debugStepPtr) ? *debugStepPtr : debugStepDefault;

	auto drawDebugBezier = [&](QuadraticBezier<float_t> bezier, float32_t4 color)
	{
		CPolyline outputPolyline;
		std::vector<shapes::QuadraticBezier<float_t>> beziers;
		shapes::QuadraticBezier<float_t> tmpBezier;
		tmpBezier.P0 = bezier.P0;
		tmpBezier.P1 = bezier.P1;
		tmpBezier.P2 = bezier.P2;
		beziers.push_back(tmpBezier);
		outputPolyline.addQuadBeziers(beziers);

		LineStyleInfo lineStyleInfo;
		lineStyleInfo.screenSpaceLineWidth = 4.0f;
		lineStyleInfo.worldSpaceLineWidth = 0.0f;
		lineStyleInfo.color = color;

		debugOutput(outputPolyline, lineStyleInfo);
	};

	auto drawDebugLine = [&](float64_t2 start, float64_t2 end, float32_t4 color)
	{
		CPolyline outputPolyline;
		std::vector<float64_t2> points;
		points.push_back(start);
		points.push_back(end);
		outputPolyline.addLinePoints(points);
		
		LineStyleInfo lineStyleInfo;
		lineStyleInfo.screenSpaceLineWidth = 2.0f;
		lineStyleInfo.worldSpaceLineWidth = 0.0f;
		lineStyleInfo.color = color;
		
		debugOutput(outputPolyline, lineStyleInfo);
	};
#endif

		// Inititalize start segments in a sorted manner
		{
			std::vector<Segment> segments;

			for (uint32_t bezierIdx = 0; bezierIdx < m_beziers.size(); bezierIdx++)
			{
				auto hatchBezier = &m_beziers[bezierIdx];
				Segment segment;
				segment.originalBezier = hatchBezier;
				segment.t_start = 0.0;
				segment.t_end = 1.0;
				segments.push_back(segment);
			}

			if (segments.empty())
			{
				m_logger.log("Empty Polylines with no segments were fed into the Hatch construction.", nbl::system::ILogger::ELL_WARNING);
				return;
			}

			std::sort(segments.begin(), segments.end(), [&](const Segment& a, const Segment& b) { return a.originalBezier->P0[MajorIdx] > b.originalBezier->P0[MajorIdx]; });
			for (Segment& segment : segments)
				starts.push(segment);

			std::sort(segments.begin(), segments.end(), [&](const Segment& a, const Segment& b) { return a.originalBezier->P2[MajorIdx] > b.originalBezier->P2[MajorIdx]; });
			for (Segment& segment : segments)
				ends.push(segment.originalBezier->P2[MajorIdx]);
			maxMajor = segments.front().originalBezier->P2[MajorIdx];
		}

		
#ifdef DEBUG_HATCH_VISUALLY
		int32_t step = 0;
#endif

		// Sweep line algorithm
		std::vector<Segment> activeCandidates; // Set of active candidates for neighbor search in sweep line

		// if we weren't spawning quads, we could just have unsorted `vector<Bezier*>`
		auto candidateComparator = [&](const Segment& lhs, const Segment& rhs)
		{
			// btw you probably want the beziers in Quadratic At^2+B+C form, not control points
			double _lhs = lhs.originalBezier->evaluate(lhs.t_start)[MinorIdx];
			double _rhs = rhs.originalBezier->evaluate(rhs.t_start)[MinorIdx];

			double lenLhs = glm::distance(lhs.originalBezier->P0, lhs.originalBezier->P2);
			double lenRhs = glm::distance(rhs.originalBezier->P0, rhs.originalBezier->P2);
			auto minLen = std::min(lenLhs, lenRhs);
#ifdef DEBUG_HATCH_VISUALLY
			if (debugOutput && step == debugStep)
			{
				//printf(std::format("comparison: lhs = ({}, {}), ({}, {}), ({}, {}) rhs = ({}, {}), ({}, {}), ({}, {})",
				//	lhs.originalBezier->P0.x, lhs.originalBezier->P0.y, 
				//	lhs.originalBezier->P1.x, lhs.originalBezier->P1.y, 
				//	lhs.originalBezier->P2.x, lhs.originalBezier->P2.y, 

				//	rhs.originalBezier->P0.x, rhs.originalBezier->P0.y, 
				//	rhs.originalBezier->P1.x, rhs.originalBezier->P1.y, 
				//	rhs.originalBezier->P2.x, rhs.originalBezier->P2.y
				//	).c_str());
				//drawDebugLine(float64_t2(_lhs, -1000.0), float64_t2(_lhs, 1000.0), float64_t4(0.1, 0.1, 1.0, 1.0));
				//drawDebugLine(float64_t2(_rhs, -1000.0), float64_t2(_rhs, 1000.0), float64_t4(0.1, 1.0, 1.0, 1.0));
				//printf(std::format("(comparing MinorIdx) _lhs: {} (len: {}) _rhs: {} (len: {}) minLen: {} diff: {} ",
				//	_lhs, lenLhs, _rhs, lenRhs, minLen, abs(_lhs - _rhs)).c_str());
			}
#endif

			// Threshhold here for intersection points, where the MinorIdx values for the curves are
			// very close but could be smaller, causing the curves to be in the wrong order
			if (abs(_lhs - _rhs) < MinorPositionComparisonThreshhold * minLen)
			{
				// this is how you want to order the derivatives dmin/dmaj=-INF dmin/dmaj = 0 dmin/dmaj=INF
				// also leverage the guarantee that `dmaj>=0` to ger numerically stable compare
				// also leverage the guarantee that `dmaj>=0` to ger numerically stable compare
				auto lhsQuadratic = Quadratic<float_t>::constructFromBezier(*lhs.originalBezier);
				auto rhsQuadratic = Quadratic<float_t>::constructFromBezier(*rhs.originalBezier);

				float64_t2 lTan = lhs.originalBezier->derivative(lhs.t_start);
				float64_t2 rTan = rhs.originalBezier->derivative(rhs.t_start);
				_lhs = lTan[MinorIdx] * rTan[MajorIdx];
				_rhs = rTan[MinorIdx] * lTan[MajorIdx];
#ifdef DEBUG_HATCH_VISUALLY
				if (false) //(debugOutput && step == debugStep)
				{
					printf(std::format("(comparing tangent) lTan: {}, {} rTan: {}, {} _lhs: {} _rhs: {} abs(_lhs - _rhs): {} abs(_lhs - 0.0): {} ",
						lTan.x, lTan.y, rTan.x, rTan.y, _lhs, _rhs, 
						abs(_lhs - _rhs), abs(_lhs - 0.0)).c_str());
				}
#endif
				// negative values mess with the comparison operator when using multiplication
				// they should be positive because of MajorIdx monotonicity
				assert(lTan[MajorIdx] >= 0.0);
				assert(rTan[MajorIdx] >= 0.0);

				if (abs(_lhs - _rhs) < TangentComparisonThreshhold)
				{
					float64_t2 lAcc = 2.0 * lhsQuadratic.A;
					float64_t2 rAcc = 2.0 * rhsQuadratic.A;

					// In this branch, _lhs == _rhs == 0 (tangents are both 0)
					if (abs(_lhs - 0.0) < TangentComparisonThreshhold)
					{
						// TODO https://discord.com/channels/593902898015109131/723305695046533151/1169377896658383008
						bool lTanSign = lTan[MinorIdx] >= 0.0;
						bool rTanSign = rTan[MinorIdx] >= 0.0;
						// CASE A
						// 
						// If the signs of the horizontal tangents differ, we know thje negative 
						// one belongs to the left curve
						if (lTanSign != rTanSign)
						{
#ifdef DEBUG_HATCH_VISUALLY
							if (false) //(debugOutput && step == debugStep)
							{
								printf(std::format("(comparing sign) lTanSign: {} rTanSign: {} ",
									lTanSign ? "positive" : "negative", rTanSign ? "positive" : "negative").c_str());
								printf("\n");
							}
#endif
							// We want to return true if lhs < rhs (lhs is to the left of rhs)
							// For this to be false, rhs would need to be the left one (and therefore negative)
							return rTanSign;
						}

						// Otherwise (CASE B)
						// 
						// In this case tangents are both on the same side
						// so only the magnitude / abs of the d2Major / dMinor2 is important
						_lhs = -abs(lAcc[MinorIdx] * lTan[MajorIdx] - lAcc[MajorIdx] * lTan[MinorIdx]) * pow(rTan[MinorIdx], 3.0);
						_rhs = -abs(rAcc[MinorIdx] * rTan[MajorIdx] - rAcc[MajorIdx] * rTan[MinorIdx]) * pow(lTan[MinorIdx], 3.0);
#ifdef DEBUG_HATCH_VISUALLY
						if (debugOutput && step == debugStep)
						{
							printf(std::format("(comparing second derivatives) A1={},{} B1={},{} A2={},{} B2={},{} _lhs={} _rhs={}",
								lhsQuadratic.A.x, lhsQuadratic.A.y, lhsQuadratic.B.x, lhsQuadratic.B.y,
								rhsQuadratic.A.x, rhsQuadratic.A.y, rhsQuadratic.B.x, rhsQuadratic.B.y,
								_lhs, _rhs
							).c_str());
						}
#endif
					}
					else
					{
						_lhs = (lAcc[MinorIdx] * lTan[MajorIdx] - lAcc[MajorIdx] * lTan[MinorIdx]) * pow(rTan[MajorIdx], 3.0);
						_rhs = (rAcc[MinorIdx] * rTan[MajorIdx] - rAcc[MajorIdx] * rTan[MinorIdx]) * pow(lTan[MajorIdx], 3.0);
					}
				}
			}
#ifdef DEBUG_HATCH_VISUALLY
			if (debugOutput && step == debugStep)
			{
				printf("\n");
			}
#endif
			return _lhs < _rhs;
		};
		auto addToCandidateSet = [&](const Segment& entry)
		{
			if (isSegmentStraightLineConstantMajor(entry))
				return;
			// Look for intersections among active candidates
			// this is a little O(n^2) but only in the `n=candidates.size()`
			for (const auto& segment : activeCandidates)
			{
				// find intersections entry vs segment
				auto intersectionPoints = intersectSegments(entry, segment);
#ifdef DEBUG_HATCH_VISUALLY
				if (debugOutput && step == debugStep)
				{
					for (uint32_t i = 0; i < intersectionPoints.size(); i++)
					{
						if (nbl::core::isnan(intersectionPoints[i]))
							continue;
						auto point = segment.originalBezier->evaluate(intersectionPoints[i]);
						auto min = point - 0.3;
						auto max = point + 0.3;
						drawDebugLine(float64_t2(min.x, min.y), float64_t2(max.x, min.y), float32_t4(0.0, 0.3, 0.0, 0.8));
						drawDebugLine(float64_t2(max.x, min.y), float64_t2(max.x, max.y), float32_t4(0.0, 0.3, 0.0, 0.8));
						drawDebugLine(float64_t2(min.x, max.y), float64_t2(max.x, max.y), float32_t4(0.0, 0.3, 0.0, 0.8));
						drawDebugLine(float64_t2(min.x, min.y), float64_t2(min.x, max.y), float32_t4(0.0, 0.3, 0.0, 0.8));
					}
				}
#endif

				for (uint32_t i = 0; i < intersectionPoints.size(); i++)
				{
					if (nbl::core::isnan(intersectionPoints[i]))
						continue;
					intersections.push(segment.originalBezier->evaluate(intersectionPoints[i])[MajorIdx]);
				}
			}
			activeCandidates.push_back(entry);
		};

		double lastMajor = starts.top().originalBezier->evaluate(starts.top().t_start)[MajorIdx];
		while (lastMajor!=maxMajor)
		{
#ifdef DEBUG_HATCH_VISUALLY
			if (debugOutput && step > debugStep)
				break;
			bool isCurrentDebugStep = step == debugStep;
#endif

			double newMajor;
			bool addStartSegmentToCandidates = false;

			if (ends.empty())
			{
				m_logger.log("Hatch Creation Failure: `ends` stack is empty in the main loop", nbl::system::ILogger::ELL_ERROR);
				_NBL_DEBUG_BREAK_IF(true); // This shouldn't happen, TODO: LOG
				break;
			}
			const double maxMajorEnds = ends.top();

			const Segment nextStartEvent = starts.empty() ? Segment() : starts.top();
			const double minMajorStart = nextStartEvent.originalBezier ? nextStartEvent.originalBezier->evaluate(nextStartEvent.t_start)[MajorIdx] : 0.0;

			// We check which event, within start, end and intersection events have the smallest
			// MajorIdx coordinate at this point
			auto intersectionVisit = [&]()
			{
				const double newMajor = intersections.top();
#ifdef DEBUG_HATCH_VISUALLY
				if (debugOutput && isCurrentDebugStep)
					drawDebugLine(float64_t2(-1000.0, newMajor), float64_t2(1000.0, newMajor), float32_t4(0.0, 0.0, 0.8, 1.0));
#endif
				intersections.pop(); // O(n)
				return newMajor;
			};

			// next start event is before next end event
			if (nextStartEvent.originalBezier && minMajorStart < maxMajorEnds)
			{
				// next start event is before next intersection event
				// (start event)
				if (intersections.empty() || minMajorStart < intersections.top()) // priority queue top() is O(1)
				{
					starts.pop();
					newMajor = minMajorStart;
					addStartSegmentToCandidates = true;
#ifdef DEBUG_HATCH_VISUALLY
					if (debugOutput && isCurrentDebugStep)
						drawDebugLine(float64_t2(-1000.0, newMajor), float64_t2(1000.0, newMajor), float32_t4(0.0, 0.8, 0.0, 1.0));
#endif
				}
				// (intersection event)
				else newMajor = intersectionVisit();
			}
			// next intersection event is before next end event
			// (intersection event)
			else if (!intersections.empty() && intersections.top() < maxMajorEnds)
				newMajor = intersectionVisit();
			else
			{
				// (end event)
				newMajor = maxMajorEnds;
				ends.pop();
#ifdef DEBUG_HATCH_VISUALLY
				if (debugOutput && isCurrentDebugStep)
					drawDebugLine(float64_t2(-1000.0, newMajor), float64_t2(1000.0, newMajor), float32_t4(0.0, 0.0, 0.8, 1.0));
#endif
				//std::cout << "End event at " << newMajor << "\n";
			}
			// spawn quads for the previous iterations if we advanced

			if (newMajor > lastMajor) 
			{
				const auto candidatesSize = std::distance(activeCandidates.begin(),activeCandidates.end());
				// Because n4ce works on loops, this must be `true` in almost every case, but can fail at times, because we skip adding beziers (lines) almost constant in MajorIdx direction
				if (candidatesSize % 2u == 0u)
				{
#ifdef DEBUG_HATCH_VISUALLY
					if (debugOutput && isCurrentDebugStep)
						drawDebugLine(float64_t2(-1000.0, lastMajor), float64_t2(1000.0, lastMajor), float32_t4(0.1, 0.1, 0.0, 0.5));
#endif
					// trim
					if ((candidatesSize % 2u) != 0u)
					{
						m_logger.log("Hatch Creation Failure: candidatesSize is odd", nbl::system::ILogger::ELL_ERROR);
						_NBL_DEBUG_BREAK_IF(true); // input polyline/polygon 
					}
#ifdef DEBUG_HATCH_VISUALLY
					if (candidatesSize % 2u == 1u)
					{
						for (uint32_t i = 0u; i < candidatesSize; i++)
						{
							const Segment& item = activeCandidates[i];
							auto curveMinEnd = intersectOrtho(*item.originalBezier, newMajor, MajorIdx);
							auto splitCurveMin = *item.originalBezier;
							splitCurveMin.splitCurveFromMinToMax(item.t_start, nbl::core::isnan(curveMinEnd) ? 1.0 : curveMinEnd);

							drawDebugBezier(splitCurveMin, (i == candidatesSize - 1) ? float32_t4(0.0, 0.0, 1.0, 1.0) : float32_t4(1.0, 0.0, 0.0, 1.0));
							if (i == candidatesSize - 1)
							{
								printf(std::format("problematic guy: ({}, {}), ({}, {}), ({}, {})",
									splitCurveMin.P0.x, splitCurveMin.P0.y,
									splitCurveMin.P1.x, splitCurveMin.P1.y,
									splitCurveMin.P2.x, splitCurveMin.P2.y
								).c_str());
							}
						}
					}
#endif
					for (auto i = 0u; i < (candidatesSize / 2) * 2;)
					{
						const Segment& left = activeCandidates[i++];
						const Segment& right = activeCandidates[i++];

						CurveBox curveBox = {};

						// Due to precision, if the curve is right at the end, intersectOrtho may return nan
						auto curveMinEnd = intersectOrtho(*left.originalBezier, newMajor, MajorIdx);
						auto curveMaxEnd = intersectOrtho(*right.originalBezier, newMajor, MajorIdx);

						auto splitCurveMin = *left.originalBezier;
						splitCurveMin.splitFromMinToMax(left.t_start, nbl::core::isnan(curveMinEnd) ? 1.0 : curveMinEnd);
						auto splitCurveMax = *right.originalBezier;
						splitCurveMax.splitFromMinToMax(right.t_start, nbl::core::isnan(curveMaxEnd) ? 1.0 : curveMaxEnd);

						assert(splitCurveMin.evaluate(0.0)[MajorIdx] <= splitCurveMin.evaluate(1.0)[MajorIdx]);
						assert(splitCurveMax.evaluate(0.0)[MajorIdx] <= splitCurveMax.evaluate(1.0)[MajorIdx]);

						auto curveMinAabb = getBezierBoundingBoxMinor(splitCurveMin);
						auto curveMaxAabb = getBezierBoundingBoxMinor(splitCurveMax);
						curveBox.aabbMin = float64_t2(std::min(curveMinAabb.first.x, curveMaxAabb.first.x), lastMajor);
						curveBox.aabbMax = float64_t2(std::max(curveMinAabb.second.x, curveMaxAabb.second.x), newMajor);

#ifdef DEBUG_HATCH_VISUALLY
						if (isCurrentDebugStep)
						{
							drawDebugBezier(splitCurveMin, float64_t4(1.0, 0.0, 0.0, 1.0));
							drawDebugBezier(splitCurveMax, float64_t4(0.0, 1.0, 0.0, 1.0));

							printf(std::format("AABB min: {}, {} max: {}, {} curve min: ({}, {}), ({}, {}), ({}, {}) curve max ({}, {}), ({}, {}), ({}, {})\n",
								curveBox.aabbMin.x, curveBox.aabbMin.y, curveBox.aabbMax.x, curveBox.aabbMax.y,

								splitCurveMin.P0.x, splitCurveMin.P0.y,
								splitCurveMin.P1.x, splitCurveMin.P1.y,
								splitCurveMin.P2.x, splitCurveMin.P2.y,
								splitCurveMax.P0.x, splitCurveMax.P0.y,
								splitCurveMax.P1.x, splitCurveMax.P1.y,
								splitCurveMax.P2.x, splitCurveMax.P2.y
							).c_str());
						}
#endif

						curveBox.minCurve = splitCurveMin;
						curveBox.maxCurve = splitCurveMax;
						consumeCurveBox(curveBox);
					}
				}

				// advance and trim all of the beziers in the candidate set
				auto oit = activeCandidates.begin();
				for (auto iit = activeCandidates.begin(); iit != activeCandidates.end(); iit++)
				{
					const double evalAtMajor = iit->originalBezier->evaluate(iit->t_end)[MajorIdx];

					auto origBez = iit->originalBezier;
					// if we scrolled past the end of the segment, remove it
					// (basically, we memcpy everything after something is different
					// and we skip on the memcpy for any items that are also different)
					// (this is supposedly a pattern with input/output operators)
					if (newMajor < evalAtMajor)
					{
						const double new_t_start = intersectOrtho(*iit->originalBezier, newMajor, MajorIdx);

						// little optimization (don't memcpy anything before something was removed)
						if (oit != iit)
							*oit = *iit;
						oit->t_start = new_t_start;
						oit++;
					}
				}
				// trim
				const auto newSize = std::distance(activeCandidates.begin(), oit);
				activeCandidates.resize(newSize);
			}

			// If we had a start event, we need to add the candidate
			if (addStartSegmentToCandidates)
			{
				addToCandidateSet(nextStartEvent);
			}
		
			// We'll need to sort if we had a start event and added to the candidate set
			// or if we have advanced our candidate set
			if (addStartSegmentToCandidates || newMajor > lastMajor)
			{
				std::sort(activeCandidates.begin(), activeCandidates.end(), candidateComparator);
			}

			if (newMajor > lastMajor)
				lastMajor = newMajor;

#ifdef DEBUG_HATCH_VISUALLY
			step++;
#endif
		}
#ifdef DEBUG_HATCH_VISUALLY
		debugStep = debugStep - step;
#endif

	}

	static bool isLineSegment(const QuadraticBezier<float_t>& bezier)
	{
		auto quadratic = Quadratic<float_t>::constructFromBezier(bezier);
		float_t lenSqA = dot(quadratic.A, quadratic.A);
		return lenSqA < exp(-23.0f) * dot(quadratic.B, quadratic.B);
	}

private:

	float_t intersectOrtho(const QuadraticBezier<float_t>& bezier, float_t lineConstant, int component)
	{
		// https://pomax.github.io/bezierinfo/#intersections
		float_t points[3];
		points[0] = bezier.P0[component];
		points[1] = bezier.P1[component];
		points[2] = bezier.P2[component];

		for (uint32_t i = 0; i < 3; i++)
			points[i] -= lineConstant;

		float_t A = points[0] - 2.0 * points[1] + points[2];
		float_t B = 2.0 * (points[1] - points[0]);
		float_t C = points[0];

		float_t2 roots = nbl::hlsl::math::equations::Quadratic<float_t>::construct(A, B, C).computeRoots();
		if (roots.x >= 0.0 && roots.x <= 1.0) return roots.x;
		if (roots.y >= 0.0 && roots.y <= 1.0) return roots.y;
		return nbl::core::nan<float_t>();
	}

	// returns two possible values of t in the lhs curve where the curves intersect
	std::array<float_t, 4> bezierBezierIntersections(const QuadraticBezier<float_t>& lhs, const QuadraticBezier<float_t>& rhs)
	{
		const auto quarticEquation = nbl::hlsl::shapes::getBezierBezierIntersectionEquation<float_t>(lhs, rhs);
	
		using nbl::hlsl::math::equations::Quartic;
		using nbl::hlsl::math::equations::Cubic;
		using nbl::hlsl::math::equations::Quadratic;
		constexpr float_t QUARTIC_THRESHHOLD = 1e-10;
	
		std::array<float_t, 4> t = { nbl::core::nan<float_t>(), nbl::core::nan<float_t>(), nbl::core::nan<float_t>(), nbl::core::nan<float_t>() }; // only two candidates in range, ever
	
		const float_t quadCoeffMag = std::max(std::abs(quarticEquation.d), std::abs(quarticEquation.e));
		const float_t cubCoeffMag = std::max(std::abs(quarticEquation.c), quadCoeffMag);
		const float_t quartCoeffMag = std::max(std::abs(quarticEquation.b), cubCoeffMag);

		if (std::abs(quarticEquation.a) > quartCoeffMag * QUARTIC_THRESHHOLD)
		{
			auto res = quarticEquation.computeRoots();
			memcpy(&t[0], &res.x, sizeof(float_t) * 4);
		}
		else if (abs(quarticEquation.b) > quadCoeffMag * QUARTIC_THRESHHOLD)
		{
			auto res = Cubic<float_t>::construct(quarticEquation.b, quarticEquation.c, quarticEquation.d, quarticEquation.e).computeRoots();
			memcpy(&t[0], &res.x, sizeof(float_t) * 3);
		}
		else
		{
			auto res = Quadratic<float_t>::construct(quarticEquation.c, quarticEquation.d, quarticEquation.e).computeRoots();
			memcpy(&t[0], &res.x, sizeof(float_t) * 2);
		}
	
		// TODO: why did we do this?
		// if (t[0] == t[1] || nbl::core::isnan(t[0]) || nbl::core::isnan(t[1]))
		//	t[0] = (t[0] != 0.0) ? 0.0 : 1.0;
	
		return t;
	}

	bool splitIntoMajorMonotonicSegments(const QuadraticBezier<float_t>& bezier, std::array<QuadraticBezier<float_t>, 2>& out)
	{
		auto quadratic = Quadratic<float_t>::constructFromBezier(bezier);

		// Getting derivatives for our quadratic bezier
		float_t a = quadratic.A[MajorIdx];
		float_t b = quadratic.B[MajorIdx];

		if (a == 0.0) // would cause inf and return false for a simple linear bezier
			return true;
		
		// Calculus 101: Finding roots for the quadratic bezier derivatives, this give us the t where the bezier changes direction, it almost always does, but if it's outside [0,1] range, then we consider it monotonic
		auto t = -b / (2.0 * a);
		if (t <= 0.0 || t >= 1.0) return true;
		QuadraticBezier<float_t> lower = bezier; lower.splitFromStart(t);
		QuadraticBezier<float_t> upper = bezier; upper.splitToEnd(t);
		out = {lower, upper};
		return false;
	}

	// https://pomax.github.io/bezierinfo/#boundingbox
	std::pair<float_t2, float_t2> getBezierBoundingBoxMinor(const QuadraticBezier<float_t>& bezier)
	{
		float_t A = bezier.P0[MinorIdx] - 2.0 * bezier.P1[MinorIdx] + bezier.P2[MinorIdx];
		float_t B = 2.0 * (bezier.P1[MinorIdx] - bezier.P0[MinorIdx]);

		const int searchTSize = 3;
		float_t searchT[searchTSize];
		searchT[0] = 0.0;
		searchT[1] = 1.0;
		searchT[2] = -B / (2 * A);

		float64_t2 min = float64_t2(std::numeric_limits<float_t>::infinity());
		float64_t2 max = float64_t2(-std::numeric_limits<float_t>::infinity());

		for (uint32_t i = 0; i < searchTSize; i++)
		{
			float_t t = searchT[i];
			if (t < 0.0 || t > 1.0 || nbl::core::isnan(t))
				continue;
			float64_t2 value = bezier.evaluate(t);
			min = float64_t2(std::min(min.x, value.x), std::min(min.y, value.y));
			max = float64_t2(std::max(max.x, value.x), std::max(max.y, value.y));
		}

		return std::pair<float64_t2, float64_t2>(min, max);
	}

	// checks if it's a straight line e.g. if you're sweeping along y axis the it's a line parallel to x
	static bool isSegmentStraightLineConstantMajor(const Segment& segment)
	{
		const float_t p0 = segment.originalBezier->P0[MajorIdx], 
			p1 = segment.originalBezier->P1[MajorIdx], 
			p2 = segment.originalBezier->P2[MajorIdx];
		//assert(p0 <= p1 && p1 <= p2); (PRECISION ISSUES ARISE ONCE MORE)
		return abs(p1 - p0) <= nbl::core::exp2(-24.0) && abs(p2 - p0) <= nbl::hlsl::exp(-24.0f);
	}

	std::array<float_t, 2> intersectSegments(const Segment& lhs, const Segment& rhs)
	{
		std::array<float_t, 2> result = { nbl::core::nan<float_t>(), nbl::core::nan<float_t>() };
		int resultIdx = 0;

		// Use line intersections if one or both of the beziers are linear (a = 0)
		const bool selfLinear = isLineSegment(*lhs.originalBezier);
		const bool otherLinear = isLineSegment(*rhs.originalBezier);
		if (selfLinear && otherLinear)
		{
			// Line/line intersection
			//TODO: use cpp-compat hlsl builtin
			auto intersectionPoint =  nbl::hlsl::shapes::util::LineLineIntersection<float_t>(
				lhs.originalBezier->P0, lhs.originalBezier->P2 - lhs.originalBezier->P0,
				rhs.originalBezier->P0, rhs.originalBezier->P2 - rhs.originalBezier->P0
			);
			const float_t x1 = lhs.originalBezier->P0.x, y1 = lhs.originalBezier->P0.y,
				x2 = lhs.originalBezier->P2.x, y2 = lhs.originalBezier->P2.y,
				x3 = rhs.originalBezier->P0.x, y3 = rhs.originalBezier->P0.y,
				x4 = rhs.originalBezier->P2.x, y4 = rhs.originalBezier->P2.y;

			// Return if point is on the lines
			if (std::min(x1, x2) <= intersectionPoint.x && y1 <= intersectionPoint.y && std::max(x1, x2) >= intersectionPoint.x && y2 >= intersectionPoint.y &&
				std::min(x3, x4) <= intersectionPoint.x && y3 <= intersectionPoint.y && std::max(x3, x4) >= intersectionPoint.x && y4 >= intersectionPoint.y)
			{
				// Gets t for "other" by using intersectOrtho
				auto otherT = intersectOrtho(*rhs.originalBezier, intersectionPoint.y, MajorIdx);
				auto intersectionMajor = rhs.originalBezier->evaluate(otherT)[MajorIdx];
				auto thisT = intersectOrtho(*lhs.originalBezier, intersectionMajor, MajorIdx);

				if (otherT >= rhs.t_start && otherT <= rhs.t_end && thisT >= lhs.t_start && thisT <= lhs.t_end)
				{
					result[0] = otherT;
				}
			}
		}
		else if (selfLinear || otherLinear)
		{
			// Line/curve intersection
			const auto& line = selfLinear ? *lhs.originalBezier : *rhs.originalBezier;
			const auto& curve = selfLinear ? *rhs.originalBezier : *lhs.originalBezier;

			float_t2  D = normalize(line.P2 - line.P0);
			float_t2x2 rotation = { 
				{D.x, D.y}, 
				{-D.y, D.x} 
			};
			QuadraticBezier<float_t> rotatedCurve = {
				mul(rotation, curve.P0 - line.P0),
				mul(rotation, curve.P1 - line.P0),
				mul(rotation, curve.P2 - line.P0)
			};

			auto intersectionCurveT = intersectOrtho(rotatedCurve, 0, (int)SweepMajorAxis::MAJOR_Y /* Always in rotation to align with X Axis */);
			auto intersectionMajor = curve.evaluate(intersectionCurveT)[MajorIdx];
			auto intersectionLineT = intersectOrtho(line, intersectionMajor, (int)MajorIdx);

			auto thisT = selfLinear ? intersectionLineT : intersectionCurveT;
			auto otherT = selfLinear ? intersectionCurveT : intersectionLineT;

			if (otherT >= rhs.t_start && otherT <= rhs.t_end && thisT >= lhs.t_start && thisT <= lhs.t_end)
			{
				result[0] = otherT;
			}
		}
		else
		{
			auto thisBezier = *lhs.originalBezier;
			thisBezier.splitFromMinToMax(lhs.t_start, lhs.t_end); // to get correct P0, P1, P2 for intersection testing

			const auto p0 = thisBezier.P0;
			const auto p1 = thisBezier.P1;
			const auto p2 = thisBezier.P2;
			const bool sideP1 = nbl::hlsl::cross2D(p2 - p0, p1 - p0) >= 0.0;
		
			const auto& otherBezier = *rhs.originalBezier;
			const std::array<float_t, 4> intersections = bezierBezierIntersections(otherBezier, thisBezier);

			for (uint32_t i = 0; i < intersections.size(); i++)
			{
				auto t = intersections[i];
				if (nbl::core::isnan(t) || rhs.t_start >= t || t >= rhs.t_end)
					continue;

				auto intersection = otherBezier.evaluate(t);
			
				// Optimization istead of doing SDF to find other T and check against bounds:
				// If both P1 and the intersection point are on the same side of the P0 -> P2 line of thisBezier, it's a a valid intersection
				const bool sideIntersection = nbl::hlsl::cross2D(p2 - p0, intersection - p0) >= 0.0;
				if (sideP1 != sideIntersection)
					continue;

				const bool duplicateT = (resultIdx > 0 && t == result[0]) || (resultIdx > 1 && t == result[1]);
				if (!duplicateT)
				{
					if (resultIdx < 2)
					{
						result[resultIdx] = t;
						resultIdx++;
					}
					else
					{
						_NBL_DEBUG_BREAK_IF(true); // more intersections that expected
					}
				}
			}
		}

		return result;
	}

	nbl::system::logger_opt_smart_ptr m_logger;
	std::vector<QuadraticBezier<float_t>> m_beziers; // Referenced into by the segments
};

} // namespace nbl::ext::csg2d

#endif

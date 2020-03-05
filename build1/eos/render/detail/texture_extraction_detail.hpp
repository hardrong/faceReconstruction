/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/detail/texture_extraction_detail.hpp
 *
 * Copyright 2014, 2015 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#ifndef TEXTURE_EXTRACTION_DETAIL_HPP_
#define TEXTURE_EXTRACTION_DETAIL_HPP_

#include "eos/render/detail/render_detail.hpp"

#include "opencv2/core/core.hpp"

/**
 * Implementations of internal functions, not part of the
 * API we expose and not meant to be used by a user.
 */
namespace eos {
	namespace render {
		namespace detail {

/**
 * Computes whether the given point is inside (or on the border of) the triangle
 * formed out of the given three vertices.
 *
 * @param[in] point The point to check.
 * @param[in] triV0 First vertex.
 * @param[in] triV1 Second vertex.
 * @param[in] triV2 Third vertex.
 * @return Whether the point is inside the triangle.
 */
inline bool is_point_in_triangle(cv::Point2f point, cv::Point2f triV0, cv::Point2f triV1, cv::Point2f triV2) {
	// See http://www.blackpawn.com/texts/pointinpoly/
	// Compute vectors
	cv::Point2f v0 = triV2 - triV0;
	cv::Point2f v1 = triV1 - triV0;
	cv::Point2f v2 = point - triV0;

	// Compute dot products
	float dot00 = v0.dot(v0);
	float dot01 = v0.dot(v1);
	float dot02 = v0.dot(v2);
	float dot11 = v1.dot(v1);
	float dot12 = v1.dot(v2);

	// Compute barycentric coordinates
	float invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	// Check if point is in triangle
	return (u >= 0) && (v >= 0) && (u + v < 1);
};

/**
 * Checks whether all pixels in the given triangle are visible and
 * returns true if and only if the whole triangle is visible.
 * The vertices should be given in screen coordinates, but with their
 * z-values preserved, so they can be compared against the depthbuffer.
 *
 * Obviously the depthbuffer given should have been created with the same projection
 * matrix than the texture extraction is called with.
 *
 * Also, we don't do perspective-correct interpolation here I think, so only
 * use it with affine and orthographic projection matrices.
 *
 * @param[in] v0 First vertex, in screen coordinates (but still with their z-value).
 * @param[in] v1 Second vertex.
 * @param[in] v2 Third vertex.
 * @param[in] depthbuffer Pre-calculated depthbuffer.
 * @return True if the whole triangle is visible in the image.
 */
bool is_triangle_visible(const cv::Vec4f& v0, const cv::Vec4f& v1, const cv::Vec4f& v2, cv::Mat depthbuffer)
{
	// #Todo: Actually, only check the 3 vertex points, don't loop over the pixels - this should be enough.

	auto viewport_width = depthbuffer.cols;
	auto viewport_height = depthbuffer.rows;

	// Well, in in principle, we'd have to do the whole stuff as in render(), like
	// clipping against the frustums etc.
	// But as long as our model is fully on the screen, we're fine. Todo: Doublecheck that.

	if (!detail::are_vertices_ccw_in_screen_space(v0, v1, v2))
		return false;

	cv::Rect bbox = detail::calculate_clipped_bounding_box(v0, v1, v2, viewport_width, viewport_height);
	int minX = bbox.x;
	int maxX = bbox.x + bbox.width;
	int minY = bbox.y;
	int maxY = bbox.y + bbox.height;

	//if (t.maxX <= t.minX || t.maxY <= t.minY) 	// Note: Can the width/height of the bbox be negative? Maybe we only need to check for equality here?
	//	continue;									// Also, I'm not entirely sure why I commented this out

	bool whole_triangle_is_visible = true;
	for (int yi = minY; yi <= maxY; yi++)
	{
		for (int xi = minX; xi <= maxX; xi++)
		{
			// we want centers of pixels to be used in computations. Do we?
			const float x = static_cast<float>(xi) + 0.5f;
			const float y = static_cast<float>(yi) + 0.5f;
			// these will be used for barycentric weights computation
			const double one_over_v0ToLine12 = 1.0 / detail::implicit_line(v0[0], v0[1], v1, v2);
			const double one_over_v1ToLine20 = 1.0 / detail::implicit_line(v1[0], v1[1], v2, v0);
			const double one_over_v2ToLine01 = 1.0 / detail::implicit_line(v2[0], v2[1], v0, v1);
			// affine barycentric weights
			const double alpha = detail::implicit_line(x, y, v1, v2) * one_over_v0ToLine12;
			const double beta = detail::implicit_line(x, y, v2, v0) * one_over_v1ToLine20;
			const double gamma = detail::implicit_line(x, y, v0, v1) * one_over_v2ToLine01;

			// if pixel (x, y) is inside the triangle or on one of its edges
			if (alpha >= 0 && beta >= 0 && gamma >= 0)
			{
				const double z_affine = alpha*static_cast<double>(v0[2]) + beta*static_cast<double>(v1[2]) + gamma*static_cast<double>(v2[2]);
				if (z_affine < depthbuffer.at<double>(yi, xi)) {
					whole_triangle_is_visible = false;
					break;
				}
			}
		}
		if (!whole_triangle_is_visible) {
			break;
		}
	}

	if (!whole_triangle_is_visible) {
		return false;
	}
	return true;
};

		} /* namespace detail */
	} /* namespace render */
} /* namespace eos */

#endif /* TEXTURE_EXTRACTION_DETAIL_HPP_ */

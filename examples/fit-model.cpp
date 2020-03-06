/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: examples/fit-model.cpp
 *
 * Copyright 2015 Patrik Huber
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
#include "../include/eos/core/Landmark.hpp"
#include "../include/eos/core/LandmarkMapper.hpp"
#include "../include/eos/fitting/nonlinear_camera_estimation.hpp"
#include "../include/eos/fitting/linear_shape_fitting.hpp"
#include "../include/eos/render/utils.hpp"
#include "../include/eos/render/texture_extraction.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>
#include <fstream>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::cout;
using std::endl;
using std::vector;
using std::string;

/**
 * Reads an ibug .pts landmark file and returns an ordered vector with
 * the 68 2D landmark coordinates.
 *
 * @param[in] filename Path to a .pts file.
 * @return An ordered vector with the 68 ibug landmarks.
 */
LandmarkCollection<cv::Vec2f> read_pts_landmarks(std::string filename)
{
	using std::getline;
	using cv::Vec2f;
	using std::string;
	LandmarkCollection<Vec2f> landmarks;
	landmarks.reserve(68);

	std::ifstream file(filename);
	if (!file.is_open()) {
		throw std::runtime_error(string("Could not open landmark file: " + filename));
	}

	string line;
	// Skip the first 3 lines, they're header lines:
	getline(file, line); // 'version: 1'
	getline(file, line); // 'n_points : 68'
	getline(file, line); // '{'

	int ibugId = 1;
	while (getline(file, line))
	{
		if (line == "}") { // end of the file
			break;
		}
		std::stringstream lineStream(line);

		Landmark<Vec2f> landmark;
		landmark.name = std::to_string(ibugId);
		if (!(lineStream >> landmark.coordinates[0] >> landmark.coordinates[1])) {
			throw std::runtime_error(string("Landmark format error while parsing the line: " + line));
		}
		// From the iBug website:
		// "Please note that the re-annotated data for this challenge are saved in the Matlab convention of 1 being
		// the first index, i.e. the coordinates of the top left pixel in an image are x=1, y=1."
		// ==> So we shift every point by 1:
		landmark.coordinates[0] -= 1.0f;
		landmark.coordinates[1] -= 1.0f;
		landmarks.emplace_back(landmark);
		++ibugId;
	}
	return landmarks;
};

/**
 * This app demonstrates estimation of the camera and fitting of the shape
 * model of a 3D Morphable Model from an ibug LFPW image with its landmarks.
 *
 * First, the 68 ibug landmarks are loaded from the .pts file and converted
 * to vertex indices using the LandmarkMapper. Then, an affine camera matrix
 * is estimated, and then, using this camera matrix, the shape is fitted
 * to the landmarks.
 */
int main(int argc, char *argv[])
{
	fs::path modelfile, isomapfile, imagefile, landmarksfile, mappingsfile, outputfile;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
			("model,m", po::value<fs::path>(&modelfile)->required()->default_value("../share/sfm_shape_3448.bin"),
				"a Morphable Model stored as cereal BinaryArchive")
			("image,i", po::value<fs::path>(&imagefile)->required()->default_value("../data/faces/image_0010.png"),
				"an input image")
			("landmarks,l", po::value<fs::path>(&landmarksfile)->required()->default_value("../data/faces/image_0010.pts"),
				"2D landmarks for the image, in ibug .pts format")
			("mapping,p", po::value<fs::path>(&mappingsfile)->required()->default_value("../share/ibug2did.txt"),
				"landmark identifier to model vertex number mapping")
			("output,o", po::value<fs::path>(&outputfile)->required()->default_value("output/out"),
				"basename for the output rendering and obj files")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: fit-model [options]" << endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
	}
	catch (const po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_SUCCESS;
	}

	// Load the image, landmarks, LandmarkMapper and the Morphable Model:
	Mat image = cv::imread(imagefile.string());
	LandmarkCollection<cv::Vec2f> landmarks;
	try {
		landmarks = read_pts_landmarks(landmarksfile.string());
		//cout << landmarks << endl;
	}
	catch (const std::runtime_error& e) {
		cout << "Error reading the landmarks: " << e.what() << endl;
		return EXIT_FAILURE;
	}
	morphablemodel::MorphableModel morphable_model;
	try {
		morphable_model = morphablemodel::load_model(modelfile.string());
	}
	catch (const std::runtime_error& e) {
		cout << "Error loading the Morphable Model: " << e.what() << endl;
		return EXIT_FAILURE;
	}
	core::LandmarkMapper landmark_mapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);

	// Draw the loaded landmarks:
	Mat outimg = image.clone();
	int i = 1;
	for (auto&& lm : landmarks) {
	    cv::Point numPoint(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f);
	    cv::rectangle(outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f), cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f), { 255, 0, 0 });
	    /// Keegan.Ren
	    /// TODO: plot the face point and point number in the image
	    char str_i[11];
	    sprintf(str_i,"%d",i);
	    cv::putText(outimg, str_i, numPoint, CV_FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(0,0,255));
	    ++i;
	}
	cout << "i = " << i << endl;
	cv::imshow("rect_outimg", outimg);
	cv::waitKey();
	
	// These will be the final 2D and 3D points used for the fitting:
	vector<Vec4f> model_points; // the points in the 3D shape model
	vector<int> vertex_indices; // their vertex indices
	vector<Vec2f> image_points; // the corresponding 2D landmark points

	// Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
	for (int i = 0; i < landmarks.size(); ++i) {
		auto converted_name = landmark_mapper.convert(landmarks[i].name);
		if (!converted_name) { // no mapping defined for the current landmark
			continue;
		}
		int vertex_idx = std::stoi(converted_name.get());
		Vec4f vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
		model_points.emplace_back(vertex);
		vertex_indices.emplace_back(vertex_idx);
		image_points.emplace_back(landmarks[i].coordinates);
	}

	/// Keegan.Ren
	cout << "model_point = " << endl;
	for (int i = 0; i < landmarks.size(); ++i) {
		//         3d points                                                    2d points
		cout << model_points[i] << "\t" << vertex_indices[i] << "\t" << image_points[i] << endl;
	}

	
	// Estimate the camera (pose) from the 2D - 3D point correspondences
	fitting::OrthographicRenderingParameters rendering_params = fitting::estimate_orthographic_camera(image_points, model_points, image.cols, image.rows);
	/// Keegan.Ren
	cout << "rendering_params = " << endl;
	std::cout << rendering_params.r_x << " " << rendering_params.r_y << " " << rendering_params.r_z << " " << rendering_params.t_x << " " << rendering_params.t_y << endl;
	std::cout << rendering_params.frustum.b << " " << rendering_params.frustum.l << " " << rendering_params.frustum.r << " " << rendering_params.frustum.t << endl; 
	Mat affine_from_ortho = get_3x4_affine_camera_matrix(rendering_params, image.cols, image.rows);
	// Keegan
	cout << "affine_from_ortho = " << endl;
	cout << affine_from_ortho << endl;
// 	cv::imshow("affine_from_ortho", affine_from_ortho);
// 	cv::waitKey();
	
	// The 3D head pose can be recovered as follows:
	float xaw_angle = glm::degrees(rendering_params.r_x);
	float yaw_angle = glm::degrees(rendering_params.r_y);
	float zaw_angle = glm::degrees(rendering_params.r_z);
	cout << "x_y_z_angle = " << endl;
	cout << xaw_angle << "\t" << yaw_angle << "\t" << zaw_angle << endl;
	// and similarly for pitch (r_x) and roll (r_z).

	// Estimate the shape coefficients by fitting the shape to the landmarks:
	//                                                                    bin模型           投影矩阵              图片二维点
	vector<float> fitted_coeffs = fitting::fit_shape_to_landmarks_linear(morphable_model, affine_from_ortho, image_points, vertex_indices);
	cout << "size = " << fitted_coeffs.size() << endl;
	for (int i = 0; i < fitted_coeffs.size(); ++i)
	  cout << fitted_coeffs[i] << endl;

	// Obtain the full mesh with the estimated coefficients:
	render::Mesh mesh = morphable_model.draw_sample(fitted_coeffs, vector<float>());

	// Extract the texture from the image using given mesh and camera parameters:
	Mat isomap = render::extract_texture(mesh, affine_from_ortho, image);

	// Save the mesh as textured obj:
	outputfile += fs::path(".obj");
	render::write_textured_obj(mesh, outputfile.string());

	// And save the isomap:
	outputfile.replace_extension(".isomap.png");
	cv::imwrite(outputfile.string(), isomap);

	cv::imshow(outputfile.string(), isomap);
	cv::waitKey();
	
	cout << "Finished fitting and wrote result mesh and isomap to files with basename " << outputfile.stem().stem() << "." << endl;

	return EXIT_SUCCESS;
}

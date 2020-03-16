#include "../include/eos/core/Landmark.hpp"
#include "../include/eos/core/LandmarkMapper.hpp"
#include "../include/eos/fitting/nonlinear_camera_estimation.hpp"
#include "../include/eos/fitting/linear_shape_fitting.hpp"
#include "../include/eos/render/utils.hpp"
#include "../include/eos/render/texture_extraction.hpp"
//OpenCV include
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
 
 
#if 0
#ifdef WIN32
#define BOOST_ALL_DYN_LINK    // Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
#define BOOST_ALL_NO_LIB    // Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#endif
#include "boost/program_options.hpp"
#include <boost/filesystem.hpp>
 
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
 
using namespace eos;
using namespace dlib;
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
using Eigen::Vector4f;
 
int main(int argc, char *argv[])
{
    /// read eos file
    fs::path modelfile, isomapfile,mappingsfile, outputfilename, outputfilepath;
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "display the help message")
            ("model,m", po::value<fs::path>(&modelfile)->required()->default_value("../share1/sfm_shape_3448.bin"), "a Morphable Model stored as cereal BinaryArchive")
            ("mapping,p", po::value<fs::path>(&mappingsfile)->required()->default_value("../share1/ibug2did.txt"), "landmark identifier to model vertex number mapping")
            ("outputfilename,o", po::value<fs::path>(&outputfilename)->required()->default_value("out"), "basename for the output rendering and obj files")
            ("outputfilepath,o", po::value<fs::path>(&outputfilepath)->required()->default_value("output/"), "basename for the output rendering and obj files")
            ;
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        if (vm.count("help")) {
            cout << "Usage: webcam_face_fit_model_keegan [options]" << endl;
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
 
    try
    {
        cv::VideoCapture cap(0);
        dlib::image_window win;
 
        // Load face detection and pose estimation models.
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        dlib::shape_predictor pose_model;
        dlib::deserialize("../share/shape_predictor_68_face_landmarks.dat") >> pose_model;
 
 
#define TEST_FRAME
        cv::Mat frame_capture;
#ifdef TEST_FRAME
        frame_capture = cv::imread("../data/image_0013.png");
        cv::imshow("input", frame_capture);
        cv::imwrite("frame_capture.png", frame_capture);
        cv::waitKey(1);
#endif
 
        // Grab and process frames until the main window is closed by the user.
        int frame_count = 0;
        while (!win.is_closed())
        {
            CAPTURE_FRAME:
            Mat image;
#ifndef TEST_FRAME
            cap >> frame_capture;
#endif
            frame_capture.copyTo(image);
 
            // Turn OpenCV's Mat into something dlib can deal with. Note that this just
            // wraps the Mat object, it doesn't copy anything. So cimg is only valid as
            // long as frame_capture is valid. Also don't do anything to frame_capture that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers. This basically means you shouldn't modify frame_capture
            // while using cimg.
            dlib::cv_image<dlib::bgr_pixel> cimg(frame_capture);
 
            // Detect faces
            std::vector<dlib::rectangle> faces = detector(cimg);
            if (faces.size() == 0) goto CAPTURE_FRAME;
            for (size_t i = 0; i < faces.size(); ++i)
            {
                cout << faces[i] << endl;
            }
            // Find the pose of each face.
            std::vector<dlib::full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i)
                shapes.push_back(pose_model(cimg, faces[i]));
 
            /// face 68 pointers
            for (size_t i = 0; i < shapes.size(); ++i)
            {
                morphablemodel::MorphableModel morphable_model;
                try
                {
                    morphable_model = morphablemodel::load_model(modelfile.string());
                }
                catch (const std::runtime_error& e)
                {
                    cout << "Error loading the Morphable Model: " << e.what() << endl;
                    return EXIT_FAILURE;
                }
                core::LandmarkMapper landmark_mapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);
 
                /// every face
                LandmarkCollection<Vec2f> landmarks;
                landmarks.reserve(68);
                cout << "point_num = " << shapes[i].num_parts() << endl;
                int num_face = shapes[i].num_parts();
                for (size_t j = 0; j < num_face; ++j)
                {
                    dlib::point pt_save = shapes[i].part(j);
                    Landmark<Vec2f> landmark;
                    /// input
                    landmark.name = std::to_string(j + 1);
                    landmark.coordinates[0] = pt_save.x();
                    landmark.coordinates[1] = pt_save.y();
                    //cout << shapes[i].part(j) << "\t";
                    landmark.coordinates[0] -= 1.0f;
                    landmark.coordinates[1] -= 1.0f;
                    landmarks.emplace_back(landmark);
                }
 
                // Draw the loaded landmarks:
                Mat outimg = image.clone();
                cv::imshow("image", image);
                cv::waitKey(10);
 
                int face_point_i = 1;
                for (auto&& lm : landmarks)
                {
                    cv::Point numPoint(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f);
                    cv::rectangle(outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f), cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f), { 255, 0, 0 });
                    char str_i[11];
                    sprintf(str_i, "%d", face_point_i);
                    cv::putText(outimg, str_i, numPoint, CV_FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(0, 0, 255));
                    ++i;
                }
                //cout << "face_point_i = " << face_point_i << endl;
                cv::imshow("rect_outimg", outimg);
                cv::waitKey(1);
 
                // These will be the final 2D and 3D points used for the fitting:
                std::vector<Vec4f> model_points; // the points in the 3D shape model
                std::vector<int> vertex_indices; // their vertex indices
                std::vector<Vec2f> image_points; // the corresponding 2D landmark points
 
                // Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
                for (int i = 0; i < landmarks.size(); ++i)
                {
                    auto converted_name = landmark_mapper.convert(landmarks[i].name);
                    if (!converted_name)
                    {
                        // no mapping defined for the current landmark
                        continue;
                    }
                    int vertex_idx = std::stoi(converted_name.get());
                    //Vec4f vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
                    auto vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
                    model_points.emplace_back(Vec4f(vertex.x(), vertex.y(), vertex.z(), 1.0f));
                    vertex_indices.emplace_back(vertex_idx);
                    image_points.emplace_back(landmarks[i].coordinates);
                }
 
                // Estimate the camera (pose) from the 2D - 3D point correspondences
                fitting::RenderingParameters rendering_params = fitting::estimate_orthographic_camera(image_points,
                                                                                                        model_points,
                                                                                                        image.cols,
                                                                                                        image.rows);
                Mat affine_from_ortho = get_3x4_affine_camera_matrix(rendering_params,
                                                                        image.cols,
                                                                        image.rows);
                //     cv::imshow("affine_from_ortho", affine_from_ortho);
                //     cv::waitKey();
 
                // The 3D head pose can be recovered as follows:
                float yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
 
 
                // Estimate the shape coefficients by fitting the shape to the landmarks:
                std::vector<float> fitted_coeffs = fitting::fit_shape_to_landmarks_linear(morphable_model,
                                                                                            affine_from_ortho,
                                                                                            image_points,
                                                                                            vertex_indices);
#if 0
                cout << "size = " << fitted_coeffs.size() << endl;
                for (int i = 0; i < fitted_coeffs.size(); ++i)
                    cout << fitted_coeffs[i] << endl;
#endif
 
                // Obtain the full mesh with the estimated coefficients:
                core::Mesh mesh = morphable_model.draw_sample(fitted_coeffs, std::vector<float>());
 
                // Extract the texture from the image using given mesh and camera parameters:
                Mat isomap = render::extract_texture(mesh, affine_from_ortho, image);
 
                ///// save obj
                std::stringstream strOBJ;
                strOBJ << std::setw(10) << std::setfill('0') << frame_count << ".obj";
 
                // Save the mesh as textured obj:
                outputfilename = strOBJ.str();
                std::cout << outputfilename << std::endl;
                auto outputfile = outputfilepath.string() + outputfilename.string();
                core::write_textured_obj(mesh, outputfile);
 
                // And save the isomap:
                outputfilename.replace_extension(".isomap.png");
                cv::imwrite(outputfilepath.string() + outputfilename.string(), isomap);
 
                cv::imshow("isomap_png", isomap);
                cv::waitKey(1);
 
                outputfilename.clear();
            }
            frame_count++;
 
            // Display it all on the screen
            win.clear_overlay();
            win.set_image(cimg);
            win.add_overlay(render_face_detections(shapes));
        }
    }
    catch (dlib::serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << " http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch (std::exception& e)
    {
        cout << e.what() << endl;
    }
 
    return EXIT_SUCCESS;
}

#pragma once
// DNN-based Face detector from 'Dlib' library (only available for DetrackerYoco version >= 2.0)
// see http://blog.dlib.net/2016/10/easily-create-high-quality-object.html 
// Notes:
// * The trained network is in the subdirectory 'data', with name 'dlib_mmod_human_face_detector.dat'
//   It was downloaded from http://dlib.net/files/mmod_human_face_detector.dat.bz2
// * The detected faces will get class id '1000' and the class string 'face'

#include "FaceDetectionRegion.h"

#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv\cv.hpp>

#include "dlib/image_transforms/../pixel.h"
#include "dlib/image_transforms/assign_image_abstract.h"
#include "dlib/image_transforms/../statistics.h"
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>

#include <stdlib.h>
#include <stdio.h>

namespace dlibwrapper {

	using namespace dlib;

	template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
	template <long num_filters, typename SUBNET> using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

	template <typename SUBNET> using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
	template <typename SUBNET> using rcon5 = relu<affine<con5<45, SUBNET>>>;

	using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

	class Net
	{
	public:
		net_type* net;

	public:
		Net()
		{
			net = NULL;
		}
	};
}

class FaceDetectorDlib 
{
public:
	/// All algorithm parameters for object detector have to provided in config file 'cfg'
	FaceDetectorDlib();
	virtual ~FaceDetectorDlib();

	virtual void doLazyInit(std::string networkFile);
	virtual std::vector<DetectionRegion*>* calculate(const cv::Mat img, double minConfidence);


protected:
	dlibwrapper::Net* mNet;
};



#include "dlib/DlibFaceDetector.h"

#include "dlib/image_transforms/../pixel.h"
#include "dlib/image_transforms/assign_image_abstract.h"
#include "dlib/image_transforms/../statistics.h"
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>

#include "dlib/ipl_image_hull.h"


#include <stdlib.h>
#include <stdio.h>

FaceDetectorDlib::FaceDetectorDlib() :
	mNet(NULL)
{
	mNet = new dlibwrapper::Net;
}

FaceDetectorDlib::~FaceDetectorDlib()
{
	if (mNet) {
	
		delete mNet->net;
	}
	delete mNet;
}

void FaceDetectorDlib::doLazyInit(std::string networkFile)
{

	assert(!mNet->net);


	// algorithm parameters	


	{

		mNet->net = new dlibwrapper::net_type;
		// load network from file
		dlib::deserialize(networkFile.c_str()) >> (*mNet->net);

	}
}

std::vector<DetectionRegion*>* FaceDetectorDlib::calculate(cv::Mat img, double minConfidence)
{
	// taken and adapted from http://dlib.net/dnn_mmod_face_detection_ex.cpp.html

	std::vector<DetectionRegion*>* result = new std::vector<DetectionRegion*>();

	IplImage iplImg = img;
	dlib::ipl_image_hull<dlib::rgb_pixel> imgDl(&iplImg);

	// FAH: Encountered the same issue as mentioned in https://github.com/davisking/dlib/issues/206
	//     - and used the same workaround from there
	dlib::matrix<dlib::rgb_pixel> imgMatrix;
	assign_image(imgMatrix, imgDl);

	//dlib::save_bmp(imgMatrix, "c:/tmp/dlib_image.bmp");	

	///
	/// ( ) Now do the inference, and populate the 'result' object with the detected faces
	///


	// TODO: We should wrarp this block into on function (returning an int), and call it via 'GPUWorker->call' !

	{
		// Note that you can process a bunch of images in a std::vector at once and it runs
		// much faster, since this will form mini-batches of images and therefore get
		// better parallelism out of your GPU hardware.  However, all the images must be
		// the same size.  To avoid this requirement on images being the same size we
		// process them individually in this example.
		auto detections = (*mNet->net)(imgMatrix);

		for (auto&& face : detections) {

			float confidence = face.detection_confidence;

			if (confidence > minConfidence) {

				FaceDetectionRegion* fdr = new FaceDetectionRegion();
				fdr->setBoundingBox(face.rect.left(), face.rect.top(), face.rect.right() - face.rect.left(), face.rect.bottom() - face.rect.top());
				fdr->setClassificationConfidence(confidence);

				result->push_back(fdr);
			}

		}
	}

	return result;
}


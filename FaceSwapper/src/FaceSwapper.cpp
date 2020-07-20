
#include <stdio.h>
#include <fstream>

#ifdef _WINDOWS
#define WIN32_LEAN_AND_MEAN
#define WIN64_LEAN_AND_MEAN

#include <windows.h>

#else
#include <unistd.h>
#endif

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

#include <iostream>

#include "IplAlg/Visual.h"
#include "iplJrs/IplJrsImage.h"
#include "TensorFlowFaceDetector.h"
#include "FaceSwapper/FaceSwapping.h"

std::vector<DetectionRegion*>* copyRegionList(std::vector<DetectionRegion*>* src) {
	std::vector<DetectionRegion*>* dst = new std::vector<DetectionRegion*>();

	for (int i = 0; i < src->size(); i++) {
		DetectionRegion* dr = new DetectionRegion();
		dr->copyFromRegion(src->at(i), true);
		dst->push_back(dr);
	}

	return dst;

}

int main(int argc, char** argv)
{

	if (argc < 4) {
		std::cerr << "Error: insufficient number of parameters" << std::endl;
		std::cerr << "Usage: FaceSwapper <inputImage> <faceImage> <outputImage>" << std::endl;
	}

	std::string inputImage = argv[1];
	std::string faceImage = argv[2];
	std::string outputImage = argv[3];

	Jrs::Ipl::ImagePtr inputPtr = NULL;
	Jrs::Ipl::ImagePtr facePtr = NULL;

	std::cout << "loading " << inputImage << std::endl;

	try {
		inputPtr = Jrs::Ipl::Visual::loadImage(inputImage);
	}
	catch (std::exception e) {
		inputPtr = Jrs::Ipl::Visual::loadImageWithCImg(inputImage);
	}

	std::cout << "loading " << faceImage << std::endl;

	try {
		facePtr = Jrs::Ipl::Visual::loadImage(faceImage);
	}
	catch (std::exception e) {
		facePtr = Jrs::Ipl::Visual::loadImageWithCImg(faceImage);
	}
	
	std::cout << "converting input " << std::endl;

		
	IplImage* inputBGR = Jrs::Ipl::Factory::cloneImage(inputPtr.get());
	Jrs::Ipl::reverseChannelSequenceOrigin(inputBGR, true, false, true);

	std::cout << "converting face " << std::endl;

	IplImage* faceBGR = Jrs::Ipl::Factory::cloneImage(facePtr.get());
	Jrs::Ipl::reverseChannelSequenceOrigin(faceBGR, true, false, true);

	std::cout << "running face det on both images " << std::endl;

	
	TensorFlowFaceDetector faceDetector;

	TensorFlowFaceDetector::TensorFlowImp tfimp = TensorFlowFaceDetector::TensorFlowImp::GPU;

	faceDetector.initCpp("./models/", "20170512-110547.pb", tfimp, 0.0);
	faceDetector.setMultiFaceDetectionMode(true);

	double moderateTh = 0.9;
	faceDetector.setParameter("DetectionThreshold", &moderateTh,0,"");

	faceDetector.analyseImage(inputBGR);
	std::vector<DetectionRegion*>* detectedInputRegions = NULL;
	detectedInputRegions = copyRegionList(faceDetector.getResults());

	printf("Input: number of detected regions:%d\n", (int)(detectedInputRegions->size()));

	double strictTh = 0.98;
	faceDetector.setParameter("DetectionThreshold", &strictTh, 0, "");
	faceDetector.analyseImage(faceBGR);
	std::vector<DetectionRegion*>* detectedFaceRegions = NULL;
	detectedFaceRegions = copyRegionList(faceDetector.getResults());

	printf("Face templates: number of detected regions:%d\n", (int)(detectedFaceRegions->size()));
	
	IplImage* targetBGR = Jrs::Ipl::Factory::cloneImage(inputBGR);
	Jrs::FaceSwapper::FaceSwapping fswap("./models/shape_predictor_68_face_landmarks.dat");
	//Jrs::FaceSwapper::FaceSwapping fswap("./models/face_landmark_model.dat",true);

	std::cout << "replacing ... " << std::endl;

	srand((unsigned)time(0));

	for (int i = 0; i < detectedInputRegions->size(); i++) {

		int faceId = rand() % detectedFaceRegions->size();

		std::cout << "replacing face " << i << " with generated face " << faceId << std::endl;

		fswap.swapFaces(inputBGR, targetBGR, faceBGR, detectedInputRegions->at(i), detectedFaceRegions->at(faceId));

		std::cout << "done " << std::endl;

		try {
			IplImage* targetRGB = Jrs::Ipl::Factory::cloneImage(targetBGR);
			Jrs::Ipl::reverseChannelSequenceOrigin(targetRGB, true, false, true);

			//Jrs::Ipl::Visual::saveImage(targetRGB, outputImage + "_" + std::to_string(i) +".png");

			Jrs::Ipl::Factory::deleteImage(targetRGB);
		}
		catch (std::exception e) {
			std::cerr << "failed to write image " << std::endl;
		}

	}

	try {
		IplImage* targetRGB = Jrs::Ipl::Factory::cloneImage(targetBGR);
		Jrs::Ipl::reverseChannelSequenceOrigin(targetRGB, true, false, true);

		Jrs::Ipl::Visual::saveImage(targetRGB, outputImage );

		Jrs::Ipl::Factory::deleteImage(targetRGB);
	}
	catch (std::exception e) {
		std::cerr << "failed to write image " << std::endl;
	}

	Jrs::Ipl::Factory::deleteImage(inputBGR);
	Jrs::Ipl::Factory::deleteImage(faceBGR);

	return 0;
}

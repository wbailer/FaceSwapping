#pragma once

#include "dlib/array2d.h"
#include "dlib/pixel.h"
#include "dlib/image_processing/generic_image.h"
#include "dlib/matrix/matrix.h"



namespace dlib
{

// ----------------------------------------------------------------------------------------

	/// This is a 'hull' around an existing dlib image 'img'. It does not hold ownership of the buffer.
	/// You can feed this to all JRS functions which operate on 'IplImage' objects
	/// Note that dlib images (usually constructed via 'dlib::array2D' type) have 'row-major' datalayout and multi-channel dlib images are 'interleaved' 
	///
    template <typename image_type>
	class dlib_image_hull_ipl
	{
	public:
		typedef typename dlib::image_traits<image_type>::pixel_type pixelType;
		
		inline dlib_image_hull_ipl(image_type* dlibImg)
		{
			// ( ) fetch image properties (depth, width, ...)

			int iplDepth = -1;
			int nChannels = -1;
			if (dlib::pixel_traits<pixelType>::num == 1) {
				// we have a one-channel image, and 'pixelType' is a 'native' Type (e.g. uint8)
				iplDepth = Jrs::Ipl::Access::TypeMapper<pixelType>();
				nChannels = 1;
			}
			else {
				// we have a multi-channel image, and 'pixelType' is a 'non-native' type (e.g. dlib::rgb_pixel)
				//  the 'native' type of the elements of a pixel can be fetched via 'dlib::pixel_traits<pixel_type>::basic_pixel_type>'
				iplDepth = Jrs::Ipl::Access::TypeMapper<dlib::pixel_traits<pixelType>::basic_pixel_type>();
				nChannels = dlib::pixel_traits<pixelType>::num;
			}
			int width = dlib::num_columns(*dlibImg);
			int height = dlib::num_rows(*dlibImg);

			// ( ) Now construct a smart-pointered IplImage with a _user_ managed buffer
			// last parameter 'false' means that create method will not allocate a buffer
			img = Jrs::Ipl::transferOwnershipImageWithUserManagedImageBuffer(Jrs::Ipl::Factory::createImage(width, height, nChannels, iplDepth, false));
			// set 'imageData' etc. so that image points to buffer of 'dlibImg'. note it is important that 'img->imageSize' is set as last statement.
			img->imageData = (char*) dlib::image_data(*dlibImg);
			img->imageDataOrigin = img->imageData;
			img->widthStep = dlib::width_step(*dlibImg);
			// we do not touch 'img->align'
			img->imageSize = Jrs::Ipl::Access::getPreciseImageSizeInBytes(img.get());
		}

		inline ~dlib_image_hull_ipl()
		{

		}

		inline IplImage* get()
			{ return img.get();  }

		inline const IplImage* getConst()
			{ return img.get(); }
		

	public:
		Jrs::Ipl::ImagePtr img;
	};
	

}


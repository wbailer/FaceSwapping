#pragma once

#include "dlib/algs.h"
#include "dlib/pixel.h"
#include "dlib/matrix/matrix_mat.h"
#include "dlib/image_processing/generic_image.h"


namespace dlib
{
	

	/// Note the 'ipl_image' is a 'dlib image hull' around The IplImage 'img' - so it references the same buffer !!
	/// You can feed an object of type 'dlib::ipl_image<T>' to _all_ dlib functions as an parameter where an dlib image is expected
	/// (a dlib image has typically the template parameter name 'image_type')
	/// Template parameter 'pixel_type' is typically 'dlib::rgb_pixel' or one of the native types (e.g., uint8_t, uint16_t, ...)
	/// See http://dlib.net/imaging.html for more info for 'pixel_type'
    template <typename pixel_type> 
	class ipl_image_hull
    {
    public:
        typedef pixel_type type;
        typedef dlib::default_memory_manager mem_manager_type;
		
		// 'img' can be also null pointer
        ipl_image_hull (IplImage* img) 
        {						
            init(img);
        }

		// 'img' can be also null pointer
		ipl_image_hull(const IplImage* img)
		{					
			init(const_cast<IplImage*>(img));
		}        

        unsigned long size () const 
		{ 
			return static_cast<unsigned long>(_nr*_nc); 
		}

        inline const pixel_type* operator[](const long row ) const
        {             
            return reinterpret_cast<const pixel_type*>( _data + _widthStep*row);
        }

		inline pixel_type* operator[](const long row) 
		{
			return reinterpret_cast<pixel_type*>(_data + _widthStep*row);
		}

        long nr() const { return _nr; }
        long nc() const { return _nc; }
        long width_step() const { return _widthStep; }        

    private:

        void init (IplImage* img) 
        {

			if (img) {

				if (img->nChannels > 1 && img->dataOrder != IPL_DATA_ORDER_PIXEL) {
					throw std::runtime_error("[JrsDlibBase] dlib::ipl_image::init: Multi-channel images must have _interleaveed_ data layout");
				}
				if ((img->depth & 0xFF) / 8 * img->nChannels != sizeof(pixel_type)) {
					throw std::runtime_error("[JrsDlibBase] dlib::ipl_image::init: The pixel type you gave doesn't match the size of pixel used by the IplImage image");
				}				
				_data = img->imageData;
				_widthStep = img->widthStep;
				_nr = img->height;
				_nc = img->width;
			} else {
				_data = 0;
				_widthStep = 0;
				_nr = 0;
				_nc = 0;
			}

        }

        char* _data;
        long _widthStep;
        long _nr;
        long _nc;
    };

	

// ----------------------------------------------------------------------------------------

	

    template <
        typename T
        >
		const matrix_op<op_array2d_to_mat<ipl_image_hull<T> > > mat(
		const ipl_image_hull<T>& m
    )
    {
		typedef op_array2d_to_mat<ipl_image_hull<T> > op;
        return matrix_op<op>(op(m));
    }

	

	// ----------------------------------------------------------------------------------------
	

	// Define the global functions that make ipl_image_hull a proper "generic image" according to
	// ../image_processing/generic_image.h
    template <typename T>
	struct image_traits<ipl_image_hull<T> >
    {
        typedef T pixel_type;
    };

    template <typename T>
	inline long num_rows(const ipl_image_hull<T>& img) { return img.nr(); }
    template <typename T>
	inline long num_columns(const ipl_image_hull<T>& img) { return img.nc(); }

    template <typename T>
    inline void* image_data(
		ipl_image_hull<T>& img
    )
    {
        if (img.size() != 0)
            return &img[0][0];
        else
            return 0;
    }

    template <typename T>
    inline const void* image_data(
		const ipl_image_hull<T>& img
    )
    {
        if (img.size() != 0)
            return &img[0][0];
        else
            return 0;
    }

    template <typename T>
    inline long width_step(
        const ipl_image_hull<T>& img
    ) 
    { 
        return img.width_step(); 
    }


// ----------------------------------------------------------------------------------------

}





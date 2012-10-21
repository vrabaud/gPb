#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>

using namespace cv;
using namespace std;

void HilbertTransform( const Mat & input, Mat & output)
{
    CV_Assert( input.depth() == CV_64F );

    // Hilbert transform is made in frequency domain
    Mat f_transform;
    dft(input, f_transform);

    double t;
    int n = input.rows;

    // multilply positive frequences by i (the following assumes compact fft
    // output format
    for (int i = 2; i < n; i+=2)
    {
        t = -f_transform.at<double>(i);
        f_transform.at<double>(i) = (f_transform.at<double>(i-1));
        f_transform.at<double>(i-1) = t;
    }

    // zero out zero frequency
    f_transform.at<double>(0) = 0;

    // and Nyquist frequency, if it exists.
    if ( n%2 == 0 )
    {
        f_transform.at<double>(n-1) = 0;
    }

    idft(f_transform, output, DFT_SCALE);
}

Mat getGaussianKernel2( int n, double sigma, int derivative, int ktype )
{
    CV_Assert( ktype == CV_32F || ktype == CV_64F );
    Mat kernel(n, 1, ktype);
    float* cf = (float*)kernel.data;
    double* cd = (double*)kernel.data;

    double sigmaX = sigma > 0 ? sigma : ((n-1)*0.5 - 1)*0.3 + 0.8;
    double scale2X = -0.5/(sigmaX*sigmaX);
    double sum = 0;

    int i;
    for( i = 0; i < n; i++ )
    {
        double x = i - (n-1)*0.5;
        double t = std::exp(scale2X*x*x);
        if (derivative == 1)
        {
            t *= 2*x*scale2X;
        }
        if (derivative == 2)
        {
            t *= 4*x*x*scale2X*scale2X - 2*scale2X;
        }
        if( ktype == CV_32F )
        {
            cf[i] = (float)t;
            sum += cf[i];
        }
        else
        {
            cd[i] = t;
            sum += cd[i];
        }
    }
    sum = 1./sum;
    for( i = 0; i < n; i++ )
    {
        if( ktype == CV_32F )
            cf[i] = (float)(cf[i]*sum);
        else
            cd[i] *= sum;
    }

    return kernel;
}


void textons(const Mat & image, Mat & text)
{

    unsigned long n_ori       = 8;                     /* number of orientations */
    double sigma_tg_filt_sm   = 2.0;                   /* sigma for small tg filters */
    double sigma_tg_filt_lg   = std::sqrt(2) * 2.0;   /* sigma for large tg filters */

    // convert to grayscale
    //Mat gray;
    //cvtColor(image, gray, CV_RGB2GRAY);

    // compute simplest of the non-simple gaussians 
    int support = 9;
    double xsigma = sigma_tg_filt_sm/std::sqrt(2);
    double ysigma = sigma_tg_filt_sm/(3*std::sqrt(2));

    Mat xgaussian = getGaussianKernel2(support, xsigma, 2, CV_64F);
    Mat ygaussian; 
    HilbertTransform(getGaussianKernel2(support, ysigma, 2, CV_64F),
                     ygaussian);

    Mat gaussian2d(support, support, CV_64F);
    for (int i = 0; i < support; i++)
    {
        for (int j = 0; j < support; j++)
        {
            gaussian2d.at<double>(i,j) = xgaussian.at<double>(i) * ygaussian.at<double>(j);
        }
    }

    double angle;
    Point2f center(4.f, 4.f);
    Size2i size(support, support);
    Mat gaussian2di(support, support, CV_8U);

    for (int i = 0; i < n_ori; i++)
    {
        angle = (360/n_ori)*i;
        Mat rotation = getRotationMatrix2D(center, angle, 1.0);
        Mat rotated;
        warpAffine(gaussian2d, rotated, rotation, size);

        rotated.convertTo(gaussian2di, CV_8U,100,10);
        cout <<gaussian2di << "\n";
    }

    /* compute texton filter set */
    //std::list<cv:Mat> filters_small;
    //for (int i = 1; i < n_ori, i++)
    //{





    /* compute texton filter set */
    //auto_collection< matrix<>, array_list< matrix<> > > filters_small = 
    //    lib_image::texton_filters(n_ori, sigma_tg_filt_sm);
    //auto_collection< matrix<>, array_list< matrix<> > > filters_large = 
    //    lib_image::texton_filters(n_ori, sigma_tg_filt_lg);
    //array_list< matrix<> > filters;
    //filters.add(*filters_small);
    //filters.add(*filters_large);

    /* compute textons */
    //auto_collection< matrix<>, array_list< matrix<> > > temtons;
    //matrix<unsigned long> t_assign = 
    //    lib_image::textons(gray, filters, temtons, 64);
    //t_assign = matrix<unsigned long>(
    //        lib_image::border_mirror_2D(
    //            lib_image::border_trim_2D(matrix<>(t_assign), border), border
    //            )
    //        );

    /* return textons */
    //textons = lib_image::border_trim_2D(matrix<>(t_assign), border);
}


int main()
{
    Mat M = getGaussianKernel2(7, 2.0, 0, CV_64F);
    Mat N = getGaussianKernel2(7, 2.0, 0, CV_64F);
    textons(M,N);
}

//auto_collection< matrix<>, array_list< matrix<> > > lib_image::texton_filters(
//        unsigned long n_ori,
//        double        sigma)
//{
//    lib_image::oe_filters(n_ori, sigma, filters_even, filters_odd);
//    matrix<> f_cs = lib_image::gaussian_cs_2D(
//            sigma, sigma, 0, M_SQRT2l, support, support);
//
//void lib_image::oe_filters(
//   unsigned long                                        n_ori,
//   double                                               sigma,
//   auto_collection< matrix<>, array_list< matrix<> > >& filters_even,
//   auto_collection< matrix<>, array_list< matrix<> > >& filters_odd)
//{
//            lib_image::oe_filters_odd(_n_ori, _sigma)
//          : lib_image::oe_filters_even(_n_ori, _sigma);
//
//auto_collection< matrix<>, array_list< matrix<> > > lib_image::oe_filters_even(
//   unsigned long n_ori,
//   double        sigma)
//{
//   return lib_image::gaussian_filters(n_ori, sigma, 2, false, 3.0);
//}
//
//auto_collection< matrix<>, array_list< matrix<> > > lib_image::oe_filters_odd(
//   unsigned long n_ori,
//   double        sigma)
//{
//   return lib_image::gaussian_filters(n_ori, sigma, 2, true, 3.0);
//}
//
//matrix<> lib_image::gaussian_cs_2D(
//   double        sigma_x, 
//   double        sigma_y,
//   double        ori,
//   double        scale_factor,
//   unsigned long support_x,
//   unsigned long support_y)
//{
//   /* compute standard deviation for center kernel */
//   double sigma_x_c = sigma_x / scale_factor;
//   double sigma_y_c = sigma_y / scale_factor;
//   /* compute center and surround kernels */
//   matrix<> m_center = lib_image::gaussian_2D(
//      sigma_x_c, sigma_y_c, ori, 0, false, support_x, support_y
//   );
//   matrix<> m_surround = lib_image::gaussian_2D(
//      sigma_x, sigma_y, ori, 0, false, support_x, support_y
//   );
//   /* compute center-surround kernel */
//   matrix<> m = m_surround - m_center;
//   /* make zero mean and unit L1 norm */
//   m -= mean(m);
//   m /= sum(abs(m));
//   return m;
//}
//
//auto_collection< matrix<>, array_list< matrix<> > > lib_image::gaussian_filters(
//   unsigned long        n_ori,
//   double               sigma,
//   unsigned int         deriv,
//   bool                 hlbrt,
//   double               elongation)
//{
//   array<double> oris = lib_image::standard_filter_orientations(n_ori);
//   return lib_image::gaussian_filters(oris, sigma, deriv, hlbrt, elongation);
//}
//
//array<double> lib_image::standard_filter_orientations(unsigned long n_ori) {
//   array<double> oris(n_ori);
//   double ori = 0;
//   double ori_step = (n_ori > 0) ? (M_PIl / static_cast<double>(n_ori)) : 0;
//   for (unsigned long n = 0; n < n_ori; n++, ori += ori_step)
//      oris[n] = ori;
//   return oris;
//}
//
////Tells someone to create filters
//auto_collection< matrix<>, array_list< matrix<> > > lib_image::gaussian_filters(
//   const array<double>& oris,
//   double               sigma,
//   unsigned int         deriv,
//   bool                 hlbrt,
//   double               elongation)
//{
//   /* compute support from sigma */
//   unsigned long support = static_cast<unsigned long>(math::ceil(3*sigma));
//   double sigma_x = sigma;
//   double sigma_y = sigma_x/elongation;
//   /* allocate collection to hold filters */
//   auto_collection< matrix<>, array_list< matrix<> > > filters(
//      new array_list< matrix<> >()
//   );
//   /* allocate collection of filter creators */
//   auto_collection< runnable, list<runnable> > filter_creators(
//      new list<runnable>()
//   );
//   /* setup filter creators */
//   unsigned long n_ori = oris.size();
//   for (unsigned long n = 0; n < n_ori; n++) {
//      auto_ptr< matrix<> > f(new matrix<>());
//      auto_ptr<filter_creator> f_creator(
//         new filter_creator(
//            sigma_x, sigma_y, oris[n], deriv, hlbrt, support, support, *f
//         )
//      );
//      filters->add(*f);
//      f.release();
//      filter_creators->add(*f_creator);
//      f_creator.release();
//   }
//   /* create filters */
//   child_thread::run(*filter_creators);
//   return filters;
//}
//
//class filter_creator : public runnable {
//public:
//   /*
//    * Constructor.
//    */
//   explicit filter_creator(
//      double        sigma_x,     /* sigma x */
//      double        sigma_y,     /* sigma y */
//      double        ori,         /* orientation */
//      unsigned int  deriv,       /* derivative in y-direction (0, 1, or 2) */
//      bool          hlbrt,       /* take hilbert transform in y-direction? */
//      unsigned long support_x,   /* x support */
//      unsigned long support_y,   /* y support */
//      matrix<>&     f)           /* output filter matrix */
//    : _sigma_x(sigma_x),
//      _sigma_y(sigma_y),
//      _ori(ori),
//      _deriv(deriv),
//      _hlbrt(hlbrt),
//      _support_x(support_x),
//      _support_y(support_y),
//      _f(f)
//   { }
//
//   /*
//    * Destructor.
//    */
//   virtual ~filter_creator() { /* do nothing */ }
//
//   /*
//    * Create the filter.
//    */
//   virtual void run() {
//      _f = lib_image::gaussian_2D(
//         _sigma_x, _sigma_y, _ori, _deriv, _hlbrt, _support_x, _support_y
//      );
//   }
//   
//protected:
//   double        _sigma_x;       /* sigma x */
//   double        _sigma_y;       /* sigma y */
//   double        _ori;           /* orientation */
//   unsigned int  _deriv;         /* derivative in y-direction (0, 1, or 2) */
//   bool          _hlbrt;         /* take hilbert transform in y-direction? */
//   unsigned long _support_x;     /* x support */
//   unsigned long _support_y;     /* y support */
//   matrix<>&     _f;             /* output filter matrix */
//};
//
//
///*
//* 1. textons_4real asks texton_filters for two families of filters (large &
// * small), with params n_ori, sigma_sm and n_ori, sigma_lg
// *
// * 2. texton_filters asks oe_filters for two families of filters (even & odd)
// * (now 4 families) with the same params n_ori and sigma (lg or sm). It also
// * asks gaussian_cs_2D for filter with sigma, sigma, 0, M_SQRT2l, support,
// * support, where support = ceil(3*sigma).
// *
// * 3a. oe_filters ask oe_filters_odd and oe_filters_even for corresponding odd
// * and even filters, sigma n_ori staying the same. Then oe_filters_odd and
// * oe_filters_even ask for gaussian_filters to create filters with parameters
// * n_ori, sigma, 2, false, 3.0 and
// * n_ori, sigma, 2, true, 3.0
// *
// * 4a. gaussian_filters finds n_ori directions, shortens sigma_y = sigma/3.0,
// * finds support = ceil(3*sigma) and asks filter creator for filter with
// * sigma, sigma_y, oris[n], 2, false, support, support and
// * sigma, sigma_y, oris[n], 2, true, support, support
// *
// * 5a. filter_creator asks gaussian_2D to create filters with the same
// * parameters as above
// *
// * 3b. gaussian_cs_2D calculates sigma_x_c = sigma_y_c = sigma/M_SQRT2l and
// * asks for gaussian_2D for a filters with 
// * sigma_x_c, sigma_y_c, 0, 0, false, support, support and 
// * sigma, sigma, 0, 0, false, support, support.
// * Returns then their difference
// *
// * 6a, 4b. gaussian_2D calculates stuff and asks gaussian for 1D filters, which
// * it then multiplies and returns. Total of 2*n_ori + 1 filter.
// * 
// * 7. gaussian does what it is told.
// */
//
//matrix<> lib_image::gaussian_2D(
//   double        sigma_x, 
//   double        sigma_y,
//   double        ori,
//   unsigned int  deriv,
//   bool          hlbrt,
//   unsigned long support_x,
//   unsigned long support_y)
//{
//   /* compute size of larger grid for axis-aligned gaussian   */
//   /* (reverse rotate corners of bounding box by orientation) */
//   unsigned long support_x_rot = support_x_rotated(support_x, support_y, -ori);
//   unsigned long support_y_rot = support_y_rotated(support_x, support_y, -ori);
//   /* compute 1D kernels */
//   matrix<> mx = lib_image::gaussian(sigma_x, 0,     false, support_x_rot);
//   matrix<> my = lib_image::gaussian(sigma_y, deriv, hlbrt, support_y_rot);
//   /* compute 2D kernel from product of 1D kernels */
//   matrix<> m(mx._size, my._size);
//   unsigned long n = 0;
//   for (unsigned long n_x = 0; n_x < mx._size; n_x++) {
//      for (unsigned long n_y = 0; n_y < my._size; n_y++) {
//         m._data[n] = mx._data[n_x] * my._data[n_y];
//         n++;
//      }
//   }
//   /* rotate 2D kernel by orientation */
//   m = lib_image::rotate_2D_crop(m, ori, 2*support_x + 1, 2*support_y + 1);
//   /* make zero mean (if returning derivative) */
//   if (deriv > 0)
//      m -= mean(m);
//   /* make unit L1 norm */
//   m /= sum(abs(m));
//   return m;
//}
//
//double support_x_rotated(double support_x, double support_y, double ori) {
//   const double sx_cos_ori = support_x * math::cos(ori);
//   const double sy_sin_ori = support_y * math::sin(ori);
//   double x0_mag = math::abs(sx_cos_ori - sy_sin_ori);
//   double x1_mag = math::abs(sx_cos_ori + sy_sin_ori);
//   return (x0_mag > x1_mag) ? x0_mag : x1_mag;
//}
//
//double support_y_rotated(double support_x, double support_y, double ori) {
//   const double sx_sin_ori = support_x * math::sin(ori);
//   const double sy_cos_ori = support_y * math::cos(ori);
//   double y0_mag = math::abs(sx_sin_ori - sy_cos_ori);
//   double y1_mag = math::abs(sx_sin_ori + sy_cos_ori);
//   return (y0_mag > y1_mag) ? y0_mag : y1_mag;
//}
//
//matrix<> lib_image::gaussian(
//   double sigma, unsigned int deriv, bool hlbrt)
//{
//   unsigned long support = static_cast<unsigned long>(math::ceil(3*sigma));
//   return lib_image::gaussian(sigma, deriv, hlbrt, support);
//}
//
//matrix<> lib_image::gaussian(
//   double sigma, unsigned int deriv, bool hlbrt, unsigned long support)
//{
//   /* enlarge support so that hilbert transform can be done efficiently */
//   unsigned long support_big = support;
//   if (hlbrt) {
//      support_big = 1;
//      unsigned long temp = support;
//      while (temp > 0) {
//         support_big *= 2;
//         temp /= 2;
//      }
//   }
//   /* compute constants */
//   const double sigma2_inv = double(1)/(sigma*sigma);
//   const double neg_two_sigma2_inv = double(-0.5)*sigma2_inv;
//   /* compute gaussian (or gaussian derivative) */
//   unsigned long size = 2*support_big + 1;
//   matrix<> m(size, 1);
//   double x = -(static_cast<double>(support_big)); 
//   if (deriv == 0) {
//      /* compute guassian */
//      for (unsigned long n = 0; n < size; n++, x++)
//         m._data[n] = math::exp(x*x*neg_two_sigma2_inv);
//   } else if (deriv == 1) {
//      /* compute gaussian first derivative */
//      for (unsigned long n = 0; n < size; n++, x++)
//         m._data[n] = math::exp(x*x*neg_two_sigma2_inv) * (-x);
//   } else if (deriv == 2) {
//      /* compute gaussian second derivative */
//      for (unsigned long n = 0; n < size; n++, x++) {
//         double x2 = x * x;
//         m._data[n] = math::exp(x2*neg_two_sigma2_inv) * (x2*sigma2_inv - 1);
//      }
//   } else {
//      throw ex_invalid_argument("only derivatives 0, 1, 2 supported");
//   }
//   /* take hilbert transform (if requested) */
//   if (hlbrt) {
//      /* grab power of two sized submatrix (ignore last element) */
//      m._size--;
//      m._dims[0]--;
//      /* grab desired submatrix after hilbert transform */
//      array<unsigned long> start(2);
//      array<unsigned long> end(2);
//      start[0] = support_big - support;
//      end[0] = start[0] + support + support;
//      m = (lib_signal::hilbert(m)).submatrix(start, end);
//   }
//   /* make zero mean (if returning derivative) */
//   if (deriv > 0)
//      m -= mean(m);
//   /* make unit L1 norm */
//   m /= sum(abs(m));
//   return m;
//}
//
//
//
//
////Filters and clusters
//matrix<unsigned long> lib_image::textons(
//   const matrix<>&                                      m,
//   const collection< matrix<> >&                        filters,
//   auto_collection< matrix<>, array_list< matrix<> > >& textons,
//   unsigned long                                        K,
//   unsigned long                                        max_iter,
//   double                                               subsampling)
//}
//
//
//
//
//
//

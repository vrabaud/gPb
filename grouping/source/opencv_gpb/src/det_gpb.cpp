#include "lib_image.h"
#include <opencv2/core/core.hpp>

void textons_4real(cv::Mat image, cv::Mat & textons)
{

    unsigned long n_ori       = 8;                     /* number of orientations */
    double sigma_tg_filt_sm   = 2.0;                   /* sigma for small tg filters */
    double sigma_tg_filt_lg   = math::sqrt(2) * 2.0;   /* sigma for large tg filters */

    /* convert to grayscale */
    matrix<> gray = lib_image::grayscale(L,a,b);


    /* compute texton filter set */
    auto_collection< matrix<>, array_list< matrix<> > > filters_small = 
        lib_image::texton_filters(n_ori, sigma_tg_filt_sm);
    auto_collection< matrix<>, array_list< matrix<> > > filters_large = 
        lib_image::texton_filters(n_ori, sigma_tg_filt_lg);
    array_list< matrix<> > filters;
    filters.add(*filters_small);
    filters.add(*filters_large);

    /* compute textons */
    auto_collection< matrix<>, array_list< matrix<> > > temtons;
    matrix<unsigned long> t_assign = 
        lib_image::textons(gray, filters, temtons, 64);
    t_assign = matrix<unsigned long>(
            lib_image::border_mirror_2D(
                lib_image::border_trim_2D(matrix<>(t_assign), border), border
                )
            );

    /* return textons */
    textons = lib_image::border_trim_2D(matrix<>(t_assign), border);
}

auto_collection< matrix<>, array_list< matrix<> > > lib_image::texton_filters(
        unsigned long n_ori,
        double        sigma)
{
    /* allocate collection to hold filters */
    auto_collection< matrix<>, array_list< matrix<> > > filters(
            new array_list< matrix<> >()
            );
    /* get even and odd-symmetric filter sets */
    auto_collection< matrix<>, array_list< matrix<> > > filters_even;
    auto_collection< matrix<>, array_list< matrix<> > > filters_odd;
    lib_image::oe_filters(n_ori, sigma, filters_even, filters_odd);
    /* add even and odd-symmetric filters to collection */
    filters->add(*filters_even); filters_even.release();
    filters->add(*filters_odd);  filters_odd.release();
    /* compute center surround filter */
    unsigned long support = static_cast<unsigned long>(math::ceil(3*sigma));
    matrix<> f_cs = lib_image::gaussian_cs_2D(
            sigma, sigma, 0, M_SQRT2l, support, support
            );
    /* add center surround filter to collection */
    auto_ptr< matrix<> > f_ptr(new matrix<>());
    matrix<>::swap(f_cs, *f_ptr);
    filters->add(*f_ptr);
    f_ptr.release();
    return filters;
}

//Tells oe_filters_odd and oe_filters_even to create filters
void lib_image::oe_filters(
   unsigned long                                        n_ori,
   double                                               sigma,
   auto_collection< matrix<>, array_list< matrix<> > >& filters_even,
   auto_collection< matrix<>, array_list< matrix<> > >& filters_odd)
{
   /* runnable class for creating oe filters */
   class oe_filters_creator : public runnable {
   public:
      /*
       * Constructor.
       */
      explicit oe_filters_creator(
         unsigned long                                        n_ori,
         double                                               sigma,
         bool                                                 even_or_odd,
         auto_collection< matrix<>, array_list< matrix<> > >& filters)
       : _n_ori(n_ori),
         _sigma(sigma),
         _even_or_odd(even_or_odd),
         _filters(filters)
      { }

      /*
       * Destructor.
       */
      virtual ~oe_filters_creator() { /* do nothing */ }

      /*
       * Create the filter set.
       */
      virtual void run() {
         _filters = _even_or_odd ?
            lib_image::oe_filters_odd(_n_ori, _sigma)
          : lib_image::oe_filters_even(_n_ori, _sigma);
      }

   protected:
      unsigned long                                        _n_ori;
      double                                               _sigma;
      bool                                                 _even_or_odd;
      auto_collection< matrix<>, array_list< matrix<> > >& _filters;
   };
   /* create oe filters */
   oe_filters_creator f_even_creator(n_ori, sigma, false, filters_even);
   oe_filters_creator f_odd_creator(n_ori, sigma, true, filters_odd);
   child_thread::run(f_even_creator, f_odd_creator);
}

auto_collection< matrix<>, array_list< matrix<> > > lib_image::oe_filters_even(
   unsigned long n_ori,
   double        sigma)
{
   return lib_image::gaussian_filters(n_ori, sigma, 2, false, 3.0);
}

auto_collection< matrix<>, array_list< matrix<> > > lib_image::oe_filters_odd(
   unsigned long n_ori,
   double        sigma)
{
   return lib_image::gaussian_filters(n_ori, sigma, 2, true, 3.0);
}

matrix<> lib_image::gaussian_cs_2D(
   double        sigma_x, 
   double        sigma_y,
   double        ori,
   double        scale_factor,
   unsigned long support_x,
   unsigned long support_y)
{
   /* compute standard deviation for center kernel */
   double sigma_x_c = sigma_x / scale_factor;
   double sigma_y_c = sigma_y / scale_factor;
   /* compute center and surround kernels */
   matrix<> m_center = lib_image::gaussian_2D(
      sigma_x_c, sigma_y_c, ori, 0, false, support_x, support_y
   );
   matrix<> m_surround = lib_image::gaussian_2D(
      sigma_x, sigma_y, ori, 0, false, support_x, support_y
   );
   /* compute center-surround kernel */
   matrix<> m = m_surround - m_center;
   /* make zero mean and unit L1 norm */
   m -= mean(m);
   m /= sum(abs(m));
   return m;
}

auto_collection< matrix<>, array_list< matrix<> > > lib_image::gaussian_filters(
   unsigned long        n_ori,
   double               sigma,
   unsigned int         deriv,
   bool                 hlbrt,
   double               elongation)
{
   array<double> oris = lib_image::standard_filter_orientations(n_ori);
   return lib_image::gaussian_filters(oris, sigma, deriv, hlbrt, elongation);
}

array<double> lib_image::standard_filter_orientations(unsigned long n_ori) {
   array<double> oris(n_ori);
   double ori = 0;
   double ori_step = (n_ori > 0) ? (M_PIl / static_cast<double>(n_ori)) : 0;
   for (unsigned long n = 0; n < n_ori; n++, ori += ori_step)
      oris[n] = ori;
   return oris;
}

//Tells someone to create filters
auto_collection< matrix<>, array_list< matrix<> > > lib_image::gaussian_filters(
   const array<double>& oris,
   double               sigma,
   unsigned int         deriv,
   bool                 hlbrt,
   double               elongation)
{
   /* compute support from sigma */
   unsigned long support = static_cast<unsigned long>(math::ceil(3*sigma));
   double sigma_x = sigma;
   double sigma_y = sigma_x/elongation;
   /* allocate collection to hold filters */
   auto_collection< matrix<>, array_list< matrix<> > > filters(
      new array_list< matrix<> >()
   );
   /* allocate collection of filter creators */
   auto_collection< runnable, list<runnable> > filter_creators(
      new list<runnable>()
   );
   /* setup filter creators */
   unsigned long n_ori = oris.size();
   for (unsigned long n = 0; n < n_ori; n++) {
      auto_ptr< matrix<> > f(new matrix<>());
      auto_ptr<filter_creator> f_creator(
         new filter_creator(
            sigma_x, sigma_y, oris[n], deriv, hlbrt, support, support, *f
         )
      );
      filters->add(*f);
      f.release();
      filter_creators->add(*f_creator);
      f_creator.release();
   }
   /* create filters */
   child_thread::run(*filter_creators);
   return filters;
}

class filter_creator : public runnable {
public:
   /*
    * Constructor.
    */
   explicit filter_creator(
      double        sigma_x,     /* sigma x */
      double        sigma_y,     /* sigma y */
      double        ori,         /* orientation */
      unsigned int  deriv,       /* derivative in y-direction (0, 1, or 2) */
      bool          hlbrt,       /* take hilbert transform in y-direction? */
      unsigned long support_x,   /* x support */
      unsigned long support_y,   /* y support */
      matrix<>&     f)           /* output filter matrix */
    : _sigma_x(sigma_x),
      _sigma_y(sigma_y),
      _ori(ori),
      _deriv(deriv),
      _hlbrt(hlbrt),
      _support_x(support_x),
      _support_y(support_y),
      _f(f)
   { }

   /*
    * Destructor.
    */
   virtual ~filter_creator() { /* do nothing */ }

   /*
    * Create the filter.
    */
   virtual void run() {
      _f = lib_image::gaussian_2D(
         _sigma_x, _sigma_y, _ori, _deriv, _hlbrt, _support_x, _support_y
      );
   }
   
protected:
   double        _sigma_x;       /* sigma x */
   double        _sigma_y;       /* sigma y */
   double        _ori;           /* orientation */
   unsigned int  _deriv;         /* derivative in y-direction (0, 1, or 2) */
   bool          _hlbrt;         /* take hilbert transform in y-direction? */
   unsigned long _support_x;     /* x support */
   unsigned long _support_y;     /* y support */
   matrix<>&     _f;             /* output filter matrix */
};


/*
* 1. textons_4real asks texton_filters for two families of filters (large &
 * small), with params n_ori, sigma_sm and n_ori, sigma_lg
 *
 * 2. texton_filters asks oe_filters for two families of filters (even & odd)
 * (now 4 families) with the same params n_ori and sigma (lg or sm). It also
 * asks gaussian_cs_2D for filter with sigma, sigma, 0, M_SQRT2l, support,
 * support, where support = ceil(3*sigma).
 *
 * 3a. oe_filters ask oe_filters_odd and oe_filters_even for corresponding odd
 * and even filters, sigma n_ori staying the same. Then oe_filters_odd and
 * oe_filters_even ask for gaussian_filters to create filters with parameters
 * n_ori, sigma, 2, false, 3.0 and
 * n_ori, sigma, 2, true, 3.0
 *
 * 4a. gaussian_filters finds n_ori directions, shortens sigma_y = sigma/3.0,
 * finds support = ceil(3*sigma) and asks filter creator for filter with
 * sigma, sigma_y, oris[n], 2, false, support, support and
 * sigma, sigma_y, oris[n], 2, true, support, support
 *
 * 5a. filter_creator asks gaussian_2D to create filters with the same
 * parameters as above
 *
 * 3b. gaussian_cs_2D calculates sigma_x_c = sigma_y_c = sigma/M_SQRT2l and
 * asks for gaussian_2D for a filters with 
 * sigma_x_c, sigma_y_c, 0, 0, false, support, support and 
 * sigma, sigma, 0, 0, false, support, support.
 * Returns then their difference
 *
 * 6a, 4b. gaussian_2D calculates stuff and asks gaussian for 1D filters, which
 * it then multiplies and returns. Total of 2*n_ori + 1 filter.
 * 
 * 7. gaussian does what it is told.
 */

matrix<> lib_image::gaussian_2D(
   double        sigma_x, 
   double        sigma_y,
   double        ori,
   unsigned int  deriv,
   bool          hlbrt,
   unsigned long support_x,
   unsigned long support_y)
{
   /* compute size of larger grid for axis-aligned gaussian   */
   /* (reverse rotate corners of bounding box by orientation) */
   unsigned long support_x_rot = support_x_rotated(support_x, support_y, -ori);
   unsigned long support_y_rot = support_y_rotated(support_x, support_y, -ori);
   /* compute 1D kernels */
   matrix<> mx = lib_image::gaussian(sigma_x, 0,     false, support_x_rot);
   matrix<> my = lib_image::gaussian(sigma_y, deriv, hlbrt, support_y_rot);
   /* compute 2D kernel from product of 1D kernels */
   matrix<> m(mx._size, my._size);
   unsigned long n = 0;
   for (unsigned long n_x = 0; n_x < mx._size; n_x++) {
      for (unsigned long n_y = 0; n_y < my._size; n_y++) {
         m._data[n] = mx._data[n_x] * my._data[n_y];
         n++;
      }
   }
   /* rotate 2D kernel by orientation */
   m = lib_image::rotate_2D_crop(m, ori, 2*support_x + 1, 2*support_y + 1);
   /* make zero mean (if returning derivative) */
   if (deriv > 0)
      m -= mean(m);
   /* make unit L1 norm */
   m /= sum(abs(m));
   return m;
}

double support_x_rotated(double support_x, double support_y, double ori) {
   const double sx_cos_ori = support_x * math::cos(ori);
   const double sy_sin_ori = support_y * math::sin(ori);
   double x0_mag = math::abs(sx_cos_ori - sy_sin_ori);
   double x1_mag = math::abs(sx_cos_ori + sy_sin_ori);
   return (x0_mag > x1_mag) ? x0_mag : x1_mag;
}

double support_y_rotated(double support_x, double support_y, double ori) {
   const double sx_sin_ori = support_x * math::sin(ori);
   const double sy_cos_ori = support_y * math::cos(ori);
   double y0_mag = math::abs(sx_sin_ori - sy_cos_ori);
   double y1_mag = math::abs(sx_sin_ori + sy_cos_ori);
   return (y0_mag > y1_mag) ? y0_mag : y1_mag;
}

matrix<> lib_image::gaussian(
   double sigma, unsigned int deriv, bool hlbrt)
{
   unsigned long support = static_cast<unsigned long>(math::ceil(3*sigma));
   return lib_image::gaussian(sigma, deriv, hlbrt, support);
}

matrix<> lib_image::gaussian(
   double sigma, unsigned int deriv, bool hlbrt, unsigned long support)
{
   /* enlarge support so that hilbert transform can be done efficiently */
   unsigned long support_big = support;
   if (hlbrt) {
      support_big = 1;
      unsigned long temp = support;
      while (temp > 0) {
         support_big *= 2;
         temp /= 2;
      }
   }
   /* compute constants */
   const double sigma2_inv = double(1)/(sigma*sigma);
   const double neg_two_sigma2_inv = double(-0.5)*sigma2_inv;
   /* compute gaussian (or gaussian derivative) */
   unsigned long size = 2*support_big + 1;
   matrix<> m(size, 1);
   double x = -(static_cast<double>(support_big)); 
   if (deriv == 0) {
      /* compute guassian */
      for (unsigned long n = 0; n < size; n++, x++)
         m._data[n] = math::exp(x*x*neg_two_sigma2_inv);
   } else if (deriv == 1) {
      /* compute gaussian first derivative */
      for (unsigned long n = 0; n < size; n++, x++)
         m._data[n] = math::exp(x*x*neg_two_sigma2_inv) * (-x);
   } else if (deriv == 2) {
      /* compute gaussian second derivative */
      for (unsigned long n = 0; n < size; n++, x++) {
         double x2 = x * x;
         m._data[n] = math::exp(x2*neg_two_sigma2_inv) * (x2*sigma2_inv - 1);
      }
   } else {
      throw ex_invalid_argument("only derivatives 0, 1, 2 supported");
   }
   /* take hilbert transform (if requested) */
   if (hlbrt) {
      /* grab power of two sized submatrix (ignore last element) */
      m._size--;
      m._dims[0]--;
      /* grab desired submatrix after hilbert transform */
      array<unsigned long> start(2);
      array<unsigned long> end(2);
      start[0] = support_big - support;
      end[0] = start[0] + support + support;
      m = (lib_signal::hilbert(m)).submatrix(start, end);
   }
   /* make zero mean (if returning derivative) */
   if (deriv > 0)
      m -= mean(m);
   /* make unit L1 norm */
   m /= sum(abs(m));
   return m;
}




//Filters and clusters
matrix<unsigned long> lib_image::textons(
   const matrix<>&                                      m,
   const collection< matrix<> >&                        filters,
   auto_collection< matrix<>, array_list< matrix<> > >& textons,
   unsigned long                                        K,
   unsigned long                                        max_iter,
   double                                               subsampling)
}







void pb_parts_final_selected(cv::Mat image, cv::Mat & textons, cv::Mat & bg, cv::Mat & cga, cv
                             auto_collection< matrix<>, array_list< matrix<> > > & bg_r3,
                             auto_collection< matrix<>, array_list< matrix<> > > & bg_r5,
                             auto_collection< matrix<>, array_list< matrix<> > > & bg_r10,
                             auto_collection< matrix<>, array_list< matrix<> > > & cga_r5,
                             auto_collection< matrix<>, array_list< matrix<> > > & cga_r10,
                             auto_collection< matrix<>, array_list< matrix<> > > & cga_r20,
                             auto_collection< matrix<>, array_list< matrix<> > > & cgb_r5,
                             auto_collection< matrix<>, array_list< matrix<> > > & cgb_r10,
                             auto_collection< matrix<>, array_list< matrix<> > > & cgb_r20,
                             auto_collection< matrix<>, array_list< matrix<> > > & tg_r5,
                             auto_collection< matrix<>, array_list< matrix<> > > & tg_r10,
                             auto_collection< matrix<>, array_list< matrix<> > > & tg_r20)
{
      /* parameters - binning and smoothing */
      unsigned long n_ori       = 8;                     /* number of orientations */
      unsigned long num_L_bins  = 25;                    /* # bins for bg */
      unsigned long num_a_bins  = 25;                    /* # bins for cg_a */
      unsigned long num_b_bins  = 25;                    /* # bins for cg_b */
      double bg_smooth_sigma    = 0.1;                   /* bg histogram smoothing sigma */
      double cg_smooth_sigma    = 0.05;                  /* cg histogram smoothing sigma */
      unsigned long border      = 30;                    /* border pixels */

      /* parameters - radii */
      unsigned long n_bg = 3;
      unsigned long n_cg = 3;
      unsigned long n_tg = 3;
      unsigned long r_bg[] = { 3, 5, 10 };
      unsigned long r_cg[] = { 5, 10, 20 };
      unsigned long r_tg[] = { 5, 10, 20 };

      /* compute bg histogram smoothing kernel */
      matrix<> bg_smooth_kernel =
         lib_image::gaussian(bg_smooth_sigma*num_L_bins);
      matrix<> cga_smooth_kernel =
         lib_image::gaussian(cg_smooth_sigma*num_a_bins);
      matrix<> cgb_smooth_kernel =
         lib_image::gaussian(cg_smooth_sigma*num_b_bins);

      /* mirror border */
      L = lib_image::border_mirror_2D(L, border);
      a = lib_image::border_mirror_2D(a, border);
      b = lib_image::border_mirror_2D(b, border);

      /* gamma correct */
      lib_image::rgb_gamma_correct(L,a,b,2.5);

      /* convert to Lab */
      lib_image::rgb_to_lab(L,a,b);
      lib_image::lab_normalize(L,a,b);

      /* quantize color channels  */
      matrix<unsigned long> Lq = lib_image::quantize_values(L, num_L_bins);
      matrix<unsigned long> aq = lib_image::quantize_values(a, num_a_bins);
      matrix<unsigned long> bq = lib_image::quantize_values(b, num_b_bins);


      /* compute bg at each radius */
      cout << "computing bg's\n";
      bg_r3 = lib_image::hist_gradient_2D(Lq, r_bg[0], n_ori, bg_smooth_kernel);
      bg_r5 = lib_image::hist_gradient_2D(Lq, r_bg[1], n_ori, bg_smooth_kernel);
      bg_r10 = lib_image::hist_gradient_2D(Lq, r_bg[2], n_ori, bg_smooth_kernel);

      for (unsigned long n = 0; n < n_ori; n++)
      {
          (*bg_r3)[n] = lib_image::border_trim_2D((*bg_r3)[n], border);
          (*bg_r5)[n] = lib_image::border_trim_2D((*bg_r5)[n], border);
          (*bg_r10)[n] = lib_image::border_trim_2D((*bg_r10)[n], border);
      }


      /* compute cga at each radius */
      cout << "computing cga's\n";
      cga_r5 = lib_image::hist_gradient_2D(aq, r_cg[0], n_ori, cga_smooth_kernel);
      cga_r10 = lib_image::hist_gradient_2D(aq, r_cg[1], n_ori, cga_smooth_kernel);
      cga_r20 = lib_image::hist_gradient_2D(aq, r_cg[2], n_ori, cga_smooth_kernel);

      for (unsigned long n = 0; n < n_ori; n++)
      {
          (*cga_r5)[n] = lib_image::border_trim_2D((*cga_r5)[n], border);
          (*cga_r10)[n] = lib_image::border_trim_2D((*cga_r10)[n], border);
          (*cga_r20)[n] = lib_image::border_trim_2D((*cga_r20)[n], border);
      }

      /* compute cgb at each radius */
      cout << "computing cgb's\n";
      cgb_r5 = lib_image::hist_gradient_2D(bq, r_cg[0], n_ori, cgb_smooth_kernel);
      cgb_r10 = lib_image::hist_gradient_2D(bq, r_cg[1], n_ori, cgb_smooth_kernel);
      cgb_r20 = lib_image::hist_gradient_2D(bq, r_cg[2], n_ori, cgb_smooth_kernel);

      for (unsigned long n = 0; n < n_ori; n++)
      {
          (*cgb_r5)[n] = lib_image::border_trim_2D((*cgb_r5)[n], border);
          (*cgb_r10)[n] = lib_image::border_trim_2D((*cgb_r10)[n], border);
          (*cgb_r20)[n] = lib_image::border_trim_2D((*cgb_r20)[n], border);
      }

      /* compute tg at each radius */
      cout << "computing tg's\n";
      tg_r5 = lib_image::hist_gradient_2D(t_assign, r_tg[0], n_ori);
      tg_r10 = lib_image::hist_gradient_2D(t_assign, r_tg[1], n_ori);
      tg_r20 = lib_image::hist_gradient_2D(t_assign, r_tg[2], n_ori);

      for (unsigned long n = 0; n < n_ori; n++)
      {
          (*tg_r5)[n] = lib_image::border_trim_2D((*tg_r5)[n], border);
          (*tg_r10)[n] = lib_image::border_trim_2D((*tg_r10)[n], border);
          (*tg_r20)[n] = lib_image::border_trim_2D((*tg_r20)[n], border);
      }
}

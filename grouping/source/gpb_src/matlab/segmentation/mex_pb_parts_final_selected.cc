/*
 * Pb. 
 */
#include "collections/pointers/auto_collection.hh"
#include "collections/array_list.hh"
#include "io/streams/cout.hh"
#include "io/streams/iomanip.hh"
#include "io/streams/ios.hh"
#include "lang/array.hh"
#include "lang/exceptions/exception.hh"
#include "lang/exceptions/ex_index_out_of_bounds.hh"
#include "lang/pointers/auto_ptr.hh"
#include "math/libraries/lib_image.hh"
#include "math/math.hh"
#include "math/matrices/matrix.hh"

#include "concurrent/threads/thread.hh"

#include <time.h>
#include <mex.h>

using collections::pointers::auto_collection;
using collections::array_list;
using io::streams::cout;
using io::streams::ios;
using io::streams::iomanip::setiosflags;
using io::streams::iomanip::setw;
using lang::array;
using lang::exceptions::exception;
using lang::exceptions::ex_index_out_of_bounds;
using lang::pointers::auto_ptr;
using math::libraries::lib_image;
using math::matrices::matrix;

using concurrent::threads::thread;

/********************************** 
 * Matlab matrix conversion routines.
 **********************************/

/*
 * Get a string from an mxArray.
 */
char *mexGetString(const mxArray *arr) {
   char *string;
   int buflen;
   buflen = (mxGetM(arr) * mxGetN(arr) * sizeof(mxChar)) + 1;
   string = new char[buflen];
   mxGetString(arr, string, buflen);
   return string;
}

/*
 * Create a single element Matlab double matrix.
 */
mxArray* mxCreateScalarDouble(double value) {
   mxArray* pa = mxCreateDoubleMatrix(1, 1, mxREAL);
   *mxGetPr(pa) = value;
   return pa;
}

/* 
 * Convert an mxArray to a matrix.
 */
matrix<> to_matrix(const mxArray *a) {
   unsigned long mrows = static_cast<unsigned long>(mxGetM(a));
   unsigned long ncols = static_cast<unsigned long>(mxGetN(a));
   double *data = mxGetPr(a);
   matrix<> m(mrows, ncols);
   for (unsigned long r = 0; r < mrows; r++) {
      for (unsigned long c = 0; c < ncols; c++) {
         m(r,c) = data[(c*mrows) + r];
      }
   }
   return m;
}
   
/*
 * Convert a 2D matrix to an mxArray.
 */
mxArray* to_mxArray(const matrix<>& m) {
   unsigned long mrows = m.size(0);
   unsigned long ncols = m.size(1);
   mxArray *a = mxCreateDoubleMatrix(
      static_cast<int>(mrows),
      static_cast<int>(ncols),
      mxREAL
   );
   double *data = mxGetPr(a);
   for (unsigned long r = 0; r < mrows; r++) {
      for (unsigned long c = 0; c < ncols; c++) {
         data[(c*mrows) + r] = m(r,c);
      }
   }
   return a;
}

/*
 * Suppress border responses.
 */
void suppress_border(matrix<>& m, unsigned long r) {
   unsigned long size_x = m.size(0);
   unsigned long size_y = m.size(1);
   for (unsigned long x = 0; x < size_x; x++) {
      for (unsigned long y = 0; y < size_y; y++) {
         if (((x < r) || ((x + r) >= size_x)) ||
             ((y < r) || ((y + r) >= size_y)))
            m(x,y) = 0;
      }
   }
}

/*
 * cpp interface
 */

void pb_parts_final_selected(matrix<> L, matrix<> a, matrix <> b,
                             matrix<> & textons,
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
      double sigma_tg_filt_sm   = 2.0;                   /* sigma for small tg filters */
      double sigma_tg_filt_lg   = math::sqrt(2) * 2.0;   /* sigma for large tg filters */

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

      /* convert to grayscale */
      matrix<> gray = lib_image::grayscale(L,a,b);

      /* gamma correct */
      lib_image::rgb_gamma_correct(L,a,b,2.5);

      /* convert to Lab */
      lib_image::rgb_to_lab(L,a,b);
      lib_image::lab_normalize(L,a,b);

      /* quantize color channels  */
      matrix<unsigned long> Lq = lib_image::quantize_values(L, num_L_bins);
      matrix<unsigned long> aq = lib_image::quantize_values(a, num_a_bins);
      matrix<unsigned long> bq = lib_image::quantize_values(b, num_b_bins);

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


/*
 * Matlab interface.
 */
void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{

    /* parameters - binning and smoothing */
    unsigned long n_ori       = 8;                     /* number of orientations */

    /* get image */
    matrix<> L = to_matrix(prhs[0]);
    matrix<> a = to_matrix(prhs[1]);
    matrix<> b = to_matrix(prhs[2]);

    /* init vars */
    matrix<> textons;
    auto_collection< matrix<>, array_list< matrix<> > >  bg_r3, bg_r5,
        bg_r10, cga_r5, cga_r10, cga_r20, cgb_r5, cgb_r10, cgb_r20, tg_r5, tg_r10,
        tg_r20;

    pb_parts_final_selected(L, a, b, textons, bg_r3, bg_r5, bg_r10, cga_r5,
            cga_r10, cga_r20, cgb_r5, cgb_r10, cgb_r20, tg_r5, tg_r10, tg_r20);

    /*return*/
    plhs[0] = to_mxArray(textons);
    unsigned long count = 1;

    mxArray* m = mxCreateCellMatrix(static_cast<int>(n_ori), 1);
    for (unsigned long n = 0; n < n_ori; n++)
        mxSetCell(m, static_cast<int>(n), to_mxArray((*bg_r3)[n]));
    plhs[count++] = m;
    mxArray* m1 = mxCreateCellMatrix(static_cast<int>(n_ori), 1);
    for (unsigned long n = 0; n < n_ori; n++)
        mxSetCell(m1, static_cast<int>(n), to_mxArray((*bg_r5)[n]));
    plhs[count++] = m1;
    mxArray* m2 = mxCreateCellMatrix(static_cast<int>(n_ori), 1);
    for (unsigned long n = 0; n < n_ori; n++)
        mxSetCell(m2, static_cast<int>(n), to_mxArray((*bg_r10)[n]));
    plhs[count++] = m2;
    mxArray* m3 = mxCreateCellMatrix(static_cast<int>(n_ori), 1);
    for (unsigned long n = 0; n < n_ori; n++)
        mxSetCell(m3, static_cast<int>(n), to_mxArray((*cga_r5)[n]));
    plhs[count++] = m3;
    mxArray* m4 = mxCreateCellMatrix(static_cast<int>(n_ori), 1);
    for (unsigned long n = 0; n < n_ori; n++)
        mxSetCell(m4, static_cast<int>(n), to_mxArray((*cga_r10)[n]));
    plhs[count++] = m4;
    mxArray* m5 = mxCreateCellMatrix(static_cast<int>(n_ori), 1);
    for (unsigned long n = 0; n < n_ori; n++)
        mxSetCell(m5, static_cast<int>(n), to_mxArray((*cga_r20)[n]));
    plhs[count++] = m5;
    mxArray* m6 = mxCreateCellMatrix(static_cast<int>(n_ori), 1);
    for (unsigned long n = 0; n < n_ori; n++)
        mxSetCell(m6, static_cast<int>(n), to_mxArray((*cgb_r5)[n]));
    plhs[count++] = m6;
    mxArray* m7 = mxCreateCellMatrix(static_cast<int>(n_ori), 1);
    for (unsigned long n = 0; n < n_ori; n++)
        mxSetCell(m7, static_cast<int>(n), to_mxArray((*cgb_r10)[n]));
    plhs[count++] = m7;
    mxArray* m8 = mxCreateCellMatrix(static_cast<int>(n_ori), 1);
    for (unsigned long n = 0; n < n_ori; n++)
        mxSetCell(m8, static_cast<int>(n), to_mxArray((*cgb_r20)[n]));
    plhs[count++] = m8;
    mxArray* m9 = mxCreateCellMatrix(static_cast<int>(n_ori), 1);
    for (unsigned long n = 0; n < n_ori; n++)
        mxSetCell(m9, static_cast<int>(n), to_mxArray((*tg_r5)[n]));
    plhs[count++] = m9;
    mxArray* m10 = mxCreateCellMatrix(static_cast<int>(n_ori), 1);
    for (unsigned long n = 0; n < n_ori; n++)
        mxSetCell(m10, static_cast<int>(n), to_mxArray((*tg_r10)[n]));
    plhs[count++] = m10;
    mxArray* m11 = mxCreateCellMatrix(static_cast<int>(n_ori), 1);
    for (unsigned long n = 0; n < n_ori; n++)
        mxSetCell(m11, static_cast<int>(n), to_mxArray((*tg_r20)[n]));
    plhs[count++] = m11;
}

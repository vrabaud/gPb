#include "lib_image.h"
#include <opencv2/core/core.hpp>


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

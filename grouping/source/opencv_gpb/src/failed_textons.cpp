#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
//for debuggin:
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;

void HilbertTransform( const Mat & input, Mat & output )
{
    CV_Assert( input.depth() == CV_32F | input.depth() == CV_64F );

    // Hilbert transform is made in frequency domain
    Mat f_transform;
    dft(input, f_transform);

    // Multilply positive frequences by -i and negative frequences by i.
    // The following assumes compact fft output format, so it only works
    // for real valued signals
    int n = input.rows;
    float tf;
    double td;

    if (input.depth() == CV_32F)
    {
        for (int i = 2; i < n; i+=2)
        {
            tf = -f_transform.at<float>(i);
            f_transform.at<float>(i) = (f_transform.at<float>(i-1));
            f_transform.at<float>(i-1) = tf;
        }
        // zero out zero frequency
        f_transform.at<float>(0) = 0;

        // and Nyquist frequency, if it exists.
        if ( n%2 == 0 )
        {
            f_transform.at<float>(n-1) = 0;
        }
    }
    else
    {
        for (int i = 2; i < n; i+=2)
        {
            td = -f_transform.at<double>(i);
            f_transform.at<double>(i) = (f_transform.at<double>(i-1));
            f_transform.at<double>(i-1) = td;
        }
        // zero out zero frequency
        f_transform.at<double>(0) = 0;

        // and Nyquist frequency, if it exists.
        if ( n%2 == 0 )
        {
            f_transform.at<double>(n-1) = 0;
        }
    }

    idft(f_transform, output, DFT_SCALE);
}

/* 
 * Returns gaussian kernel or its derivative. Returned kernel is of unit L1
 * norm and, if returning derivative, 0 mean.
 */
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
            t *= (-x);
        }
        if (derivative == 2)
        {
            t *= x*x/(sigma*sigma) - 1;
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

    //set L1 norm to 1
    //normalize(kernel, kernel, 1, 0, NORM_L1);

    return kernel;
}

/* 
 * Returns 2d kernel made of gaussians, derivatives of gaussians or
 * hilbert transforms of derivatives of gaussians. Returned kernel is of
 * unit L1 norm.
 */
Mat getGaussian2D( int xsupport, int ysupport,
                   double xsigma, double ysigma,
                   int yderivative, bool yhilbert, int ktype )
{
    Mat xgaussian = getGaussianKernel2(xsupport, xsigma, 0, ktype);
    Mat ygaussian;
    if ( yhilbert == true )
    {
        // get nearest power of two
        int enlarged = 1;
        for ( int i = ysupport; i >= 1; i /= 2 )
            enlarged *= 2;
        enlarged += 1;

        ygaussian = getGaussianKernel2(enlarged, ysigma, yderivative, ktype);

        // drop last element
        ygaussian = ygaussian.Mat::rowRange(0,enlarged-1);
        HilbertTransform(ygaussian,ygaussian);

        // extract gaussian of original support
        ygaussian = ygaussian.Mat::rowRange((enlarged-ysupport)/2, (enlarged+ysupport)/2);
    }
    else
    {
        ygaussian = getGaussianKernel2(ysupport, ysigma, yderivative, ktype);
    }

    normalize(xgaussian, xgaussian, 1, 0, NORM_L1);
    normalize(ygaussian, ygaussian, 1, 0, NORM_L1);
    return xgaussian*ygaussian.t(); 
}

// Loads filtered images from original gPb implementation for testing the
// differences between clusterers. Images are loaded to format that
// kmeans clusterer uses
Mat readFiltered()
{
    std::ifstream input("filterResponses");
    std::string line;
    Mat ksamples(200*300, 34, CV_32F);
    float temp;

    for ( int k = 0; k < 34; k++ )
    {
        for ( int i = 0; i < 300; i++ )
        {
            getline(input, line);
            std::istringstream is(line);
            for ( int j = 0; j < 200; j++ )
            {
                is >> temp;
                ksamples.at<float>(i*j, k) = temp;
            }
        }
    }
    input.close();
    return ksamples;
}

// Loads filters from original gPb implementation for testing the
// differences between filter creation.
vector<Mat> readFilters()
{
    std::ifstream input("filters");
    std::string line;
    Mat small(13, 13, CV_32F);
    Mat large(19, 19, CV_32F);
    vector<Mat> filters;
    for ( int i = 0; i < 17; i++ )
        filters.push_back(Mat::zeros(13, 13, CV_32F));
    for ( int i = 0; i < 17; i++ )
        filters.push_back(Mat::zeros(19, 19, CV_32F));

    float temp;

    for ( int k = 0; k < 17; k++ )
    {
        for ( int j = 0; j < 13; j++ )
        {
            getline( input, line );
            std::istringstream is( line );
            for (int i = 0; i < 13; i++ )
            {
                is >> temp;
                filters[k].at<float>(j,i) = temp;
            }
        }
    }

    for ( int k = 0; k < 17; k++ )
    {
        for ( int j = 0; j < 19; j++ )
        {
            getline( input, line );
            std::istringstream is( line );
            for (int i = 0; i < 19; i++ )
            {
                is >> temp;
                filters[17+k].at<float>(j,i) = temp;
            }
        }
    }

    return filters;
}

// Enlarges small image.
Mat enlarged( Mat & input )
{
    int k = input.rows;
    int l = 30;
    Mat larger = Mat::zeros(k, k*l, CV_32F);
    for ( int i = 0; i < k; i++ )
    {
        for ( int j = i*l; j < i*l + l; j++ )
        {
            larger.at<float>(i,j) = 1;
        }
    }

    return larger.t()*input*larger;
}

void textons(const Mat & image, Mat & texts, bool old_filters = false)
{

    int support               = 13;
    unsigned long n_ori       = 8;                     // number of orientations
    double sigma_sm           = 2.0;                   // sigma for small tg filters
    double sigma_lg           = 2.0*std::sqrt(2);      // sigma for large tg filters
    int border = support/2;
    int rows = image.rows + 2*border;
    int cols = image.cols + 2*border;

    // convert to grayscale
    Mat gray(image.rows, image.cols, CV_8U);
    cvtColor(image, gray, CV_RGB2GRAY);

    // add border
    copyMakeBorder(gray, gray, border, border, border, border, BORDER_REFLECT);

    // compute non rotated versions of gaussian derivatives
    vector<Mat> nr_gaussians;
    nr_gaussians.push_back( getGaussian2D(support, support, sigma_sm, sigma_sm/3.0, 2, true, CV_32F) );
    nr_gaussians.push_back( getGaussian2D(support, support, sigma_sm, sigma_sm/3.0, 2, false, CV_32F) );
    nr_gaussians.push_back( getGaussian2D(support, support, sigma_lg, sigma_lg/3.0, 2, true, CV_32F) );
    nr_gaussians.push_back( getGaussian2D(support, support, sigma_lg, sigma_lg/3.0, 2, false, CV_32F) );

    // compute rotated versions of gaussian derivatives
    vector<Mat> filters(4*n_ori+2, Mat::zeros(rows, cols, CV_32F)) ;
    double angle;
    Point2f center(support/2, support/2);
    Mat rotated;
    Size out_size = nr_gaussians[0].size();

    for ( int ori = 0; ori < n_ori; ori++ )
    {
        angle = (180.0/n_ori)*ori;
        Mat rotation = getRotationMatrix2D(center, angle, 1.0);

        for ( int k = 0; k < 4; k++ )
        {
            warpAffine(nr_gaussians[k], filters[ori+n_ori*k], rotation, out_size);
            filters[ori+n_ori*k] = filters[ori+n_ori*k] - mean(filters[ori+n_ori*k]);
            normalize(filters[ori+n_ori*k], filters[ori+n_ori*k], 1, 0, NORM_L1);
        }
    }

    //compute centre surround filters
    filters[4*n_ori] = getGaussian2D(support, support, sigma_sm, sigma_sm, 0, false, CV_32F) -
        getGaussian2D(support, support, sigma_sm/std::sqrt(2), sigma_sm/std::sqrt(2), 0, false, CV_32F);
    filters[4*n_ori+1] = getGaussian2D(support, support, sigma_lg, sigma_lg, 0, false, CV_32F) -
        getGaussian2D(support, support, sigma_lg/std::sqrt(2), sigma_lg/std::sqrt(2), 0, false, CV_32F);

    // if old_filters option is selected, load old filters
    if ( old_filters = true )
    {
        filters = readFilters();
    }

    // display filters if needed
    //for ( int f = 0; f < 4*n_ori + 2; f++ )
    //{
        //imshow("bla", enlarged(filters[f]));
        //waitKey(0);
    //}

    // filter the image and store results for clustering
    Mat blurred;
    Mat ksamples(rows*cols, 4*n_ori + 2, CV_32F);

    for ( int f = 0; f < 4*n_ori + 2; f++ )
    {
        filter2D(gray, blurred, CV_32F, filters[f], Point(-1,-1), 0, BORDER_REFLECT);
        //imshow("bla", blurred);
        //waitKey(0);

        for ( int i = 0; i < rows*cols; i++ )
        {
            ksamples.at<float>(i,f) = blurred.at<int>((i/cols),(i%cols))/255.0;
        }
    }

    // cluster filter responses
    Mat labels;
    int K = 32;
    kmeans(ksamples, K, labels, 
           TermCriteria( CV_TERMCRIT_EPS, 10, 0.0001 ),
           3, KMEANS_PP_CENTERS);

    Mat clustered = Mat(rows, cols, CV_32F);

    for(int i = 0; i<rows*cols; i++) 
    {
        clustered.at<float>(i/cols, i%cols) = (float)labels.at<int>(0,i);
    }

    clustered.convertTo(clustered, CV_8U);
    texts = clustered(Range(support/2, rows - support/2), Range(support/2, cols - support/2));
}


int main()
{
    Mat image = imread("../../../data/101087.jpg", CV_LOAD_IMAGE_COLOR);
    Mat texts(image.rows, image.cols, CV_8U);
    textons(image, texts, true);
    imshow("", texts*8);
    waitKey(0);
}

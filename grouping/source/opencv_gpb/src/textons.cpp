#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

int sq_derivative(Vec3b x, Vec3b y, int diag)
{
    int sum = 0;
    for ( int i = 0; i < 3; i++ )
    {
        sum += (x[i] - y[i])*(x[i] - y[i]);
    }
    return (diag == 1) ? 2*sum : sum;
}

void textons(Mat & image, Mat & texts)
{
    int TRESH = 20;
    int rows = image.rows;
    int cols = image.cols;
    Mat neighbor(rows*cols, 9, CV_8U);

    for ( int i = 0; i < rows; i++) 
    {
        for ( int j = 0; j < cols; j++) 
        {
            for ( int x = -1; x < 2; x++ )
            {
                for ( int y = -1; y < 2; y++ )
                {
                    if ( 0 <= i + x and i + x <= rows and 0 <= j + y and j + y <= cols )
                    {
                        int sq_de = sq_derivative(image.at<Vec3b>(i, j), 
                                image.at<Vec3b>(i + x, j + y), (x+y)%2);
                        texts.at<uchar>(i, j) += (sq_de <= TRESH) ? 1 : 0;
                    }
                }
            }
        }
    }
}

int main()
{
    Mat image = imread("testnormal.jpg", CV_LOAD_IMAGE_COLOR);
    Mat texts = Mat::zeros(image.rows, image.cols, CV_8U);
    textons(image, texts);
    imshow("", texts);
    waitKey(0);
}

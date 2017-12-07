#ifndef DR_HH
# define DR_HH
# include <opencv/cv.h>
# include <opencv/cvaux.h>
# include <opencv/highgui.h>
# include <opencv/cvwimage.h>

# include <list>
# include <iostream>
# include <time.h>
# include <algorithm>

// Class for Diminished reality purposes.
// Handle images and video streams (online and offline).

class DR
{
	typedef std::vector<cv::Mat> Mat_v;


  public:

    // Policy for the patch copying.
    enum COPY_P
    {
        SIMPLE,
        INTENSITY
    };

    // Constructor with the input and the contour.
    DR(char* mask, char* input, std::string& prefix, COPY_P cp = SIMPLE);
    DR(cv::Mat& mask, cv::Mat& input, std::string& prefix, COPY_P cp = SIMPLE);
    void inpaint();

  private:
	void init();
    // Copy the previous layer of the pyramid.
    void offset_scaling_up();

    // From a binary mask, build the list of points.
    void build_mask();
    void build_mask_coarse();

    // Basic cost comparing patches in the LAB colorspace.
    double cost_bullshit(cv::Point& p, double curr_cost, bool& stop);

    void random_search(cv::Point& p, double& curr_cost);

    void improve(cv::Point p, size_t cpt, double& cost);

    // Copy the patch or raise intensity.
    void patch_copy(cv::Point dst, cv::Point src);

    COPY_P cp_;
    cv::Mat mask_;
    cv::Mat input_;
	std::string prefix_;
    size_t nb_iter_;
    size_t iter_;
    size_t max_scale_;
    size_t max_iter_;
    int scale_iter_;
    int patch_size_;

    // Pyramid attributes.

    Mat_v pyramid_image_;
    Mat_v pyramid_mapping_;
    Mat_v pyramid_mask_;
    Mat_v pyramid_cost_;

    // Size of patches for every layers of the pyramid.
    std::vector<int> pyramid_size_;

	// FIXME: have a matrix for each pyramid.
    std::vector<std::list<cv::Point> > pyramid_target_pixels_;
    cv::Mat res_;
};

#endif

#ifndef TRACKING_HH
# define TRACKING_HH

# include <iostream>
# include <opencv/cv.h>
# include <opencv/cvaux.h>
# include <opencv/highgui.h>
# include <opencv/cvwimage.h>

# include <cfloat>

# define DBL_EPSILON     2.2204460492503131e-016

class Tracking
{
	typedef std::vector<cv::Mat> Mat_v;

	public:
		Tracking(const std::vector<cv::Point>& rough_selection);
		void build_mask(const cv::Mat& frame);
		cv::Mat get_mask();

		// FIXME: Remove me.
		cv::Mat illustration;

	private:
		void selection_refinement(const std::vector<cv::Point>& hull);
		void hull_sampling(const std::vector<cv::Point>& hull, unsigned int factor_nb_points);
		void clustering();
		void dissimilarity_computing();
		cv::Vec3f maximal_deviation(int cluster);
		cv::Vec3f characteristic_computing(const cv::Point& p);
		void init();
		void dissimilarity_scaling_up();

		std::vector<cv::Point> rough_selection_;
		bool debug_;
		cv::Mat mask_;
		// Current frame.
		cv::Mat frame_;
		// Fingerprints.
		std::vector<cv::Point> outline_;
		// Fingerprints characteristics (usually R, G, B).
		std::vector<cv::Vec3f> finger_char_;
		// Dissimilarity map.
		cv::Mat d_map_;
		// Cluster map.
		std::vector<int> cluster_;
		// deviation for each cluster.
		std::vector<cv::Vec3f> v_;
		cv::Vec3f vmax_;
		float tolerance_thres_;
		cv::Mat3f gauss_img_;
		Mat_v pyramid_image_;
		std::vector<std::vector<cv::Point>> pyramid_rough_cont_;
		Mat_v dissimilarity_piramid_;
		unsigned int multiply_contours_;
};

# endif /* !TRACKING_HH */
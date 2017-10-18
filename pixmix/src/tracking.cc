#include "tracking.hh"


Tracking::Tracking(const std::vector<cv::Point>& rough_selection)
	: rough_selection_(rough_selection),
	  outline_(),
	  d_map_(),
	  cluster_(),
	  v_(),
	  vmax_(0, 0, 0),
	  tolerance_thres_(0.95),
	  gauss_img_(),
	  pyramid_image_(5),
	  pyramid_rough_cont_(5),
	  dissimilarity_piramid_(5),
	  multiply_contours_(10)
{
}


void Tracking::hull_sampling(const std::vector<cv::Point>& hull, unsigned int factor_nb_points)
{
	// Double the initial number of points.
	unsigned int nb_points = hull.size() * factor_nb_points;

	outline_ = std::vector<cv::Point>(0);

	float perimeter = cv::arcLength(hull, true);
	float step = perimeter / nb_points;
	cv::circle(illustration, hull[0], 1, cv::Scalar(255, 0, 0));

	// Index array used to link the last and first vertex in a single loop.
	std::vector<unsigned int> index_v(hull.size() + 1);
	for (unsigned int i = 0; i < hull.size(); ++i)
		index_v[i] = i;
	index_v[hull.size()] = 0;

	for (unsigned int k = 1; k < index_v.size(); ++k)
	{
		unsigned int i = index_v[k], prev_i = index_v[k - 1];
		float cur_dis = sqrt(pow(hull[i].x - hull[prev_i].x, 2.0) + pow(hull[i].y - hull[prev_i].y, 2.0));
		unsigned int cur_step = cur_dis / step;
		cv::Point2f dir = cv::Point2f((hull[i].x - hull[prev_i].x) / cur_dis, (hull[i].y - hull[prev_i].y) / cur_dis);
		for (unsigned int j = 1; j <= int(cur_step); ++j)
		{
			cv::Point2f tmp = cv::Point2f(hull[prev_i].x + dir.x * step * j, hull[prev_i].y + dir.y * step * j);
			outline_.push_back(tmp);
		}
	}
	
#ifdef DEBUG_TRACKING
	// Draw the rough selection.
	for (unsigned int i = 1; i < outline_.size(); ++i)
	{
		cv::line(illustration, outline_[i - 1], outline_[i], cv::Scalar(0, 255, 0));
		cv::circle(illustration, outline_[i], 10, cv::Scalar(0, 255, 0), -1);
	}
	cv::circle(illustration, outline_[0], 10, cv::Scalar(0, 255, 0), -1);
	cv::line(illustration, outline_[0], outline_[outline_.size() - 1], cv::Scalar(0, 255, 0));
#endif
}

cv::Vec3f Tracking::characteristic_computing(const cv::Point& p)
{
	/*
	// FIXME: handle corners, not a acceptable case or find something else.
	int kernel_size = 5;
	std::vector<unsigned char> median_r(kernel_size * kernel_size);
	std::vector<unsigned char> median_g(kernel_size * kernel_size);
	std::vector<unsigned char> median_b(kernel_size * kernel_size);
	// R,G and B are faster.
	int x = p.y, y = p.x;

	// Compute the median filter, 3x3 kernel.
	unsigned int k = 0;
	for (int i = -(kernel_size / 2); i <= kernel_size / 2; ++i)
	{
		for (int j = -(kernel_size  / 2); j <= kernel_size / 2; ++j)
		{
			median_b[k] = frame_.at<cv::Vec3b>(x + i, y + j)[0];
			median_g[k] = frame_.at<cv::Vec3b>(x + i, y + j)[1];
			median_r[k] = frame_.at<cv::Vec3b>(x + i, y + j)[2];
			++k;
		}
	}

	auto median_it = median_b.begin() + median_b.size() / 2;
	std::nth_element(median_b.begin(), median_it , median_b.end());
	auto b = *median_it;
		
	median_it = median_g.begin() + median_g.size() / 2;
	std::nth_element(median_g.begin(), median_it , median_g.end());
	auto g = *median_it;
		
	median_it = median_r.begin() + median_r.size() / 2;
	std::nth_element(median_r.begin(), median_it , median_r.end());
	auto r = *median_it;
	
	return cv::Vec3f(b, g, r);
	*/

	cv::Vec3f tmp = gauss_img_.at<cv::Vec3f>(p.y, p.x);
	return tmp;
}

void Tracking::selection_refinement(const std::vector<cv::Point>& hull)
{
	// We want enough fingerprints to correctly segment the object.
	hull_sampling(hull, multiply_contours_);

	std::vector<std::vector<cv::Point>> cont(1);
	cont[0] = outline_;
	cv::drawContours(mask_, cont, 0, cv::Scalar(255), -1);

	// Compute the characteristics.
	finger_char_.resize(outline_.size());
	for (unsigned int i = 0; i < outline_.size(); ++i)
		finger_char_[i] = characteristic_computing(outline_[i]);

	// Clustering.
	unsigned int iteration = 0, max_int = 5;
	std::vector<cv::Vec3f> min_v;
	float min_res = FLT_MAX;
	std::vector<int> min_clu;
	cv::Vec3f min_vmax(FLT_MAX, FLT_MAX, FLT_MAX);
	while (iteration != max_int)
	{
		clustering();
		unsigned int k = 0;
		float res = 0;
		std::cout << "Current maximal deviation " << vmax_ << std::endl;
		for (unsigned int c = 0; c < 3; ++c)
		{
			//if (min_vmax[c] < vmax_[c])
			//	++k;
			res += vmax_[c];
		}
		//if (k == 3)
		if (min_res > res)
		{
			std::cout << "Replace by " << vmax_ << std::endl;
			min_clu = cluster_;
			min_v = v_;
			min_vmax = vmax_;
			iteration = 0;
			min_res = res;
		}
		else
			++iteration;
	}

	cluster_ = min_clu;
	v_ = min_v;
	vmax_ = min_vmax;
	
	// Compute the dissimilarity map.
	dissimilarity_computing();
}

void Tracking::clustering()
{
	cv::RNG rng(12345);
	// FIXME: the structure is not very good, use a vector of vector? After finishing the clustering, init a fixed array.
	// Init with every cluster assigned to 0.
	cluster_ = std::vector<int>(outline_.size() ,-1);
	v_.clear();
	vmax_ = cv::Vec3f(0, 0, 0);

	unsigned int nb_assigned = 1, cur_clu = 1;

	cv::Vec3f tmp = maximal_deviation(-1);
	v_.push_back(tmp);
	
	cv::Scalar col = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));

	// First cluster.
	unsigned int first = rand() % outline_.size();

	for (unsigned int i = 0; i < outline_.size(); ++i)
	{
		if (i != first)
		{
			unsigned int k = 0;
			for (unsigned int c = 0; c < 3; ++c)
			{
				if (abs(finger_char_[i][c] - finger_char_[first][c]) > v_[0][c])
					++k;
			}
			//if (!k)
			if (k != 3)
			{
				cluster_[i] = 0;
				cv::circle(illustration, outline_[i], 1, col, 10);
				++nb_assigned;
			}
		}
	}
	cluster_[first] = 0;
	// Update the first cluster (with less nodes).
	tmp = maximal_deviation(0);
	v_[0] = tmp;

	tmp = maximal_deviation(-1);
	v_.push_back(tmp);
	
	cv::circle(illustration, outline_[first], 1, col, 10);

	while (nb_assigned != outline_.size())
	{
		col = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
		unsigned int first;
		do
		{
			first = rand() % outline_.size();
		}
		while (cluster_[first] != - 1);
		cluster_[first] = cur_clu;
		++nb_assigned;

		for (unsigned int i = 0; i < outline_.size(); ++i)
		{
			if (i != first && cluster_[i] == -1)
			{
				unsigned int k = 0;
				for (unsigned int c = 0; c < 3; ++c)
				{
					if (abs(finger_char_[i][c] - finger_char_[first][c]) > v_[cur_clu][c])
						++k;
				}
				//if (!k)
				if (k != 3)
				{
					cluster_[i] = cur_clu;
					cv::circle(illustration, outline_[i], 1, col, 10);
					++nb_assigned;
				}
			}
		}
		
		cv::circle(illustration, outline_[first], 1, col, 10);
		
		tmp = maximal_deviation(cur_clu);
		v_[cur_clu] = tmp;

		tmp = maximal_deviation(-1);
		v_.push_back(tmp);

		// Next cluster.
		++cur_clu;
	}

	for (unsigned int i = 0; i < cur_clu; ++i)
	{
		for (unsigned int c = 0; c < 3; ++c)
		{
			if (v_[i][c] > vmax_[c])
				vmax_[c] = v_[i][c];
		}
	}
}

cv::Vec3f Tracking::maximal_deviation(int cluster)
{
	// Compute E[X²] and E[X]².
	unsigned nb_clust_elt = 0;

	cv::Vec3f square_mean(0, 0, 0), mean(0, 0, 0);
	for (unsigned int i = 0; i < outline_.size(); ++i)
	{
		if (cluster_[i] == cluster)
		{
			++nb_clust_elt;
			for (unsigned int c = 0; c < 3; ++c)
			{
				square_mean[c] += pow(finger_char_[i][c], 2);
				mean[c] += finger_char_[i][c];
			}
		}
	}
	cv::Vec3f tmp;
	for (unsigned int c = 0; c < 3; ++c)
	{
		square_mean[c] /= nb_clust_elt;
		mean[c] /= nb_clust_elt;
		tmp[c] = sqrt(square_mean[c] - pow(mean[c], 2));
	}


	return tmp;
}

void Tracking::dissimilarity_computing()
{
	// FIXME: pyramid for the tracking too.
	// FIXME: don't go on the full frame.
	for (unsigned int i = 0; i < mask_.rows; ++i)
	{
		for (unsigned int j = 0; j < mask_.cols; ++j)
		{
			if (mask_.at<uchar>(i, j) == 255)
			{
				cv::Vec3f chara(characteristic_computing(cv::Point(j, i)));
				unsigned int tmp = 0;
				for (unsigned int k = 0; k < outline_.size(); ++k)
				{				
					unsigned int tmp_c = 0;
					for (unsigned int c = 0; c < 3; ++c)
					{
						if (abs(finger_char_[k][c] - chara[c]) > /*v_[cluster_[k]][c]*/ vmax_[c])
							++tmp_c;
					}
					//if (tmp_c == 3)
					if (tmp_c)
						++tmp;
				}
				if (tmp >= outline_.size() * tolerance_thres_)
					d_map_.at<uchar>(i, j) = 255;
			}
		}
	}
}

cv::Mat Scale2(int numberOfScales, cv::Mat image)
{
    cv::Mat temp = image.clone();
    for (int i = 0; i < numberOfScales; i++)
        cv::resize(image, temp, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
    return temp;
}

void Tracking::dissimilarity_scaling_up()
{
	//cv::dilate(dissimilarity_piramid_[4], dissimilarity_piramid_[4], cv::Mat(), cv::Point(-1,-1));
	for (int scale = 3; scale >= 0; --scale)
    {
		for (unsigned int i = 0; i < pyramid_image_[scale + 1].rows; ++i)
		{
			for (unsigned int j = 0; j < pyramid_image_[scale + 1].cols; ++j)
			{
				if (dissimilarity_piramid_[scale + 1].at<uchar>(i, j) == 255)
				{
					cv::Point m(i, j);
					//dissimilarity_piramid_[scale].at<char>(m.x * 2, m.y * 2) = 255;
					dissimilarity_piramid_[scale].at<char>(m.x, m.y) = 255;
				}
			}
		}
		//cv::dilate(dissimilarity_piramid_[scale], dissimilarity_piramid_[scale], cv::Mat(), cv::Point(-1,-1));
    }
}

void Tracking::init()
{
	unsigned int max_scale_ = 4;
	cv::Mat curr = frame_;
	pyramid_image_[0] = frame_;
	pyramid_rough_cont_[0] = rough_selection_;
	dissimilarity_piramid_[0] = d_map_;

	for (size_t i = 1; i <= max_scale_; ++i)
	{
		//curr = Scale2(1, curr);
		dissimilarity_piramid_[i] = cv::Mat(curr.size(), CV_8UC1, cv::Scalar(0));
		pyramid_image_[i] = curr;
		pyramid_rough_cont_[i].resize(rough_selection_.size());
		for (unsigned int k = 0; k < rough_selection_.size(); ++k)
			//pyramid_rough_cont_[i][k] = cv::Point(pyramid_rough_cont_[i - 1][k].x / 2, pyramid_rough_cont_[i - 1][k].y / 2);
			pyramid_rough_cont_[i][k] = cv::Point(pyramid_rough_cont_[i - 1][k].x, pyramid_rough_cont_[i - 1][k].y);
	}
}


void Tracking::build_mask(const cv::Mat& frame)
{
	frame.copyTo(frame_);
	frame.copyTo(illustration);
	d_map_ = cv::Mat(frame_.size(), CV_8UC1, cv::Scalar(0));
	
	init();
	frame_ = pyramid_image_[4];
	d_map_ = dissimilarity_piramid_[4];
	illustration = pyramid_image_[4];
	
	mask_ = cv::Mat(frame_.size(), CV_8UC1, cv::Scalar(0));
	
	int kernel_size = 11;
	cv::Mat3f tmp_img(frame_);
	cv::GaussianBlur(tmp_img, gauss_img_, cv::Size(kernel_size, kernel_size), 0, 0);
	//cv::medianBlur(tmp_img, gauss_img_, 5);
	//cv::bilateralFilter(tmp_img, gauss_img_, -1, -1, -1);
	cv::imshow("Blur", cv::Mat3b(gauss_img_));
	cv::waitKey(0);
	
	selection_refinement(pyramid_rough_cont_[4]);
	dissimilarity_piramid_[4] = d_map_;
	dissimilarity_scaling_up();
	dissimilarity_piramid_[0].copyTo(mask_);

	// Post treatment.
	int largest_area=0;
    int largest_contour_index=0;

    std::vector<std::vector<cv::Point>> contours; // Vector for storing contour
    std::vector<cv::Vec4i> hierarchy;

    findContours(mask_, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
	for (int i = 0; i< contours.size(); i++ )
	{
        double a = contourArea(contours[i], false);
        if(a > largest_area)
		{
            largest_area = a;
            largest_contour_index = i;
        }
    }
	mask_ = cv::Mat(mask_.size(), CV_8UC1, cv::Scalar(0));

	cv::drawContours(mask_, contours, largest_contour_index, cv::Scalar(255), CV_FILLED, 8, hierarchy);
	cv::dilate(mask_, mask_, cv::Mat(), cv::Point(-1,-1), 3);
	// End post.
	// Print the result.

	illustration = pyramid_image_[0];
	for (unsigned int i = 0; i < pyramid_image_[0].rows; ++i)
	{
		for (unsigned int j = 0; j < pyramid_image_[0].cols; ++j)
		{
			if (mask_.at<uchar>(i, j) == 255)
			{
				illustration.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
			}
		}
	}
	cv::imshow("final", illustration);
	cv::waitKey(0);

}


cv::Mat Tracking::get_mask()
{
	return mask_;
}
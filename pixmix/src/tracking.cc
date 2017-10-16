#include "tracking.hh"


Tracking::Tracking(const std::vector<cv::Point>& rough_selection)
	: rough_selection_(rough_selection),
	  outline_(),
	  d_map_(),
	  cluster_(),
	  v_(),
	  vmax_(0, 0, 0),
	  tolerance_thres_(0.95)
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
}

void Tracking::selection_refinement(const std::vector<cv::Point>& hull)
{
	// We want enough fingerprints to correctly segment the object.
	hull_sampling(hull, 3);

	std::vector<std::vector<cv::Point>> cont(1);
	cont[0] = outline_;
	cv::drawContours(mask_, cont, 0, cv::Scalar(255), -1);

	// Compute the characteristics.
	finger_char_.resize(outline_.size());
	for (unsigned int i = 0; i < outline_.size(); ++i)
		finger_char_[i] = characteristic_computing(outline_[i]);

	// Clustering.
	unsigned int iteration = 4;
	std::vector<cv::Vec3f> min_v;
	std::vector<int> min_clu;
	cv::Vec3f min_vmax(FLT_MAX, FLT_MAX, FLT_MAX);
	for (unsigned int i = 0; i < iteration; ++i)
	{
		clustering();
		unsigned int k = 0;
		for (unsigned int c = 0; c < 3; ++c)
		{
			if (min_vmax[c] < vmax_[c])
				++k;
		}
		if (k == 3)
		{
			std::cout << "Current minimal deviation " << vmax_ << std::endl;
			min_clu = cluster_;
			min_v = v_;
			min_vmax = vmax_;
		}
	}

	cluster_ = min_clu;
	v_ = min_v;
	vmax_ = min_vmax;
	
	// Compute the dissimilarity map.
	dissimilarity_computing();

	cv::imshow("lolilol", illustration);
	cv::waitKey(0);
}

void Tracking::clustering()
{
	cv::RNG rng(12345);
	// FIXME: the structure is not very good, use a vector of vector? After finishing the clustering, init a fixed array.
	// Init with every cluster assigned to 0.
	cluster_ = std::vector<int>(outline_.size() ,-1);
	v_.clear();

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
			if (k == 0)
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
				if (!k)
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
		// Since the # of cluster is one at the begining, don't do a max.
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
						if (abs(finger_char_[k][c] - chara[c]) > /*v_[cluster_[k]][c]*/vmax_[c])
							++tmp_c;
					}
					if (tmp_c == 3)
						++tmp;
				}
				if (tmp >= outline_.size() * tolerance_thres_)
				{
					illustration.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
					d_map_.at<uchar>(i, j) = 255;
				}
			}
		}
	}
}

void Tracking::build_mask(const cv::Mat& frame)
{
	frame.copyTo(frame_);
	frame.copyTo(illustration);
	d_map_ = cv::Mat(frame_.size(), CV_8UC1);
	
	// Convex Hull.
	std::vector<cv::Point> hull(rough_selection_.size());
	mask_ = cv::Mat(frame.size(), CV_8UC1, cv::Scalar(0));
	cv::convexHull(rough_selection_, hull, false);
	
	selection_refinement(hull);
}


cv::Mat Tracking::get_mask()
{
	return mask_;
}
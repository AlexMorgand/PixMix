#include "dr.hh"
#include "tracking.hh"

std::vector<cv::Point> cont;
unsigned int k = 0;
cv::Mat img, debug;

void mouseEvent(int evt, int x, int y, int flags, void *param)
{
    if (evt == CV_EVENT_LBUTTONDOWN)
	{
		cont.push_back(cv::Point(x, y));
		cv::circle(debug, cont[k], 10, cv::Scalar(0, 0, 155), -1);
		if (cont.size() > 1)
			cv::line(debug, cont[k - 1], cont[k], cv::Scalar(0, 0, 255));
		++k;
		cv::imshow("Mask building", debug);
    }
}


int main(int argc, char* argv[])
{
	bool manual = true;
	std::string prefix("C:/Users/am237982/Desktop/cea/CEA/Alexandre/Dev/supermedia_inpaint/PixMix/pixmix/input/");

	std::list<std::pair<char*, char*> > tests;
	//tests.push_back(std::make_pair(strdup("or.jpeg"), strdup("or.jpeg")));
	//tests.push_back(std::make_pair(strdup("gens.jpeg"), strdup("gens.jpeg")));
	//tests.push_back(std::make_pair(strdup("money.jpg"), strdup("money.jpg")));
	//tests.push_back(std::make_pair(strdup("aure.png"), strdup("aure.png")));
	tests.push_back(std::make_pair(strdup("her2mask.png"), strdup("her2.png")));
	tests.push_back(std::make_pair(strdup("her1mask.png"), strdup("her1.png")));
	//tests.push_back(std::make_pair(strdup("rock.png"), strdup("rock.png")));
	//tests.push_back(std::make_pair(strdup("chips.png"), strdup("chips.png")));
	//tests.push_back(std::make_pair(strdup("shield.png"), strdup("shield.png")));
	//tests.push_back(std::make_pair(strdup("pixmixex1_mask.png"), strdup("pixmixex1.png")));
	//tests.push_back(std::make_pair(strdup("pixmixex2_mask.png"), strdup("pixmixex2.png")));
	//tests.push_back(std::make_pair(strdup("pixmixex3_mask.png"), strdup("pixmixex3.png")));
	tests.push_back(std::make_pair(strdup("tatoo_mask.jpg"), strdup("tatoo.jpg")));
	//tests.push_back(std::make_pair(strdup("fontainmask.png"), strdup("fontainreal.jpg")));
	//tests.push_back(std::make_pair(strdup("bungeemask.png"), strdup("bungeereal.jpg")));
	//tests.push_back(std::make_pair(strdup("boat_mask.jpg"), strdup("boat.jpg")));
	//tests.push_back(std::make_pair(strdup("panneaumask.png"), strdup("panneau.jpg")));
	//tests.push_back(std::make_pair(strdup("ombres/Guo4.png"), strdup("ombres/Guo4.tif")));
	//tests.push_back(std::make_pair(strdup("ombres/Guo8.png"), strdup("ombres/Guo8.jpg")));
	//tests.push_back(std::make_pair(strdup("ombres/Guo1.png"), strdup("ombres/Guo1.tif")));
	tests.push_back(std::make_pair(strdup("doudou_mask.jpg"), strdup("doudou.jpg")));
	tests.push_back(std::make_pair(strdup("lolilol_mask.jpg"), strdup("lolilol.jpg")));
	tests.push_back(std::make_pair(strdup("Alexandre/mask_85.png"), strdup("Alexandre/0085.jpg")));
	tests.push_back(std::make_pair(strdup("Alexandre/mask_145.png"), strdup("Alexandre/0145.jpg")));
	tests.push_back(std::make_pair(strdup("Alexandre/mask_175.png"), strdup("Alexandre/0175.jpg")));
	tests.push_back(std::make_pair(strdup("Alexandre/mask_205.png"), strdup("Alexandre/0205.jpg")));
	tests.push_back(std::make_pair(strdup("chairi.PNG"), strdup("chair.PNG")));
	tests.push_back(std::make_pair(strdup("doggy_mask.jpg"), strdup("doggy.jpg")));
	tests.push_back(std::make_pair(strdup("bw1mask.png"), strdup("bw1.png")));
	tests.push_back(std::make_pair(strdup("argmask.png"), strdup("arg.jpg")));



	std::list<std::pair<char*, char*> >::iterator it = tests.begin();
	for (; it != tests.end(); ++it)
	{
		// 2D inpainting, with a binary mask.

		if (manual)
		{
			img = cv::imread(prefix + it->second);
			img.copyTo(debug);
			cv::imshow("Mask building", debug);
			cvSetMouseCallback("Mask building", mouseEvent, 0);
			cv::waitKey(0);
			Tracking tr(cont);
			tr.build_mask(img);
			DR dr(tr.get_mask(), img, prefix/*, DR::INTENSITY*/);
			dr.inpaint();
		}
		else
		{
			DR dr(it->first, it->second, prefix/*, DR::INTENSITY*/);
			dr.inpaint();
		}
		cv::waitKey(0);
		cont.clear();
		cv::destroyAllWindows();
	}

  return 0;
}

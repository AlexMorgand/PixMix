#include "dr.hh"

int main(int argc, char* argv[])
{
	std::string prefix("C:/Users/am237982/Desktop/cea/CEA/Alexandre/Dev/supermedia_inpaint/PixMix/pixmix/input/");

  std::list<std::pair<char*, char*> > tests;
  tests.push_back(std::make_pair(strdup("fontainmask.png"), strdup("fontainreal.jpg")));
  tests.push_back(std::make_pair(strdup("bungeemask.png"), strdup("bungeereal.jpg")));
  tests.push_back(std::make_pair(strdup("boat_mask.jpg"), strdup("boat.jpg")));
  tests.push_back(std::make_pair(strdup("boat_mask.jpg"), strdup("boat.jpg")));
  tests.push_back(std::make_pair(strdup("panneaumask.png"), strdup("panneau.jpg")));
  tests.push_back(std::make_pair(strdup("ombres/Guo4.png"), strdup("ombres/Guo4.tif")));
  tests.push_back(std::make_pair(strdup("ombres/Guo8.png"), strdup("ombres/Guo8.jpg")));
  tests.push_back(std::make_pair(strdup("ombres/Guo1.png"), strdup("ombres/Guo1.tif")));
  tests.push_back(std::make_pair(strdup("doudou_mask.jpg"), strdup("doudou.jpg")));
  tests.push_back(std::make_pair(strdup("lolilol_mask.jpg"), strdup("lolilol.jpg")));
  tests.push_back(std::make_pair(strdup("Alexandre/mask_85.png"), strdup("Alexandre/0085.jpg")));
  tests.push_back(std::make_pair(strdup("Alexandre/mask_145.png"), strdup("Alexandre/0145.jpg")));
  tests.push_back(std::make_pair(strdup("Alexandre/mask_175.png"), strdup("Alexandre/0175.jpg")));
  tests.push_back(std::make_pair(strdup("Alexandre/mask_205.png"), strdup("Alexandre/0205.jpg")));
  tests.push_back(std::make_pair(strdup("chairi.PNG"), strdup("chair.PNG")));
  tests.push_back(std::make_pair(strdup("doggy_mask.jpg"), strdup("doggy.jpg")));
  tests.push_back(std::make_pair(strdup("bw1mask.png"), strdup("bw1.png")));
  tests.push_back(std::make_pair(strdup("her2mask.png"), strdup("her2.png")));
  tests.push_back(std::make_pair(strdup("her1mask.png"), strdup("her1.png")));
  tests.push_back(std::make_pair(strdup("argmask.png"), strdup("arg.jpg")));
  tests.push_back(std::make_pair(strdup("1mask.png"), strdup("1.png")));
  tests.push_back(std::make_pair(strdup("2mask.png"), strdup("2.png")));

  // 2D inpainting, with a binary mask.

  std::list<std::pair<char*, char*> >::iterator it = tests.begin();
  for (; it != tests.end(); ++it)
  {
      DR dr(it->first, it->second, prefix/*, DR::INTENSITY*/);
      dr.inpaint();
      cv::waitKey(0);
  }

  return 0;
}

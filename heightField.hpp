#ifndef SRC_HEIGHTFIELD_HPP
#define SRC_HEIGHTFIELD_HPP

// This file is created by Michel Aractingi//
// michel.aractingi@naverlabs.com//

#include <vector>
#include <Eigen/Core>
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "Yaml.hpp"

class HeightField {

 public:
 HeightField(){}
 HeightField(raisim::World* world, double max_height){
   max_height_ = max_height;
   world_ = world;

   heightSampler_ = std::uniform_real_distribution<double>(0.001, max_height_);
   
   heightVec.setZero(samples*samples);
 }

 void remove(){
   if (loaded_){
     world_->removeObject(hm);
     loaded_ = false;
   }
 }

 void update(){
   remove();
   heightVec.setZero(samples*samples);
   // Fill map
   float height = 0.0;
   for(int j=0; j<samples/2; j++){
     for (int i=0; i<samples/2; i++){
       height = heightSampler_(generator_);
       heightVec(2*i+2*j*samples)=height;
       heightVec(2*i+1+2*j*samples)=height;
       heightVec(2*i+(2*j+1)*samples)=height;
       heightVec(2*i+1+(2*j+1)*samples)=height;
     }
   }
   std::vector<double> hmvec(heightVec.data(),
		        heightVec.data() + heightVec.rows() * heightVec.cols());
   hm = world_->addHeightMap(samples, samples, map_dim, map_dim, 0, 0, hmvec);
   loaded_=true;
 }

 void viz(raisim::RaisimServer* server, Eigen::MatrixXd positions){
	 for (int i = 0; i < positions.rows() ; i++){
           auto v_sphere = server->addVisualSphere("sphere"+std::to_string(i), 0.02, 0.96, 0.15, 0.1, 0.0);
           v_sphere->setPosition(positions.coeff(i, 0), positions.coeff(i, 1), hm->getHeight(positions.coeff(i, 0), positions.coeff(i, 1)));
	 }
 }

 bool is_loaded(){
   return loaded_;
 }

 void getState(Eigen::VectorXd state){}

 private:
  double max_height_;
  int samples = 500;
  double map_dim=50;
  Eigen::VectorXd heightVec;
  raisim::HeightMap *hm;
  raisim::World* world_;

  // random dist parameters
  //
  std::default_random_engine generator_ = std::default_random_engine (std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<double> heightSampler_;

  bool loaded_=false;

};

#endif //SRC_HEIGHTFIELD_HPP

/*
 * terrainProperties = raisim.TerrainProperties()
    ...: terrainProperties.frequency = config.get('frequency', 4)
    ...: terrainProperties.zScale = config.get('zscale', 0.5)
    ...: terrainProperties.xSize = config.get('xSize', 50)
    ...: terrainProperties.ySize = config.get('ySize', 50)
    ...: terrainProperties.xSamples = config.get('xSamples', 250)
    ...: terrainProperties.ySamples = config.get('ySamples', 250)
    ...: terrainProperties.fractalOctaves = config.get('fractalOctaves', 3)
    ...: terrainProperties.fractalLacunarity = config.get('fractalLacunarity', 2)
    ...: terrainProperties.fractalGain = config.get('fractalGain', 0.5)
*/

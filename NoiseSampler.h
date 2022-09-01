// This file is created by Michel Aractingi//
// michel.aractingi@naverlabs.com//
//
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include "eigenmvn.h"


class NoiseSampler {

  public:
    NoiseSampler() {
	
	Eigen::DiagonalMatrix<double, Eigen::Dynamic> var;
	Eigen::Matrix<double, Eigen::Dynamic, 1> mean;

	// Initialize noise scales vectors
	heightScale.setZero(1);
        rpyScale.setZero(3);
        jointScale.setZero(12);
        velScale.setZero(3);
        omegaScale.setZero(3);


	heightScale.setConstant( 0.01);
        rpyScale.setConstant(0.005);
        jointScale.setConstant( 0.005);
        velScale.setConstant( 0.01);
        omegaScale.setConstant(0.01);

	mean.setZero(obDim_);
	var.diagonal().setZero(obDim_);

	var.diagonal() << heightScale, rpyScale, jointScale,
	                  velScale, omegaScale, jointScale;

        Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic>  covar = var.diagonal().asDiagonal();
	
	ob_noise.setCovar(covar);
	ob_noise.setMean(mean);    
    }

    Eigen::Matrix<double, Eigen::Dynamic, -1> sample(){
        return ob_noise.samples(1,1);
    }

 private:

   int obDim_=34;
   unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
   Eigen::EigenMultivariateNormal<double> ob_noise = Eigen::EigenMultivariateNormal<double>(false, seed);

   Eigen::VectorXd heightScale;
   Eigen::VectorXd rpyScale;
   Eigen::VectorXd jointScale;
   Eigen::VectorXd velScale;
   Eigen::VectorXd omegaScale;


};

/*
 * Initial State randomizer
 */
class InitialStateRandomizer {

  public:
    InitialStateRandomizer() {
	
	Eigen::DiagonalMatrix<double, Eigen::Dynamic> gcVar;
	Eigen::DiagonalMatrix<double, Eigen::Dynamic> gvVar;

	// Initialize noise scales vectors
	gcVar.diagonal().setZero(gcDim_);
        gvVar.diagonal().setZero(gvDim_);

	gcVar.diagonal().tail(12).setConstant(0.1);
        gvVar.diagonal().tail(12).setConstant(0.01);

        gcNoise.setCovar(gcVar.diagonal().asDiagonal());
        gvNoise.setCovar(gvVar.diagonal().asDiagonal());
	
	gcNoise.setMean(Eigen::VectorXd::Zero(gcDim_));    
	gvNoise.setMean(Eigen::VectorXd::Zero(gvDim_));    
    }

    Eigen::Matrix<double, Eigen::Dynamic, -1> sample_gc(){
        return gcNoise.samples(1,1);
    }
    Eigen::Matrix<double, Eigen::Dynamic, -1> sample_gv(){
        return gvNoise.samples(1,1);
    }

 private:

   unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
   Eigen::EigenMultivariateNormal<double> gcNoise = Eigen::EigenMultivariateNormal<double>(false, seed);
   Eigen::EigenMultivariateNormal<double> gvNoise = Eigen::EigenMultivariateNormal<double>(false, seed);

   int gcDim_ = 19;
   int gvDim_ = 18;


};


/*
 * Sampling noise vector from a uniform distribution
 * 
 */

class UniformNoiseSampler {

  public:

    UniformNoiseSampler(int size, float minval, float maxval) {

	size_ = size;
	minval_ = minval;
	maxval_ = maxval;

	distribution_ = std::uniform_real_distribution<double>(minval_, maxval_);
    }

    UniformNoiseSampler () = default;

    Eigen::VectorXd sample(){
        auto sample_uniform_ = [&] (double) {return distribution_(generator_);};
	return Eigen::VectorXd::NullaryExpr(size_, sample_uniform_);

    }

 private:

   unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
   int size_;
   float minval_, maxval_;

   std::default_random_engine generator_ = std::default_random_engine (seed);
   std::uniform_real_distribution<double> distribution_;

};

// This file is created by Michel Aractingi//
// michel.aractingi@naverlabs.com//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../include/RaisimGymEnv.hpp"
#include "NoiseSampler.h"
#include "heightField.hpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    robot_ = world_->addArticulatedSystem(resourceDir_+"/solo/solo.urdf");
    robot_->setName("solo");

    robot_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);

    world_->addGround();

    /// get robot data
    gcDim_ = robot_->getGeneralizedCoordinateDim();
    gvDim_ = robot_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_); old_gc_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_); old_gv_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of solo

    gc_init_ << 0, 0, 1, 1.0, 0.0, 0.0, 0.0, -0.04, 0.7, -1.4, 0.04, 0.7, -1.4, -0.04, 0.7, -1.4, 0.04, 0.7, -1.4;
    if (cfg_["use_symmetric_pose"].template As<bool>())
      gc_init_ << 0, 0, 1, 1.0, 0.0, 0.0, 0.0, -0.04, 0.8, -1.6, 0.04, 0.8, -1.6, -0.04, -0.8, +1.6, 0.04, -0.8, +1.6;

    /// set pd gains
    jointPgain.setZero(gvDim_); jointPgain.tail(nJoints_).setConstant(3.);
    jointDgain.setZero(gvDim_); jointDgain.tail(nJoints_).setConstant(0.2);
    robot_->setPdGains(Eigen::VectorXd::Zero(gvDim_), Eigen::VectorXd::Zero(gvDim_));
    robot_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    lastAction_.setZero(actionDim_); lastlastAction_.setZero(actionDim_);

    bodyLinearVel_.setZero(); 
    bodyAngularVel_.setZero();

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.3);

    BodyStateSize_ = 3 + 3 + 3 + 3; // ori/ vel/ angvel/ command body vel part of state

    obDim_ = BodyStateSize_ + 2 * nJoints_ + actionDim_*2 + 3*(2*nJoints_);
    obDouble_.setZero(obDim_);

    historyTempMem_.setZero(num_history_stack_*nJoints_); jointPosErrorHist_.setZero(num_history_stack_*nJoints_);
    jointVelHist_.setZero(num_history_stack_*nJoints_); 

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// Initialize Torque holder
    torque_holder_.setZero(nJoints_);
    desired_force_.setZero(gvDim_);
    filter_torques_.setZero(gvDim_);
    powerSum_.setZero(nJoints_);
    frictionTorque.setZero(nJoints_);

    /// indices of links that should not make contact with ground
    footIndices_.insert(robot_->getBodyIdx("FR_LOWER_LEG"));
    footIndices_.insert(robot_->getBodyIdx("FL_LOWER_LEG"));
    footIndices_.insert(robot_->getBodyIdx("HR_LOWER_LEG"));
    footIndices_.insert(robot_->getBodyIdx("HL_LOWER_LEG"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(robot_);
    }

    // Add Terrain Generator
    flat_terrain = cfg_["flat_terrain"].template As<bool>();

    // Initialize samplers for velocity commands
    velCommand_.setZero(3);
    VxSample_ = UniformNoiseSampler(1, -1.5, +1.);
    VySample_ = UniformNoiseSampler(1, -.75, .75);
    WzSample_ = UniformNoiseSampler(1, -1., 1.);

    if (randomDynamics_){
      // friction
      friction_ = UniformNoiseSampler(1, -0.4, 0.2);

      // PD gains
      kpNoise_ = UniformNoiseSampler(1, -0.5, 2);
      kdNoise_ = UniformNoiseSampler(1, -0.1, 0.1);

      // Body mass mimics load on robot
      initMass_ = robot_->getMass(0);
      BodyMassNoise_ = UniformNoiseSampler(1, 0.0, .3);
    }

    if (randomObservation_){
      // joint positions and velocities
      qNoise_ = UniformNoiseSampler(12, -0.05, 0.05);
      qdNoise_ = UniformNoiseSampler(12, -0.5, 0.5);

      // Body Orientation / velocity/ angular velocity
      bodyOrientationNoise_ = UniformNoiseSampler(4, -0.03, 0.03);
      AngularVelNoise_ = UniformNoiseSampler(3, -0.1, 0.1);
      LinearVelNoise_ = UniformNoiseSampler(3, -0.1, 0.1);
    }

    // Initialize Curriculum Factr
    use_curriculum_ = cfg_["use_curriculum"].template As<bool>();
    if (use_curriculum_){
        kc = 0.1;
        kc_noise = 0.0;
    }
    else{
        kc = 1.0;
        kc_noise = 1.0;
    }

    heightField_ = HeightField(world_.get(), 0.05);

  }

  void init() final { }

  void reset() final { 
    // 25-percent chance of completing last episode if last didn't fail
    robot_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
    if (!loaded_ || done_ || rand() % 100 > 25){ // Start from Init State
       robot_->setState(gc_init_, gv_init_);

       filter_torques_.tail(nJoints_) = actionMean_;
       if (!flat_terrain && rand() % 10 < 8 && kc_noise == 1.0)
           heightField_.update();
       else
	   heightField_.remove();

       this->setRobotToGround();
       done_ = false;

       lastAction_.setZero();
       lastlastAction_.setZero();
       old_gc_.setZero();
       old_gv_.setZero();
       historyTempMem_.setZero(num_history_stack_*nJoints_); jointPosErrorHist_.setZero(num_history_stack_*nJoints_); jointVelHist_.setZero(num_history_stack_*nJoints_); 

       // Fill in history
       pTarget12_ = actionMean_;
       lastlastAction_ = pTarget12_;
       lastAction_ = pTarget12_;
       for (int i=0; i<num_history_stack_ ; i++){
         world_->integrate();   
         robot_->getState(gc_, gv_);
         jointVelHist_.segment(nJoints_ * i, nJoints_) << gv_.tail(nJoints_);
       }
       loaded_ = true;
    }

    obDouble_.setZero();
    error_ = false;
    timestep = 0;

    // randomize dynamics if necessary
    if (randomDynamics_)
    {
      double friction = 0.8 + kc_noise * friction_.sample().coeff(0,0);
      world_->setDefaultMaterial(friction,0.0, 0.0, 0.0, 0.0);

      double kp = 3. +  kc_noise  * kpNoise_.sample().coeff(0,0);
      double kd = 0.2 + kc_noise * kdNoise_.sample().coeff(0,0);
      if (kd < 0.0) kd = 0.0;

      jointPgain.tail(nJoints_).setConstant(kp);
      jointDgain.tail(nJoints_).setConstant(kd);

      // modify mass of the body (0 index)
      double mass = initMass_ + kc_noise * BodyMassNoise_.sample().coeff(0,0) * 0;
      robot_->setMass(0, mass);
    }

    velCommand_<< .5 + (kc - 0.1 )* VxSample_.sample().coeff(0,0) , (kc - 0.1 )* VySample_.sample().coeff(0,0), (kc - 0.1 )* WzSample_.sample().coeff(0,0) ;
    if (std::abs(velCommand_(0))<0.2) velCommand_(0) = 0.;
    if (std::abs(velCommand_(1))<0.2) velCommand_(1) = 0.;
    if (std::abs(velCommand_(2))<0.2) velCommand_(2) = 0.;

    if(cfg_["print_command"].template As<bool>())
        std::cout<<"vel:\n "<<velCommand_<<std::endl;

    InfoMap_["torque"] = 0;
    InfoMap_["energy"] = 0;
    InfoMap_["jointsPos"] = 0;
    InfoMap_["jointsVel"] = 0;
    InfoMap_["jointsAcc"] = 0;
    InfoMap_["ActionSmoothness1"] = 0;
    InfoMap_["ActionSmoothness2"] = 0;
    InfoMap_["bodyVel"] = 0;
    InfoMap_["bodyOrn"] = 0;
    InfoMap_["footSlip"] = 0;
    InfoMap_["footClearance"] = 0;
    InfoMap_["linVelReward"] = 0;
    InfoMap_["angVelReward"] = 0;
    InfoMap_["linVelError"] = 0;
    InfoMap_["linVelError_abs"] = 0;
    InfoMap_["angVelError"] = 0;
    InfoMap_["reward_total"] = 0; 
    InfoMap_["length"] = 0;
    InfoMap_["base_height"] = 0;
    InfoMap_["Vx"] = velCommand_(0);
    InfoMap_["Vy"] = velCommand_(1);
    InfoMap_["Wz"] = velCommand_(2);

  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);//.cwiseMin(M_PI).cwiseMax(-M_PI);
    pTarget12_ += actionMean_;
    
    pTarget_.tail(nJoints_)  = pTarget12_;

    powerSum_.setZero();
    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){

      //robot_->getState(gc_, gv_);
      updateObservation();
      
      // Calculate torques
      desired_force_.tail(nJoints_) = jointPgain(7) * (pTarget12_ - gc_.tail(nJoints_))
      	                      - jointDgain(7) * (gv_.tail(nJoints_));
      
      // Clamp torques
      desired_force_ = desired_force_.cwiseMin(3).cwiseMax(-3);
      
      // Low pass filter the torques
      filter_torques_ = alpha_ * filter_torques_ + (1 - alpha_) * desired_force_;
      
      // Send torque commands to raisim
      robot_->setGeneralizedForce(filter_torques_);
      
      // caculate energy consumption term
      calculatePower();

      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    timestep += 1;

    // reset very every episode length 
    if (velSwitches_ && timestep > 0 && timestep % (max_timestep/2) == 0){ 
      velCommand_<< .5 + (kc - 0.1 )* VxSample_.sample().coeff(0,0) , (kc - 0.1 )* VySample_.sample().coeff(0,0), (kc - 0.1 )* WzSample_.sample().coeff(0,0) ;
      if (std::abs(velCommand_(0))<0.2) velCommand_(0) = 0.;
      if (std::abs(velCommand_(1))<0.2) velCommand_(1) = 0.;
      if (std::abs(velCommand_(2))<0.2) velCommand_(2) = 0.;
      if(cfg_["print_command"].template As<bool>())
         std::cout<<"vel:\n "<<velCommand_<<std::endl;
      
      InfoMap_["Vx"] = velCommand_(0);
      InfoMap_["Vy"] = velCommand_(1);
      InfoMap_["Wz"] = velCommand_(2);
    }
   
    updateObservation();
    updateHistory();

    // Check for errors
    error_ = NanInState(gc_, gv_);

    if (error_) return 0.f;

    // Calculate negative rewards 
    double torque = robot_->getGeneralizedForce().squaredNorm();
    double energy = powerSum_.sum() * simulation_dt_;
    double jointsPos = (gc_.tail(nJoints_) - actionMean_).squaredNorm();
    double jointsVel = gv_.tail(nJoints_).squaredNorm();
    double jointsAcc = (gv_.tail(nJoints_) - old_gv_.tail(nJoints_)).squaredNorm();
    double ActionSmoothness1 = (pTarget12_ - lastAction_).squaredNorm();
    double ActionSmoothness2 = (pTarget12_ - 2*lastAction_ + lastlastAction_).squaredNorm();
    double bodyVel = std::pow(bodyLinearVel_[2],2);
    double bodyOrn = (gc_.segment(3,4)- gc_init_.segment(3,4)).squaredNorm();

    // foot slip reward
    double footSlip = 0.0, footClearance = 0.0;
    double cidx = 0;
    raisim::Vec<3> cvel, cpos;
    for(auto& contact: robot_->getContacts()){
        if(footIndices_.find(contact.getlocalBodyIndex()) != footIndices_.end()){
            robot_->getContactPointVel(cidx, cvel);
            footSlip += cvel.e().head(2).squaredNorm();
	}
	cidx += 1;
    }

    robot_->getFramePosition(robot_->getFrameByName("FL_ANKLE"), cpos);
    robot_->getFrameVelocity(robot_->getFrameByName("FL_ANKLE"), cvel);
    feetHeight_(1) = cpos[2];
    footClearance += std::pow(cpos[2] - footZoffset_ - maxfh_, 2) * std::pow(cvel.e().head(2).norm(),0.5);

    robot_->getFramePosition(robot_->getFrameByName("FR_ANKLE"), cpos);
    robot_->getFrameVelocity(robot_->getFrameByName("FR_ANKLE"), cvel);
    feetHeight_(0) = cpos[2];
    footClearance += std::pow(cpos[2] - footZoffset_ - maxfh_, 2) * std::pow(cvel.e().head(2).norm(),0.5);

    robot_->getFramePosition(robot_->getFrameByName("HL_ANKLE"), cpos);
    robot_->getFrameVelocity(robot_->getFrameByName("HL_ANKLE"), cvel);
    feetHeight_(3) = cpos[2];
    footClearance += std::pow(cpos[2] - footZoffset_ - maxfh_, 2) * std::pow(cvel.e().head(2).norm(),0.5);

    robot_->getFramePosition(robot_->getFrameByName("HR_ANKLE"), cpos);
    robot_->getFrameVelocity(robot_->getFrameByName("HR_ANKLE"), cvel);
    feetHeight_(2) = cpos[2];
    footClearance += std::pow(cpos[2] - footZoffset_ - maxfh_, 2) * std::pow(cvel.e().head(2).norm(),0.5);

    double reward_neg = w_fs * footSlip + w_fcl * footClearance + w_bOrn * bodyOrn
	                + w_torque * torque + w_jvel * jointsVel + w_jacc * jointsAcc
		       	+ w_jpos * jointsPos + w_as1 * ActionSmoothness1 + w_as2 * ActionSmoothness2
		       	+ w_bVel * bodyVel + w_energy * energy;

    // Calculate Positive Rewards
    double linVelReward = std::exp(-(bodyLinearVel_.head(2) - velCommand_.head(2)).squaredNorm());
    double angVelReward = std::exp(-std::pow(bodyAngularVel_[2] - velCommand_[2],2));

    double reward_pos =  (w_linVel*linVelReward + w_angVel*angVelReward);
    
    double reward_total = reward_pos + std::abs(w_rewardNeg) * kc * reward_neg;

    rewards_.record("reward_total", reward_total);

    InfoMap_["torque"] = torque;
    InfoMap_["energy"] = energy;
    InfoMap_["reward_pos"] = reward_pos;
    InfoMap_["reward_neg"] = reward_neg;
    InfoMap_["jointsPos"] = jointsPos;
    InfoMap_["jointsVel"] = jointsVel;
    InfoMap_["jointsAcc"] = jointsAcc;
    InfoMap_["ActionSmoothness1"] = ActionSmoothness1;
    InfoMap_["ActionSmoothness2"] = ActionSmoothness2;
    InfoMap_["bodyVel"] = bodyVel;
    InfoMap_["bodyOrn"] = bodyOrn;
    InfoMap_["footSlip"] = footSlip;
    InfoMap_["footClearance"] = footClearance;
    InfoMap_["linVelReward"] = linVelReward;
    InfoMap_["angVelReward"] = angVelReward;
    InfoMap_["reward_total"] = reward_total;
    InfoMap_["length"] = timestep;
    InfoMap_["base_height"] = gc_(2);

    lastlastAction_ = lastAction_;
    lastAction_ = pTarget12_;

    old_gc_ = gc_;
    old_gv_ = gv_;

    return rewards_.sum();
 }

  void updateHistory() { 
    historyTempMem_ = jointVelHist_;
    jointVelHist_.head((num_history_stack_-1) * nJoints_) = historyTempMem_.tail((num_history_stack_-1) * nJoints_);
    jointVelHist_.tail(nJoints_) = gv_.tail(nJoints_);

    historyTempMem_ = jointPosErrorHist_;
    jointPosErrorHist_.head((num_history_stack_-1) * nJoints_) = historyTempMem_.tail((num_history_stack_-1) * nJoints_);
    jointPosErrorHist_.tail(nJoints_) = pTarget12_ - gc_.tail(nJoints_);

    if (randomObservation_){
      jointPosErrorHist_.tail(nJoints_) += kc_noise * qNoise_.sample(); 
      jointVelHist_.tail(nJoints_) += kc_noise * qdNoise_.sample();
    }
  }

  void updateObservation() {
    robot_->getState(gc_, gv_);
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    obDouble_ <<
                getMod2PI(rot.e().row(2).transpose()),
                bodyLinearVel_,
                bodyAngularVel_,
                velCommand_,
		gc_.tail(nJoints_), 
		gv_.tail(nJoints_),
		lastAction_, 
		lastlastAction_,
                jointPosErrorHist_.segment(0*nJoints_, nJoints_),
                jointVelHist_.segment(0*nJoints_, nJoints_),
                jointPosErrorHist_.segment(2*nJoints_, nJoints_),
                jointVelHist_.segment(2*nJoints_, nJoints_),
                jointPosErrorHist_.segment(4*nJoints_, nJoints_),
                jointVelHist_.segment(4*nJoints_, nJoints_);


    if (randomObservation_){
      obDouble_.segment(0,3) += kc_noise * bodyOrientationNoise_.sample();
      obDouble_.segment(3,3) += kc_noise * LinearVelNoise_.sample();
      obDouble_.segment(6,3) += kc_noise * AngularVelNoise_.sample();
      obDouble_.segment(12,12) += qNoise_.sample();
      obDouble_.segment(24,12) += qdNoise_.sample();
    }

    ob = obDouble_.cast<float>();
  }

  void getJointTorque(Eigen::Ref<EigenVec> tau)  {
    tau << robot_->getGeneralizedForce().e().tail(nJoints_).cast<float>();
  }

  void setReferenceVelocities(const Eigen::Ref<Eigen::Vector3d>& refLin, const Eigen::Ref<Eigen::Vector3d>& refAng)  {
    velCommand_(0) = refLin.cast<double>()(0);
    velCommand_(1) = refLin.cast<double>()(1);
    velCommand_(2) = refAng.cast<double>()(2);
    if (std::abs(velCommand_(0))<0.2) velCommand_(0) = 0.;
    if (std::abs(velCommand_(1))<0.2) velCommand_(1) = 0.;
    if (std::abs(velCommand_(2))<0.2) velCommand_(2) = 0.;
  }

  void calculatePower(){
  /* 
   * coulomb_tau = 0.0477
   * viscous_b = 0.000135
   * K_motor = 4.81
   *
   * P = P_t + P_f
   * P_f: power loss due to friction, tau_f * qa_dot
   * P_t: power due to torques, K * tau**2
   * ------------------------------------------
   * tau_f: friction torque, tau_c * sign(qa_dot) + b * qa_dot
   * tau_c: columb friction 
   * b: viscous friction
   * K: scale motor resistance
   * These constant values are provided by the lab
   */
   frictionTorque = 0.0477 * gv_.tail(nJoints_).array().sign().matrix() + 0.000135 * gv_.tail(nJoints_);

   powerSum_ += (frictionTorque.cwiseProduct(gv_.tail(nJoints_)) 
		 + 4.81 * filter_torques_.cwiseProduct(filter_torques_));
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    if (error_)
    {
     std::cout<<"Be careful my friend I found NANS in your state."<<std::endl;
     done_ = true;
     return true;
    }

    /// if the contact body is not feet
    for(auto& contact: robot_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()){
        done_ = true;
        return true;
      }
    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() {
    if (kc < 1.) kc += cfg_["reward_curriculum_increment"].template As<double>();
  }
  void curriculumUpdateNoise() {
    if (kc_noise < 1.) kc_noise += cfg_["noise_curriculum_increment"].template As<double>();
  }

  void setRobotToGround() { 
    Eigen::VectorXd gc_init_copy = gc_init_.replicate(1,1);
    while (robot_->getContacts().size() == 0){
      gc_init_copy[2] = gc_init_copy[2] - 0.01 ;
      robot_->setState(gc_init_copy, gv_init_) ;
      world_->integrate();
    }

    // Fix all legs to ground
    pTarget_.tail(nJoints_) = actionMean_;

    robot_->setPdTarget(pTarget_, vTarget_);

    while (true) {
      world_->integrate();
      std::unordered_set<int> contacts_;
      for(auto& contact: robot_->getContacts()){
        contacts_.insert(contact.getlocalBodyIndex());
      }
      if (contacts_.size() == 4){
        break;
      }
    }
  }


 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* robot_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, old_gc_, old_gv_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_, torque_holder_, desired_force_, obAux_;
  Eigen::VectorXd lastAction_, lastlastAction_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  Eigen::VectorXd feetHeight_, contactState_;
  bool loaded_ = false;
  bool addExternalForce_ = false;
  bool randomInitState_ = false;
  bool use_curriculum_ = false;
  int num_history_stack_ = 6;
  double alpha_ = cfg_["alpha"].template As<double>();

  int k_rl = int(control_dt_ / simulation_dt_ + 1e-10);

  Eigen::Vector3d force;
  raisim::Vec<4> quat;
  raisim::Mat<3,3> rot;

  Eigen::VectorXd qMin, qMax;
  Eigen::VectorXd filter_torques_;

  int max_timestep = static_cast<int>(cfg_["max_time"].template As<double>()/control_dt_);

  float kc, kc_noise;
  raisim::HeightMap *hm;

  bool randomDynamics_ = cfg_["random_dynamics"].template As<bool>();
  bool randomObservation_ = cfg_["random_observations"].template As<bool>();

  // Noise variables for randomization
  UniformNoiseSampler friction_, kpNoise_, kdNoise_, qNoise_, qdNoise_, bodyOrientationNoise_, AngularVelNoise_, LinearVelNoise_, BodyMassNoise_;

  UniformNoiseSampler VxSample_, VySample_, WzSample_;

  Eigen::VectorXd frictionTorque, powerSum_;
  double initMass_; 

  Eigen::VectorXd velCommand_;

  Eigen::VectorXd historyTempMem_, jointPosErrorHist_, jointVelHist_; 
  Eigen::VectorXd jointPgain, jointDgain;

  // reward params
  double w_fs = cfg_["reward"]["foot_slip"]["coeff"].template As<double>();
  double w_fcl = cfg_["reward"]["foot_clearance"]["coeff"].template As<double>();
  double w_bOrn = cfg_["reward"]["body_orientation"]["coeff"].template As<double>();
  double w_torque = cfg_["reward"]["torque"]["coeff"].template As<double>();
  double w_jpos = cfg_["reward"]["joints_pos"]["coeff"].template As<double>();
  double w_jvel = cfg_["reward"]["joints_vel"]["coeff"].template As<double>();
  double w_jacc = cfg_["reward"]["joints_acc"]["coeff"].template As<double>();
  double w_as1 = cfg_["reward"]["action_smoothness1"]["coeff"].template As<double>();
  double w_as2 = cfg_["reward"]["action_smoothness2"]["coeff"].template As<double>();
  double w_bVel = cfg_["reward"]["body_vel"]["coeff"].template As<double>();

  double w_linVel = cfg_["reward"]["linear_vel"]["coeff"].template As<double>();
  double w_angVel = cfg_["reward"]["angular_vel"]["coeff"].template As<double>();
  double w_rewardNeg = cfg_["reward"]["reward_neg"]["coeff"].template As<double>();
  double w_energy = cfg_["reward"]["energy"]["coeff"].template As<double>();

  float maxfh_ = cfg_["max_foot_height"].template As<double>();

  // Initial Foot z offset
  float footZoffset_ = 0.0148;

  // define step terrain
  HeightField heightField_;

 };
}


//
// Created by xiang on 2022/2/15.
//

#include "core/lio/eskf.hpp"

namespace lightning {

/**
 * @brief
 *
 * @param dt 积分时间
 * @param Q 陀螺仪噪声矩阵
 * @param gyro 角速度输入
 * @param acce 加速度输入
 */
void ESKF::Predict(const double& dt, const ESKF::ProcessNoiseType& Q, const Vec3d& gyro, const Vec3d& acce) {
    // (ESKF) step1: 名义状态下imu积分递推
    Eigen::Matrix<double, 24, 1> f_ = x_.get_f(gyro, acce);  // 调用get_f 获取 速度 角速度 加速度
    Eigen::Matrix<double, 24, 23> f_x_ = x_.df_dx(acce);     // 名义变量对状态量的求导

    Eigen::Matrix<double, 24, 12> f_w_ = x_.df_dw();  //
    Eigen::Matrix<double, 23, process_noise_dim_> f_w_final;

    NavState x_before = x_;
    x_.oplus(f_, dt);

    F_x1_ = CovType::Identity();

    // set f_x_final
    CovType f_x_final;  // 23x23
    for (auto st : x_.vect_states_) {
        int idx = st.idx_;
        int dim = st.dim_;
        int dof = st.dof_;

        for (int i = 0; i < 23; i++) {
            for (int j = 0; j < dof; j++) {
                f_x_final(idx + j, i) = f_x_(dim + j, i);
            }
        }

        for (int i = 0; i < process_noise_dim_; i++) {
            for (int j = 0; j < dof; j++) {
                f_w_final(idx + j, i) = f_w_(dim + j, i);
            }
        }
    }

    Mat3d res_temp_SO3;
    Vec3d seg_SO3;
    for (auto st : x_.SO3_states_) {
        int idx = st.idx_;
        int dim = st.dim_;
        for (int i = 0; i < 3; i++) {
            seg_SO3(i) = -1 * f_(dim + i) * dt;
        }

        F_x1_.block<3, 3>(idx, idx) = math::exp(seg_SO3, 0.5).matrix();

        res_temp_SO3 = math::A_matrix(seg_SO3);
        // 什么时候会用这个template呢
        /**
         * c++ .template消除歧义符
         * 为什么使用：编译器在编译代码时，不知道成员函数为模板函数，会解析错误
         * 使用环境：当访问一个对象的成员函数时，而这个对象的类型是不确定的，而且要调用的这个成员函数本身也是模板函数（在模板函数里调用另一个模板函数）
         *
         */
        for (int i = 0; i < state_dim_; i++) {
            f_x_final.template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_.block<3, 1>(dim, i));
        }

        for (int i = 0; i < process_noise_dim_; i++) {
            f_w_final.template block<3, 1>(idx, i) = res_temp_SO3 * (f_w_.block<3, 1>(dim, i));
        }
    }

    // 关于重力向量的变化，关于这方面的推导，公式中未提及（应该是高翔自己加的），后面自己推导一下
    Eigen::Matrix<double, 2, 3> res_temp_S2;
    Vec3d seg_S2;
    for (auto st : x_.S2_states_) {
        int idx = st.idx_;
        int dim = st.dim_;
        for (int i = 0; i < 3; i++) {
            seg_S2(i) = f_(dim + i) * dt;
        }

        SO3 res = math::exp(seg_S2, 0.5f);

        Vec2d vec = Vec2d::Zero();
        Eigen::Matrix<double, 2, 3> Nx = x_.grav_.S2_Nx_yy();
        Eigen::Matrix<double, 3, 2> Mx = x_before.grav_.S2_Mx(vec);

        F_x1_.block<2, 2>(idx, idx) = Nx * res.matrix() * Mx;

        Eigen::Matrix<double, 3, 3> x_before_hat = x_before.grav_.S2_hat();
        res_temp_S2 = -Nx * res.matrix() * x_before_hat * math::A_matrix(seg_S2).transpose();

        for (int i = 0; i < state_dim_; i++) {
            f_x_final.block<2, 1>(idx, i) = res_temp_S2 * (f_x_.block<3, 1>(dim, i));
        }
        for (int i = 0; i < process_noise_dim_; i++) {
            f_w_final.block<2, 1>(idx, i) = res_temp_S2 * (f_w_.block<3, 1>(dim, i));
        }
    }

    F_x1_ += f_x_final * dt;
    P_ = (F_x1_)*P_ * (F_x1_).transpose() +
         (dt * f_w_final) * Q * (dt * f_w_final).transpose();  // 对应fast_lio中预测的公式
}

/**
 * 原版的迭代过程中，收敛次数大于1才会结果，所以需要两次收敛。
 * 在未收敛时，实际上不会计算最近邻，也就回避了一次ObsModel的计算
 * 如果这边对每次迭代都计算最近邻的话，时间明显会变长一些，并不是非常合理。。
 *
 * @param obs
 * @param R
 */
void ESKF::Update(ESKF::ObsType obs, const double& R) {
    custom_obs_model_.valid_ = true;
    custom_obs_model_.converge_ = true;

    CovType P_propagated = P_;

    Eigen::Matrix<double, 23, 1> K_r;
    Eigen::Matrix<double, 23, 23> K_H;

    StateVecType dx_current = StateVecType::Zero();  // 本轮迭代的dx

    NavState start_x = x_;  // 迭代的起点
    NavState last_x = x_;

    int converged_times = 0;
    double last_lidar_res = 0;

    double init_res = 0.0;
    static double iterated_num = 0;
    static double update_num = 0;
    update_num += 1;
    for (int i = -1; i < maximum_iter_; i++) {
        custom_obs_model_.valid_ = true;

        /// 计算observation function，主要是residual_, h_x_, s_
        /// x_ 在每次迭代中都是更新的，线性化点也会更新
        if (obs == ObsType::LIDAR || obs == ObsType::WHEEL_SPEED_AND_LIDAR) {
            lidar_obs_func_(x_, custom_obs_model_);
        } else if (obs == ObsType::WHEEL_SPEED) {
            wheelspeed_obs_func_(x_, custom_obs_model_);
        } else if (obs == ObsType::ACC_AS_GRAVITY) {
            acc_as_gravity_obs_func_(x_, custom_obs_model_);
        } else if (obs == ObsType::GPS) {
            gps_obs_func_(x_, custom_obs_model_);
        } else if (obs == ObsType::BIAS) {
            bias_obs_func_(x_, custom_obs_model_);
        }

        if (use_aa_ && i > -1 && (obs == ObsType::LIDAR || obs == ObsType::WHEEL_SPEED_AND_LIDAR) &&
            custom_obs_model_.lidar_residual_mean_ >= last_lidar_res * 1.01) {
            x_ = last_x;
            break;
        }
        iterated_num += 1;

        if (!custom_obs_model_.valid_) {
            continue;
        }

        if (i == -1) {
            init_res = custom_obs_model_.lidar_residual_mean_;
            if (init_res < 1e-9) {
                init_res = 1e-9;  // 可能有零
            }
        }

        iterations_ = i + 2;  // i从-1开始计
        final_res_ = custom_obs_model_.lidar_residual_mean_ / init_res;

        int dof_measurement = custom_obs_model_.h_x_.rows();
        StateVecType dx = x_.boxminus(start_x);  // 当前x与起点之间的dx
        dx_current = dx;                         //

        P_ = P_propagated;

        /// 更新P 和 dx
        /// P = J*P*J^T（对应自动驾驶与机器人中的SLAM技术中的公式8.6），
        /// 其中J是增量值的右雅克比矩阵，详看自动驾驶与机器人中的SLAM技术中的3.61及手动推导部分
        /// dx = J * dx 这个对应fast_lio课件中的公式19 J.inv * dx
        for (auto it : x_.SO3_states_) {
            int idx = it.idx_;
            Vec3d seg_SO3 = dx.block<3, 1>(idx, 0);
            Mat3d res_temp_SO3 = math::A_matrix(seg_SO3).transpose();  // 小块的J阵, SO3上的雅可比？

            dx_current.block<3, 1>(idx, 0) = res_temp_SO3 * dx.block<3, 1>(idx, 0);

            /// P 上面有SO3的行 进行转换
            for (int j = 0; j < state_dim_; j++) {
                P_.block<3, 1>(idx, j) = res_temp_SO3 * (P_.block<3, 1>(idx, j));
            }
            /// P 上面有SO3的列 进行转换
            for (int j = 0; j < state_dim_; j++) {
                P_.block<1, 3>(j, idx) = (P_.block<1, 3>(j, idx)) * res_temp_SO3.transpose();
            }
        }

        for (auto it : x_.S2_states_) {
            int idx = it.idx_;

            Vec2d seg_S2 = dx.block<2, 1>(idx, 0);

            Eigen::Matrix<double, 2, 3> Nx = x_.grav_.S2_Nx_yy();
            Eigen::Matrix<double, 3, 2> Mx = start_x.grav_.S2_Mx(seg_S2);
            Mat2d res_temp_S2 = Nx * Mx;

            dx_current.block<2, 1>(idx, 0) = res_temp_S2 * dx.block<2, 1>(idx, 0);

            for (int j = 0; j < state_dim_; j++) {
                P_.block<2, 1>(idx, j) = res_temp_S2 * (P_.block<2, 1>(idx, j));
            }

            for (int j = 0; j < state_dim_; j++) {
                P_.block<1, 2>(j, idx) = (P_.block<1, 2>(j, idx)) * res_temp_S2.transpose();
            }
        }

        /// 处理各类观测模型
        if (state_dim_ > dof_measurement) {
            // 这里使用最传统的ESKF中增益计算方式
            Eigen::MatrixXd h_x_cur = Eigen::MatrixXd::Zero(dof_measurement, state_dim_);
            h_x_cur.topLeftCorner(dof_measurement, 12) = custom_obs_model_.h_x_;
            custom_obs_model_.R_ = R * Eigen::MatrixXd::Identity(dof_measurement, dof_measurement);

            // NOTE: 常规ESKF的增益计算（自动驾驶与机器人中的SLAM技术公式3.5.1a 和 fast_lio课件公式18
            Eigen::MatrixXd K =
                P_ * h_x_cur.transpose() * (h_x_cur * P_ * h_x_cur.transpose() + custom_obs_model_.R_).inverse();
            K_r = K * custom_obs_model_.residual_;
            K_H = K * h_x_cur;
        } else {
            /// 这里是针对将雷达误差迭代过程融入IESKF中后，误差函数维度与激光点相关，维度过高计算复杂而优化的ESKF增益计算
            /// 这里的K对应fast_lio课件中的公式36
            /// 纯雷达观测
            double R_inv = 1.0 / (R * dof_measurement);

            // HTRH = H^T R^-1 H
            Eigen::Matrix<double, 12, 12> HTH = custom_obs_model_.h_x_.transpose() * custom_obs_model_.h_x_;

            CovType P_temp = (P_ / R).inverse();  // P阵上面已经更新
            P_temp.block<12, 12>(0, 0) += HTH;    // Q in (38) 对应公式36中的(H.Trans * R.inv * H + P.inv)
            CovType Q_inv = P_temp.inverse();     // Q inv

            // Q*H^T * R^-1 * r = K * r
            // <-- K ----->
            K_r = Q_inv.template block<23, 12>(0, 0) * custom_obs_model_.h_x_.transpose() * custom_obs_model_.residual_;

            // K_H = Q^-1 H^T R^-1 H
            //       <--  K     ->
            K_H.setZero();
            K_H.template block<23, 12>(0, 0) = Q_inv.template block<23, 12>(0, 0) * HTH;
        }

        // dx = Kr + (KH-I) dx
        dx_current = K_r + (K_H - Eigen::Matrix<double, 23, 23>::Identity()) *
                               dx_current;  // fast_lio中的公式35 不知道为啥差个负号

        // check nan
        for (int j = 0; j < 23; ++j) {
            if (std::isnan(dx_current(j, 0))) {
                return;
            }
        }

        if (!use_aa_) {
            x_ = x_.boxplus(dx_current);
        } else {
            // 转到起点的线性空间
            x_ = x_.boxplus(dx_current);

            if (i == -1) {
                aa_.init(dx_current);  // 初始化AA
            } else {
                // 利用AA计算dx from start
                auto dx_all = x_.boxminus(start_x);
                auto new_dx_all = aa_.compute(dx_all);
                x_ = start_x.boxplus(new_dx_all);
            }
        }

        last_x = x_;

        // update last res
        last_lidar_res = custom_obs_model_.lidar_residual_mean_;
        custom_obs_model_.converge_ = true;

        for (int j = 0; j < 23; j++) {
            if (std::fabs(dx_current[j]) > limit_[j]) {
                custom_obs_model_.converge_ = false;
                break;
            }
        }

        if (custom_obs_model_.converge_) {
            converged_times++;
        }

        if (!converged_times && i == maximum_iter_ - 2) {
            custom_obs_model_.converge_ = true;
        }

        if (converged_times > 0 || i == maximum_iter_ - 1) {
            /// 结束条件：已经收敛
            /// 更新P阵, using (45)
            L_ = P_;
            Mat3d res_temp_SO3;
            Vec3d seg_SO3;
            for (auto it : x_.SO3_states_) {
                int idx = it.idx_;
                for (int j = 0; j < 3; j++) {
                    seg_SO3(j) = dx_current(j + idx);
                }

                res_temp_SO3 = math::A_matrix(seg_SO3).transpose();
                for (int j = 0; j < 23; j++) {
                    // block<3, 1>(idx, j) 指从idx, j开始截取3行1列
                    L_.block<3, 1>(idx, j) = res_temp_SO3 * (P_.block<3, 1>(idx, j));
                }

                for (int j = 0; j < 15; j++) {
                    K_H.block<3, 1>(idx, j) = res_temp_SO3 * (K_H.block<3, 1>(idx, j));
                }

                for (int j = 0; j < 23; j++) {
                    L_.block<1, 3>(j, idx) = (L_.block<1, 3>(j, idx)) * res_temp_SO3.transpose();
                    P_.block<1, 3>(j, idx) = (P_.block<1, 3>(j, idx)) * res_temp_SO3.transpose();
                }
            }

            Mat2d res_temp_S2;
            Vec2d seg_S2;
            for (auto it : x_.S2_states_) {
                int idx = it.idx_;

                for (int j = 0; j < 2; j++) {
                    seg_S2(j) = dx_current(j + idx);
                }

                Eigen::Matrix<double, 2, 3> Nx = x_.grav_.S2_Nx_yy();
                Eigen::Matrix<double, 3, 2> Mx = start_x.grav_.S2_Mx(seg_S2);
                res_temp_S2 = Nx * Mx;

                for (auto j = 0; j < 23; j++) {
                    L_.block<2, 1>(idx, j) = res_temp_S2 * (P_.block<2, 1>(idx, j));
                }

                for (auto j = 0; j < 15; j++) {
                    K_H.block<2, 1>(idx, j) = res_temp_S2 * (K_H.block<2, 1>(idx, j));
                }

                for (int j = 0; j < 23; j++) {
                    L_.block<1, 2>(j, idx) = (L_.block<1, 2>(j, idx)) * res_temp_S2.transpose();
                    P_.block<1, 2>(j, idx) = (P_.block<1, 2>(j, idx)) * res_temp_S2.transpose();
                }
            }

            // 与自动驾驶与机器人中的SLAM技术中8.8完全一致 P_k+1 = (I - K_k·H_k)J_k · P_pred · J_k.trans
            // 公式展开为 J_k · P_pred · J_k.trans - K_k·H_k·J_k · P_pred · J_k.trans
            // 这里L_是J_k · P_pred · J_k.trans
            // 这里P_是P_pred · J_k.trans
            // 下式中的K_H已做了处理，为K_k·H_k·J_k
            P_ = L_ - K_H.block<23, 15>(0, 0) * P_.template block<15, 23>(0, 0);

            break;
        }
    }
}

}  // namespace lightning
//
// Created by xiang on 2022/2/15.
//
#pragma once

#include "core/lightning_math.hpp"

namespace lightning {

/**
 * sphere 2
 *
 * 知识补充：将三维流形的变换转换到二维切平面
 * 为什么不把g作为一个普通的三维向量去优化呢？
 * 答：重力的模长时固定的，若用普通的加法去更新，模长大概率就不等于9.81
 * 简单理解：现在这个类主要用于表征重力向量，重力向量的大小是死的，就是9.8，现在就只需要给出位置即可，因为方向是向心的，比如地球仪给出经纬度就可以定出来
 *
 *
 * 为什么要用切平面这个东西呢？
 * 答：球面（流形）不是平的，不能直接用线性代数（加减法）。但在球面上任意一点，都切着一个平面的纸（切空间）。我们在纸上画箭头（2D），然后把纸再贴回球面上，进而对应球面的运动
 *
 *
 *
 * 基底（矩阵）坐标系S2_BX：矩阵中两个列向量为两个正交对
 * 广义加用于快速计算重力向量的偏移（比如人上坡，重力偏）
 * 广义减用于计算重力向量的偏差
 * @todo S2的长度应当可以外部指定
 */
struct S2 {
    using SO3 = Sophus::SO3d;

    static constexpr int den_ = 98090;
    static constexpr int num_ = 10000;
    static constexpr double length_ = double(den_) / double(num_);
    Vec3d vec_;

   public:
    S2() { vec_ = length_ * Vec3d(1, 0, 0); }
    S2(const Vec3d &vec) : vec_(vec) {
        vec_.normalize();
        vec_ = vec_ * length_;
    }

    double operator[](int idx) const { return vec_[idx]; }

    // 反对称矩阵
    // 现有向量[a, b, c].transpose
    // 反对称矩阵
    /**
     *  | 0, -c,  b |
     *  | c,  0, -a |
     *  | -b, a,  0 |
     */
    Eigen::Matrix<double, 3, 3> S2_hat() const {
        Eigen::Matrix<double, 3, 3> skew_vec;
        skew_vec << double(0), -vec_[2], vec_[1], vec_[2], double(0), -vec_[0], -vec_[1], vec_[0], double(0);
        return skew_vec;
    }

    /**
     * S2_Mx: the partial derivative of x.boxplus(u) w.r.t. u
     */
    // 数学公式推导：
    // 恒等式公式：R·x^·R.Trans() = [Rx]^
    //            R·x^ = [Rx]^·R
    // 对应：如果输入量delta变大一点点，输出的状态vec_会怎么变

    /**
     *  当△较小时可以直接用泰勒展开
     *
     *  这个旋转向量就是李代数
     *  关于雅克比矩阵的推导过程（扰动法）
     *
     *  给出计算公式：f(u) = exp(u^)x  式中u为旋转向量
     *  step1: f(u + △) = exp[(u+△)^]x
     *
     *
     *
     *
     *
     *
     *
     *
     */
    Eigen::Matrix<double, 3, 2> S2_Mx(const Eigen::Matrix<double, 2, 1> &delta) const {
        Eigen::Matrix<double, 3, 2> res;
        Eigen::Matrix<double, 3, 2> Bx = S2_Bx();

        if (delta.norm() < 1e-5) {
            res = -SO3::hat(vec_) * Bx;
        } else {
            Vec3d Bu = Bx * delta;
            SO3 exp_delta = math::exp(Bu, 0.5f);
            /**
             * Derivation of d(Exp(Bx dx)x)/d(dx)=d(Exp(Bu)x)/d(dx):
             *  d(Exp(Bu)x)/d(dx)=d(Exp(Bu)x)/d(Bu) Bx; then
             *  d(Exp(Bu)x)/d(Bu)=d[Exp(Bu+dBu)x]/d(dBu)=d[Exp(Jl(Bu)dBu)Exp(Bu)x]/d(dBu)=d[(Jl(Bu)dBu)^Exp(Bu)x]/d(dBu)
             *   =d[-(Exp(Bu)x)^Jl(Bu)dBu]/d(dBu)=-Exp(Bu)x^Exp(-Bu)Jl(Bu);
             *    for Exp(x+dx)=Exp(x)Exp(Jr(x)dx)=Exp(Jl(x)dx)Exp(x)=Exp(x)Exp(Exp(-x)Jl(x)dx) =>
             *    Exp(-x)Jl(x)=Jr(x)=Jl(-x) =>
             *   =-Exp(Bu)x^Jl(-Bu) => d(Exp(Bu)x)/d(dx)= -Exp(Bu) x^ Jl(Bu)^T Bx or A_matrix is just Jl()
             */
            res = -exp_delta.matrix() * SO3::hat(vec_) * math::A_matrix(Bu).transpose() * Bx;
        }
        return res;
    }

    /**
     * Bx两个列向量为正切空间的局部坐标系
     *
     * S2 = [a b c], l = length
     * Bx = [
     * -b              -c
     * l-bb/(l+a)      -bc/(l+a)
     * -bc/(l+a)        l-cc/(l+a)
     * ] / l
     * Derivation of origin MTK: (112) Rv = [v Bx] = [x -r 0; y xc -s; z xs c]
     *  where c=cos(alpha),s=sin(alpha),r=sqrt(y^2+z^2),||v||=1:
     *  For y-axis or (0,1,0) in local frame is mapped to (-r;xc;xs) in world frame,
     *  meaning x/OG(or gravity vector) projects to A of yz plane,
     *  and this projecting line will intersect the perpendicular plane of OG at O at the point B,
     *  then from OB we can found OC with ||OC||=1, which is just the y-axis, then we get its coordinate in world frame:
     *  ∠AOG+∠AGO=pi/2=∠AOG+∠AOB => ∠AOB=∠AGO, for sin(∠AGO)=r/||v||=r => ||OC||*sin(∠AOB)=r;||OC||*cos(∠AOB)=x =>
     *  (-r;xc;xs) is the y-axis coordinate in world frame, then z-axis is easy to get
     * Derivation of current MTK with S2 = [x y z], ||S2|| = 1:
     *  just a rotation of origin one around x/OG axis to meet y-axis coordinate in word frame to be (-y;a;c),
     *  then z-axis must be (+-z;b;d) for x^+y^+z^=1, where a,b,c,d is just variable to be solved
     *  for current MTK chooses clockwise rotation(meaning -z):
     *  Rv=[x -y -z;
     *      y  a  b;
     *      z  c  d]
     *  then for -xy+ya+zc=0=xy-ya-zb => b=c
     *  for yz+ab+cd=0; b=c => a=-d-yz/c; for -xz+yb+zd=0; b=c => d=x-yc/z
     *  for -xy+ya+zc=0; a=-d-yz/c=yc/z-x-yz/c => -xy+y^2c/z-xy-y^2z/c+zc=0 => (z+y^2/z)c^2 -2xy c -y^2z = 0 =>
     *  c=[2xy +- sqrt(4x^2y^2 + 4(z+y^2/z)y^2z)]/[2(z+y^2/z)]; for z^2+y^2=1-x^2 =>
     *  c=[xy +- y]z/(1-x^2), for rotation is clocewise, thus c<0 => c=(xy-y)z/(1-x^2)=-yz/(1+x)
     *  then b,a,d is easy to get and also if ||S2||=l, it is easy to prove c=-ylzl/(l+lx)/l=-bc/(l+a)/l
     */

    // 切空间基底矩阵
    // 计算公式是如何处理的呢？
    // 罗德里格斯公式：R = I + sin(theta)K + (1-cos(theta))K^2  式中I为单位阵，K为旋转轴的反对称矩阵，theta为旋转角
    // 这是有旋转轴和旋转角，如果仅有当前向量u和目标向量v的话，怎么处理
    // a、制造旋转轴，垂直于两向量的向量就是w = u x v ，同时u和v均为单位向量，且u x v = sin(theta)，故单位旋转轴k =
    // w/sin（theta) b、找旋转角，u · v = cos(theta) c、代入R = I + [w]x + 1/(1 + u·v)[w]x^2  其中[w]x是反对称矩阵
    // step1: 首先将（1,0,0）转换至当前vec_，进而求得R
    // step2：R再乘（0,1,0）和（0,0,1）就能求出来啦
    //
    Eigen::Matrix<double, 3, 2> S2_Bx() const {
        Eigen::Matrix<double, 3, 2> res;
        if (vec_[0] + length_ > 1e-5) {
            res << -vec_[1], -vec_[2], length_ - vec_[1] * vec_[1] / (length_ + vec_[0]),
                -vec_[2] * vec_[1] / (length_ + vec_[0]), -vec_[2] * vec_[1] / (length_ + vec_[0]),
                length_ - vec_[2] * vec_[2] / (length_ + vec_[0]);
            res /= length_;
        } else {
            // 奇异点：当向量完全指向X轴的负半轴时（即加速度大于-9.81时），此时当前重力向量为[-L, 0,
            // 0]，切平面一定在YZ平面上 为什么是（0,0,1）和（0，-1,0）而不能是（0,0,1）和（0,1,0）
            // 为保证数学上的连续性，选择了这个
            res = Eigen::Matrix<double, 3, 2>::Zero();
            res(1, 1) = -1;
            res(2, 0) = 1;
        }
        return res;
    }

    /**
     * S2_Nx: the partial derivative of x.boxminus(y) w.r.t. x, where x and y belong to S2
     * S2_Nx_yy: simplified S2_Nx when x is equal to y
     */
    // 计算误差函数相对于状态的雅克比
    Eigen::Matrix<double, 2, 3> S2_Nx_yy() const {
        Eigen::Matrix<double, 2, 3> res;
        Eigen::Matrix<double, 3, 2> Bx = S2_Bx();
        res = 1 / length_ / length_ * Bx.transpose() * SO3::hat(vec_);
        return res;
    }

    void oplus(const Vec3d &delta, double scale = 1.0) { vec_ = math::exp(delta, scale * 0.5) * vec_; }

    /**
     * 广义减
     * @param res
     * @param other
     */
    Vec2d boxminus(const S2 &other) const {
        Vec2d res;
        // hat是生成反对称矩阵，vee是反对称到向量
        double v_sin = (SO3::hat(vec_) * other.vec_).norm();  // 叉乘
        double v_cos = vec_.transpose() * other.vec_;         // 点乘
        double theta = std::atan2(v_sin, v_cos);
        if (v_sin < 1e-5) {
            if (std::fabs(theta) > 1e-5) {
                res[0] = 3.1415926;
                res[1] = 0;
            } else {
                res[0] = 0;
                res[1] = 0;
            }
        } else {
            // 如何求两个向量之间的角度差呢(比如重力变化后的)
            // 两个单位向量 u 和 v
            // step1: 找到旋转轴 u叉乘v，旋转轴标准化（即模为1），即旋转轴为 (u叉乘v)/sin
            // step2: 找到旋转角 theta = atan2(sin, cos), sin = （u 叉乘 v）/(|u| * |v|)  cos = (u 点乘 v)/(|u| * |v|)
            // 进而求得旋转向量
            // step3: 已知旋转向量w = Bx * 增量
            // step4: 进而求得增量 = Bx.Transpose() * w
            // step5: 已知旋转向量 w 由步骤1和2求得
            S2 other_copy = other;
            Eigen::Matrix<double, 3, 2> Bx = other_copy.S2_Bx();
            res = theta / v_sin * Bx.transpose() * SO3::hat(other.vec_) * vec_;
        }
        return res;
    }

    /**
     * 广义加
     * @param delta
     * @param scale
     */
    // 广义加有什么作用呢，x广义加delta = exp([Bx · delta]_x)·x
    // 输入：一个2D的增量delta（在切平面上）
    // 升维：用基底矩阵Bx将2D增量映射为3D向量Bx·delta。这个3D向量与当前状态vec_垂直
    // 指数映射：将这个垂直向量视为旋转向量（轴角），通过exp转换为旋转矩阵res
    // 作用：用旋转矩阵旋转当前vec_得到新的位置。因为是旋转，模长保持不变，仍在球面上
    void boxplus(const Vec2d &delta, double scale = 1) {
        Eigen::Matrix<double, 3, 2> Bx = S2_Bx();
        SO3 res = math::exp(
            Bx * delta,
            scale /
                2);  // 计算旋转,其实Bx *
                     // delta生成的是旋转向量（与旋转矩阵同等存在的一种表征方式，其中向量方向表征旋转轴，向量大小表征旋转角度）
        // 同样产生疑问，为什么Bx *
        // delta是旋转向量呢，其中Bx为两个基准轴（模为1，正交对），此时delta就像是权重，用于分配这个旋转轴朝哪，大小是多少
        vec_ = res.matrix() * vec_;  // 旋转原向量
    }
};

}  // namespace lightning

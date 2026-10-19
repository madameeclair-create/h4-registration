// 独立开发实验：相同候选缓存；不读取真值。原版函数来自冻结静态库。
#include <pcl/common/transforms.h>
#include <Eigen/SVD>
#include "Eva.h"
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
bool add_overlap=false,low_inlieratio=false,no_logs=false;
std::string folderPath;
using Pose=Eigen::Matrix4d;
const std::array<double,5> scales{{.3,.45,.6,.9,1.2}};
struct Outcome { Pose pose; double score; int attempts=0,accepted=0; std::string status="iteration_limit"; };

// 与原代码相同：PCL存储float点，变换后用double计算距离。
std::vector<double> residuals(PointCloudPtr s,PointCloudPtr t,const Pose& p) {
    pcl::PointCloud<pcl::PointXYZ> transformed;
    pcl::transformPointCloud(*s,transformed,p);
    std::vector<double> r(s->size());
    for(size_t i=0;i<r.size();++i) r[i]=Distance(transformed[i],(*t)[i]);
    return r;
}
// 【对应点评分｜论文式8】累加max(0,1-r/tau)，值越大支持越强；不是直接最小化平均误差。
double score(const std::vector<double>& r,double tau=.6) {
    double q=0;for(double d:r) if(d<tau) q+=(tau-d)/tau;return q;
}

// KITTI精化使用0.60m门限，长度尺度统一为室内的6倍；加权质心与接受逻辑不变。
// 【加权SVD精化｜论文式12—14】残差产生权重，计算加权质心及协方差，再求旋转和平移。
// 属于基于已有求解思想的精化实现；本文核心创新在Python中的DS、BG/NC/DB和MS。
bool fit_weighted(PointCloudPtr s,PointCloudPtr t,const std::vector<double>& r,Pose& result) {
    Eigen::Vector3d ms=Eigen::Vector3d::Zero(),mt=ms;double sum=0;int n=0;
    for(size_t i=0;i<r.size();++i) if(r[i]<.6) {
        double w=1/(1+std::pow(r[i]/.6,2));
        ms+=w*(*s)[i].getVector3fMap().cast<double>();
        mt+=w*(*t)[i].getVector3fMap().cast<double>();sum+=w;++n;
    }
    if(n<3 || !std::isfinite(sum) || sum<=0)return false;
    ms/=sum;mt/=sum;Eigen::Matrix3d h=Eigen::Matrix3d::Zero();
    for(size_t i=0;i<r.size();++i) if(r[i]<.6) {
        double w=1/(1+std::pow(r[i]/.6,2));
        h+=w*((*s)[i].getVector3fMap().cast<double>()-ms)*
              ((*t)[i].getVector3fMap().cast<double>()-mt).transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(h,Eigen::ComputeFullU|Eigen::ComputeFullV);
    // 排除共线退化；非共线平面允许刚体估计，不要求三维满秩。
    if(!svd.singularValues().allFinite() || svd.singularValues()[0]<=0 ||
       svd.singularValues()[1]<=1e-12*svd.singularValues()[0])return false;
    Eigen::Matrix3d correction=Eigen::Matrix3d::Identity();
    correction(2,2)=(svd.matrixV()*svd.matrixU().transpose()).determinant()<0?-1:1;
    result=Pose::Identity();result.topLeftCorner<3,3>()=svd.matrixV()*correction*svd.matrixU().transpose();
    result.topRightCorner<3,1>()=mt-result.topLeftCorner<3,3>()*ms;
    return result.allFinite();
}
// 【控制对照】旧MAC的加权SVD路径，用于数值对照，不是终稿完整方法的新模块。
bool fit_old(PointCloudPtr s,PointCloudPtr t,const std::vector<double>& r,Pose& p) {
    PointCloudPtr a(new pcl::PointCloud<pcl::PointXYZ>),b(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<double> weights;
    for(size_t i=0;i<r.size();++i)if(r[i]<.6){a->push_back((*s)[i]);b->push_back((*t)[i]);weights.push_back(1/(1+std::pow(r[i]/.6,2)));}
    if(weights.size()<3)return false;
    Eigen::VectorXd w=Eigen::Map<Eigen::VectorXd>(weights.data(),weights.size());
    weight_SVD(a,b,w,0,p);return p.allFinite();
}
// 【单调接受｜论文式15】最多20轮；不能提高对应点评分则拒绝新位姿并停止。
Outcome refine(PointCloudPtr s,PointCloudPtr t,Pose p,bool centroid,bool rollback) {
    Outcome out;out.pose=p;out.score=score(residuals(s,t,p));double pre=out.score;
    for(int it=0;it<20;++it) {
        auto r=residuals(s,t,out.pose);double current=score(r);
        if(!rollback && current<pre){out.status="old_break_without_rollback";break;}
        pre=current;Pose trial;out.attempts++;
        bool valid=centroid?fit_weighted(s,t,r,trial):fit_old(s,t,r,trial);
        if(!valid){out.status="insufficient_or_degenerate";break;}
        double trial_score=score(residuals(s,t,trial));
        if(rollback && trial_score<=out.score){out.status=trial_score<out.score?"rejected_score_drop":"score_plateau";break;}
        out.pose=trial;out.score=trial_score;out.accepted++;
    }
    out.score=score(residuals(s,t,out.pose));return out;
}
void write_pose(const std::string& path,const Pose& p){std::ofstream f(path);f<<std::setprecision(17)<<p<<"\n";}


void load_corr(const std::string& path,PointCloudPtr& s,PointCloudPtr& t){
    s.reset(new pcl::PointCloud<pcl::PointXYZ>);t.reset(new pcl::PointCloud<pcl::PointXYZ>);
    std::ifstream f(path);float x,y,z,a,b,c;
    while(f>>x>>y>>z>>a>>b>>c){s->push_back(pcl::PointXYZ(x,y,z));t->push_back(pcl::PointXYZ(a,b,c));}
    if(!f.eof() || s->size()<3)throw std::runtime_error("invalid correspondence file");
}
// 从已有候选缓存导出评分，不改变MAC的图、团或候选生成。
void export_scores(const std::string& id,const std::string& cachepath,const std::string& corrpath,const std::string& output){
    PointCloudPtr s,t;load_corr(corrpath,s,t);
    std::ifstream cache(cachepath,std::ios::binary);char magic[8];uint64_t n;
    cache.read(magic,8);cache.read(reinterpret_cast<char*>(&n),8);
    if(!cache || std::string(magic,8)!="MACPOSE1" || n>10000000)throw std::runtime_error("bad cache");
    std::ofstream out(output+"/"+id+"_scores.csv");out<<std::setprecision(17)<<"candidate_id,finite,q0,q1,q2,q3,q4\n";
    for(uint64_t h=0;h<n;++h){Pose p;
        for(int row=0;row<4;++row)for(int col=0;col<4;++col)cache.read(reinterpret_cast<char*>(&p(row,col)),8);
        if(!cache)throw std::runtime_error("truncated pose");
        out<<h<<","<<p.allFinite();
        if(p.allFinite()){auto r=residuals(s,t,p);for(double tau:scales)out<<","<<score(r,tau);}
        else for(int j=0;j<5;++j)out<<",nan";
        out<<"\n";
    }
}
// 命令行仅有score和refine两种工具模式，由Python入口组织调用，不需要人工运行此文件。
int main(int argc,char** argv){
    if(argc!=4)return 2;std::string mode=argv[1],output=argv[3];std::filesystem::create_directories(output);
    std::ifstream input(argv[2]);std::string line,last;PointCloudPtr s,t;bool failed=false;
    std::ofstream raw(output+"/tool_results.csv");raw<<std::setprecision(17)<<"id,status,initial_score,final_score,attempts,accepted,termination\n";
    while(std::getline(input,line)){if(!line.empty() && line.back()=='\r')line.pop_back();
        std::stringstream row(line);std::string id,a,b;std::getline(row,id,'\t');std::getline(row,a,'\t');std::getline(row,b,'\t');
        try{
            if(mode=="score"){export_scores(id,a,b,output);raw<<id<<",completed,0,0,0,0,score_export\n";}
            else if(mode=="refine"){
                if(a!=last){load_corr(a,s,t);last=a;}
                Pose p;std::ifstream f(b);for(int r=0;r<4;++r)for(int c=0;c<4;++c)f>>p(r,c);
                if(!f || !p.allFinite())throw std::runtime_error("invalid input pose");
                double pre=score(residuals(s,t,p));auto result=refine(s,t,p,true,true);
                write_pose(output+"/"+id+"_final.txt",result.pose);
                raw<<id<<",completed,"<<pre<<","<<result.score<<","<<result.attempts<<","<<result.accepted<<","<<result.status<<"\n";
            }else throw std::runtime_error("unknown mode");
        }catch(const std::exception& e){std::cerr<<"FAILED "<<id<<" "<<e.what()<<std::endl;raw<<id<<",failed,0,0,0,0,see_log\n";failed=true;}
        raw.flush();std::cout<<"TOOL "<<id<<std::endl;
    }
    return failed?1:0;
}

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
const std::array<double,5> scales{{.05,.075,.1,.15,.2}};
struct Outcome { Pose pose; double score; int attempts=0,accepted=0; std::string status="iteration_limit"; };

// 与原代码相同：PCL存储float点，变换后用double计算距离。
std::vector<double> residuals(PointCloudPtr s,PointCloudPtr t,const Pose& p) {
    pcl::PointCloud<pcl::PointXYZ> transformed;
    pcl::transformPointCloud(*s,transformed,p);
    std::vector<double> r(s->size());
    for(size_t i=0;i<r.size();++i) r[i]=Distance(transformed[i],(*t)[i]);
    return r;
}
double score(const std::vector<double>& r,double tau=.1) {
    double q=0;for(double d:r) if(d<tau) q+=(tau-d)/tau;return q;
}

// 加权最小二乘必须使用加权质心；仍沿用原残差权重与0.1m门限。
bool fit_weighted(PointCloudPtr s,PointCloudPtr t,const std::vector<double>& r,Pose& result) {
    Eigen::Vector3d ms=Eigen::Vector3d::Zero(),mt=ms;double sum=0;int n=0;
    for(size_t i=0;i<r.size();++i) if(r[i]<.1) {
        double w=1/(1+std::pow(r[i]/.1,2));
        ms+=w*(*s)[i].getVector3fMap().cast<double>();
        mt+=w*(*t)[i].getVector3fMap().cast<double>();sum+=w;++n;
    }
    if(n<3 || !std::isfinite(sum) || sum<=0)return false;
    ms/=sum;mt/=sum;Eigen::Matrix3d h=Eigen::Matrix3d::Zero();
    for(size_t i=0;i<r.size();++i) if(r[i]<.1) {
        double w=1/(1+std::pow(r[i]/.1,2));
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
bool fit_old(PointCloudPtr s,PointCloudPtr t,const std::vector<double>& r,Pose& p) {
    PointCloudPtr a(new pcl::PointCloud<pcl::PointXYZ>),b(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<double> weights;
    for(size_t i=0;i<r.size();++i)if(r[i]<.1){a->push_back((*s)[i]);b->push_back((*t)[i]);weights.push_back(1/(1+std::pow(r[i]/.1,2)));}
    if(weights.size()<3)return false;
    Eigen::VectorXd w=Eigen::Map<Eigen::VectorXd>(weights.data(),weights.size());
    weight_SVD(a,b,w,0,p);return p.allFinite();
}
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

void run_case(const std::string& id,const std::string& cachepath,const std::string& corrpath,
              const std::string& output,std::ofstream& raw) {
    auto started=std::chrono::steady_clock::now();
    PointCloudPtr s(new pcl::PointCloud<pcl::PointXYZ>),t(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<Corre_3DMatch> corr;
    std::ifstream f(corrpath);float x,y,z,a,b,c;
    while(f>>x>>y>>z>>a>>b>>c){Corre_3DMatch q{};q.src=pcl::PointXYZ(x,y,z);q.des=pcl::PointXYZ(a,b,c);q.score=1;corr.push_back(q);s->push_back(q.src);t->push_back(q.des);}
    if(!f.eof() || corr.size()<3)throw std::runtime_error("invalid correspondences");
    std::ifstream cache(cachepath,std::ios::binary);char magic[8];uint64_t n;
    cache.read(magic,8);cache.read(reinterpret_cast<char*>(&n),8);
    if(!cache || std::string(magic,8)!="MACPOSE1" || n>10000000)throw std::runtime_error("bad cache");
    std::vector<Pose> poses;std::vector<std::array<double,5>> values;std::vector<uint64_t> indices;
    uint64_t invalid=0;
    for(uint64_t h=0;h<n;++h){Pose p;
        for(int row=0;row<4;++row)for(int col=0;col<4;++col)cache.read(reinterpret_cast<char*>(&p(row,col)),8);
        if(!cache)throw std::runtime_error("truncated cache");
        if(!p.allFinite()){invalid++;continue;}
        auto r=residuals(s,t,p);std::array<double,5> q;
        for(int j=0;j<5;++j)q[j]=score(r,scales[j]);
        poses.push_back(p);values.push_back(q);indices.push_back(h);
    }
    char extra;if(cache.read(&extra,1))throw std::runtime_error("extra cache bytes");
    std::vector<std::array<double,5>> ranks(poses.size());
    std::vector<size_t> order(poses.size());std::iota(order.begin(),order.end(),0);
    for(int j=0;j<5;++j){
        std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return values[a][j]>values[b][j];});
        for(size_t start=0;start<order.size();){size_t end=start+1;
            while(end<order.size() && values[order[end]][j]==values[order[start]][j])++end;
            double r=order.size()>1?(.5*(start+end-1))/(order.size()-1):0;
            for(size_t k=start;k<end;++k)ranks[order[k]][j]=r;start=end;
        }
    }
    std::array<std::string,7> variants{{"original_replay","centroid_only","rollback_only","fixed_control","mean_score","mean_rank","worst_rank"}};
    for(const auto& variant:variants){
        int selected=-1;double best=0;std::array<double,3> bestkey{{1e100,1e100,1e100}};
        for(size_t h=0;h<poses.size();++h){
            // 所有方法共享有效性规则：原阈值下无任何正支持的候选不选。
            if(values[h][2]<=0)continue;
            if(variant=="mean_score"){
                double v=std::accumulate(values[h].begin(),values[h].end(),0.0)/5;
                if(v>best){best=v;selected=h;}
            }else if(variant=="mean_rank" || variant=="worst_rank"){
                double mean=std::accumulate(ranks[h].begin(),ranks[h].end(),0.0)/5;
                double worst=*std::max_element(ranks[h].begin(),ranks[h].end());
                std::array<double,3> key{{variant=="mean_rank"?mean:worst,mean,-values[h][2]}};
                if(key<bestkey){bestkey=key;selected=h;}
            }else if(values[h][2]>best){best=values[h][2];selected=h;}
        }
        std::string directory=output+"/"+variant;std::filesystem::create_directories(directory);
        if(selected<0){raw<<id<<","<<variant<<",no_hypothesis,-1,"<<n<<","<<invalid<<",0,0,0,0,0,no_hypothesis\n";raw.flush();continue;}
        auto pre=poses[selected];write_pose(directory+"/"+id+"_initial.txt",pre);
        Outcome result;
        if(variant=="original_replay"){
            result.pose=pre;result.score=values[selected][2];
            post_refinement(corr,s,t,result.pose,result.score,.1,20,"MAE");
            result.status="original_status_not_instrumented";
            result.score=score(residuals(s,t,result.pose));
        }else result=refine(s,t,pre,variant!="rollback_only",variant!="centroid_only");
        if(!result.pose.allFinite())throw std::runtime_error("nonfinite final pose");
        write_pose(directory+"/"+id+"_final.txt",result.pose);
        double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        raw<<id<<","<<variant<<",completed,"<<indices[selected]<<","<<n<<","<<invalid<<","<<values[selected][2]<<","<<result.score<<","<<result.attempts<<","<<result.accepted<<","<<sec<<","<<result.status<<"\n";raw.flush();
    }
    std::cout<<"COMPLETED "<<id<<" candidates="<<n<<std::endl;
}
int main(int argc,char** argv){
    if(argc!=3){std::cerr<<"usage: pilot GT-free_input.tsv output\n";return 2;}
    std::filesystem::create_directories(argv[2]);std::ofstream raw(std::string(argv[2])+"/engine_raw.csv");
    raw<<std::setprecision(17)<<"case_id,variant,status,candidate_id,candidate_count,nonfinite_candidates,initial_score,final_score,attempts,accepted,case_elapsed_seconds,termination\n";
    std::ifstream batch(argv[1]);std::string line;bool failed=false;
    while(std::getline(batch,line)){
        if(!line.empty() && line.back()=='\r')line.pop_back(); // Windows清单的CRLF不是路径内容。
        std::stringstream row(line);std::string id,cache,corr;
        std::getline(row,id,'\t');std::getline(row,cache,'\t');std::getline(row,corr,'\t');
        try{run_case(id,cache,corr,argv[2],raw);}catch(const std::exception& e){std::cerr<<"FAILED "<<id<<" "<<e.what()<<std::endl;failed=true;}
    }
    return failed?1:0;
}
